import streamlit as st
import os
import tempfile
import pandas as pd
import json
from pipeline.parser import PDFParser
from pipeline.chunker import TextChunker
from pipeline.embedder import Embedder
from pipeline.retriever import VectorRetriever
from pipeline.extractor import SpecExtractor

# Page Config
st.set_page_config(page_title="Vehicle Spec Extractor", layout="wide")

# Title and Description
st.title("🚗 Vehicle Specification Extractor")
st.markdown("Upload a service manual and extract structured vehicle specifications using RAG.")

# Initialize Session State
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None

# Sidebar - Configuration & Upload
with st.sidebar:
    st.header("Configuration")
    
    # Embedding Provider Selection
    provider = st.selectbox("Embedding Provider", ["huggingface", "openai"], index=0)
    
    # API Key Handling (if not in env) or Retrieval Only Mode
    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" and not api_key:
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            api_key = api_key_input
    
    # Retrieval Settings
    st.markdown("### Search Settings")
    retrieval_only = st.checkbox("Retrieval Only (No LLM)", value=(not api_key))
    top_k = st.slider("Number of Chunks (K)", 1, 20, 5)

    st.header("Upload Manual")
    uploaded_file = st.file_uploader("Upload Service Manual (PDF)", type=["pdf"])
    
    process_btn = st.button("Build Knowledge Base")

# Progress & Processing
if uploaded_file and process_btn:
    with st.spinner("Processing PDF... This may take a moment."):
        try:
            # 1. Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # 2. Pipeline Initialization
            parser = PDFParser()
            chunker = TextChunker()
            embedder = Embedder(provider=provider)
            
            # 3. Execution
            st.text("Parsing PDF...")
            pages_data = parser.extract_text(tmp_path)
            
            st.text(f"Chunking {len(pages_data)} pages...")
            chunks = chunker.chunk_documents(pages_data)
            
            st.text("Building Vector Store...")
            # Initialize Retriever
            retriever_instance = VectorRetriever(embedder.get_embedding_function())
            retriever_instance.build_knowledge_base(chunks)
            
            # Store in session state
            st.session_state.retriever = retriever_instance
            st.session_state.processed_file = uploaded_file.name
            
            # Cleanup
            os.remove(tmp_path)
            
            st.success(f"Knowledge Base Built! Processed {len(chunks)} chunks.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Main Interface
if st.session_state.retriever:
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Query Specifications")
        user_query = st.text_input("Enter vehicle specification query", placeholder="e.g., Torque for brake caliper bolts")
        
        # Sample Prompts
        st.markdown("**Quick Try:**")
        sample_cols = st.columns(4)
        samples = [
            "Torque for brake caliper bolts",
            "Engine oil capacity",
            "Wheel nut torque",
            "Brake pad wear limit"
        ]
        
        selected_sample = None
        for i, sample in enumerate(samples):
            if sample_cols[i].button(sample):
                selected_sample = sample
        
        # Handle Query Execution
        query_to_run = selected_sample if selected_sample else user_query
        
        if query_to_run:
            with st.spinner(f"Retrieving top {top_k} chunks..."):
                try:
                    # Retrieve
                    retrieved_docs = st.session_state.retriever.retrieve(query_to_run, k=top_k)
                    
                    if retrieval_only:
                        st.subheader(f"Top {top_k} Retrieved Chunks")
                        for i, doc in enumerate(retrieved_docs):
                            with st.expander(f"Chunk {i+1} (Page {doc.metadata.get('page')})", expanded=True):
                                st.write(doc.page_content)
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    else:
                        # Extract
                        extractor = SpecExtractor() # Uses ENV key
                        results = extractor.extract(query_to_run, retrieved_docs)
                        
                        # Display Results
                        st.subheader("Extraction Results")
                        
                        # Tabs for different views
                        tab1, tab2, tab3 = st.tabs(["JSON", "Table", "Context (Debug)"])
                        
                        with tab1:
                            st.json(results)
                        
                        with tab2:
                            if results:
                                df = pd.DataFrame(results)
                                st.dataframe(df)
                                
                                # CSV Download
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    "specifications.csv",
                                    "text/csv"
                                )
                            else:
                                st.info("No specifications found.")
                        
                        with tab3:
                            for i, doc in enumerate(retrieved_docs):
                                with st.expander(f"Chunk {i+1} (Page {doc.metadata.get('page')})"):
                                    st.write(doc.page_content)
                                
                except Exception as e:
                    st.error(f"Error during extraction: {e}")

    with col2:
        st.info(f"Active Knowledge Base: {st.session_state.processed_file}")
        # Could add more stats here
else:
    st.info("Please upload a PDF and click 'Build Knowledge Base' to start.")
