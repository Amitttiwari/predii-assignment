import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from typing import List

class VectorRetriever:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vectorstore = None

    def build_knowledge_base(self, documents: List[Document]):
        """
        Builds a FAISS vector store from the provided documents.
        """
        if not documents:
            print("No documents specific to build knowledge base.")
            return

        print(f"Building vector store with {len(documents)} chunks...")
        self.vectorstore = FAISS.from_documents(documents, self.embedding_function)
        print("Vector store built successfully.")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieves top-k relevant documents for a given query.
        """
        if not self.vectorstore:
            raise ValueError("Knowledge base not built. Please upload a PDF and build the knowledge base first.")
        
        return self.vectorstore.similarity_search(query, k=k)

if __name__ == "__main__":
    # Test
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = VectorRetriever(embeddings)
    
    docs = [Document(page_content="The brake caliper torque is 50 Nm.", metadata={"page": 1})]
    retriever.build_knowledge_base(docs)
    
    results = retriever.retrieve("What is the torque?")
    if results:
        print(f"Retrieved: {results[0].page_content}")
