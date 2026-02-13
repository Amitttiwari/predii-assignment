# Vehicle Specification Extraction System

A RAG-based system designed to extract structured vehicle specifications (e.g., torque values, fluid capacities) from service manual PDFs. This project uses a modular pipeline to parse, chunk, embed, and retrieve technical data, presented through an interactive Streamlit interface.

## 🏗️ System Architecture

The system follows a standard Retrieval-Augmented Generation (RAG) pipeline:

### 1. PDF Parsing (`pipeline/parser.py`)
**Library:** `PyMuPDF` (aka `fitz`)
- **Process:** We read the PDF file and iterate through each page.
- **Logic:** We define a crop box to exclude headers and footers (common in manuals) to reduce noise. The raw text is extracted from the remaining area.
- **Output:** A list of dictionaries containing the text and metadata (page number, source file).

### 2. Text Chunking (`pipeline/chunker.py`)
**Library:** `LangChain` (`RecursiveCharacterTextSplitter`)
- **Process:** The extracted text is often too long for a single LLM context window and lacks semantic isolation. We split the text into smaller "chunks".
- **Logic:** We use a chunk size of 600 characters with a 100-character overlap. The "Recursive" splitter tries to break at logical separators (paragraphs `\n\n`, newlines `\n`, spaces) to keep related text together.
- **Example Chunk:**
  ```json
  {
    "page_content": "Torque Specifications\n\nCylinder head bolts: 30 Nm + 90 degrees\nMain bearing caps: 60 Nm\nConnecting rod caps: 40 Nm\n\nEnsure all bolts are oiled before installation.",
    "metadata": {
      "source": "manual.pdf",
      "page": 42
    }
  }
  ```

### 3. Embedding Generation (`pipeline/embedder.py`)
**Libraries:** `sentence-transformers` (HuggingFace) or `OpenAI`
- **Process:** Converting text chunks into vector representations (lists of numbers) that capture semantic meaning.
- **Logic:**
    -   **HuggingFace (Default)**: Uses the `all-MiniLM-L6-v2` model locally to create 384-dimensional vectors. This is free and fast.
    -   **OpenAI (Optional)**: Uses `text-embedding-3-small` if an API key is provided, for potentially higher accuracy.

### 4. Vector Retrieval (`pipeline/retriever.py`)
**Library:** `FAISS` (Facebook AI Similarity Search)
- **Process:** Storing the generated vectors in an index for fast searching.
- **Logic:** When a user asks a query (e.g., "What is the cylinder head torque?"), we convert the query into a vector and use **Cosine Similarity** to find the top `K` (default 5) chunks that are most similar to the query vector.

### 5. Specification Extraction (`pipeline/extractor.py`)
**Library:** `LangChain` + `OpenAI GPT-4o`
- **Process:** Passing the retrieved chunks to a Large Language Model (LLM) to extract specific data.
- **Logic:** We construct a prompt that includes the user's query and the retrieved context. The LLM is instructed to:
    1.  Analyze the text.
    2.  Extract specifications (Component, Value, Unit, Conditions).
    3.  Return the result as a structured JSON object.

### 6. User Interface (`app.py`)
**Library:** `Streamlit`
- **Function:** Orchestrates the entire pipeline.
-   **Upload**: Handles PDF ingestion.
-   **Processing**: Triggers parsing, chunking, and embedding.
-   **Retrieval-Only Mode**: Allows testing the search accuracy without using an LLM API.
-   **Context Viewer**: Shows exactly what text was sent to the LLM (useful for debugging).

---

## 🚀 How to Run

### Prerequisities
- Python 3.10+
- OpenAI API Key (Optional, for extraction features)

### Installation
1.  **Clone/Download** the repository.
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate      # Windows
    # source venv/bin/activate # Mac/Linux
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
Execute the Streamlit application:
```bash
streamlit run app.py
```

## 🎮 Usage Guide

1.  **Sidebar Setup**:
    -   Select "Embedding Provider" (HuggingFace is free/local).
    -   Check "Retrieval Only (No LLM)" if you don't have an OpenAI API key.
    -   Upload your Service Manual PDF.
2.  **Build Knowledge Base**:
    -   Click the "Build Knowledge Base" button. Wait for the success message.
3.  **Query**:
    -   Type a question like *"Torque for brake caliper"* or use the sample buttons.
    -   Adjust the "Number of Chunks (K)" slider to control how much context is retrieved.
4.  **Analyze Results**:
    -   **JSON/Table Tabs**: View structured data (if LLM is active).
    -   **Context Tab**: Read the actual text chunks retrieved from the PDF to verify accuracy.
