from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, pages_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunks the extracted text into LangChain Documents with metadata.
        """
        documents = []
        for page in pages_data:
            text = page["text"]
            metadata = page["metadata"]
            
            chunks = self.splitter.create_documents([text], metadatas=[metadata])
            documents.extend(chunks)
            
        return documents

if __name__ == "__main__":
    # Test
    chunker = TextChunker()
    # Mock data
    data = [{"text": "This is a test text related to vehicle specs. " * 50, "metadata": {"page": 1}}]
    chunks = chunker.chunk_documents(data)
    print(f"Created {len(chunks)} chunks.")
    if chunks:
        print(chunks[0].page_content)
        print(chunks[0].metadata)
