import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), 'vehicle-spec-extractor'))

try:
    print("Testing imports...")
    from pipeline.parser import PDFParser
    print("PDFParser imported.")
    
    from pipeline.chunker import TextChunker
    print("TextChunker imported.")
    
    from pipeline.embedder import Embedder
    print("Embedder imported.")
    
    from pipeline.retriever import VectorRetriever
    print("VectorRetriever imported.")
    
    from pipeline.extractor import SpecExtractor
    print("SpecExtractor imported.")
    
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
