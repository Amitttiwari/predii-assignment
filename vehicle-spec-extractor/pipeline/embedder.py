import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, provider: str = "huggingface", model_name: str = "all-MiniLM-L6-v2"):
        self.provider = provider
        self.model_name = model_name
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        
        elif self.provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_embedding_function(self):
        return self.embeddings

if __name__ == "__main__":
    try:
        embedder = Embedder(provider="huggingface")
        func = embedder.get_embedding_function()
        vec = func.embed_query("Test vehicle specification")
        print(f"Embedding dimension: {len(vec)}")
    except Exception as e:
        print(f"Error: {e}")
