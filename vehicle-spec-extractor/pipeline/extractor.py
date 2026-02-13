import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

EXTRACTION_PROMPT = """
You are an automotive specification extraction assistant.

Extract structured vehicle specifications from the provided context.

Query:
{user_query}

Context:
{retrieved_chunks}

Return JSON only:

[
  {
    "component": "",
    "spec_type": "",
    "value": "",
    "unit": "",
    "conditions": ""
  }
]

Rules:
- No hallucinations
- If not found → []
- Extract all specs
- Normalize units
- Separate multiple values
"""

class SpecExtractor:
    def __init__(self, model_name: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback or warning - for now, assuming env is set or user will set it
            print("Warning: OPENAI_API_KEY not set. Extraction will fail unless set.")
        
        # Use a lower temperature for extraction to be deterministic
        self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)

    def extract(self, query: str, context_chunks: list) -> list:
        """
        Extracts specifications from the combined context chunks based on the query.
        """
        # Combine context
        context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])
        
        # Format prompt
        formatted_prompt = EXTRACTION_PROMPT.format(
            user_query=query,
            retrieved_chunks=context_text
        )

        messages = [
            SystemMessage(content="You are a helpful assistant that extracts structured data."),
            HumanMessage(content=formatted_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content.strip())
            return data
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {content}")
            return []
        except Exception as e:
            print(f"Error during extraction: {e}")
            return []

if __name__ == "__main__":
    # Mock test
    # This requires an API key to run
    pass
