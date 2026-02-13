import fitz  # pymupdf
from typing import List, Dict, Any

class PDFParser:
    def __init__(self, header_height: int = 50, footer_height: int = 50):
        """
        Args:
            header_height: Approximate height of header to ignore (in points).
            footer_height: Approximate height of footer to ignore (in points).
        """
        self.header_height = header_height
        self.footer_height = footer_height

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF file, filtering out headers and footers.
        Returns a list of dictionaries containing page content and metadata.
        """
        doc = fitz.open(pdf_path)
        extracted_pages = []

        for page_num, page in enumerate(doc):
            # Get page dimensions
            width = page.rect.width
            height = page.rect.height

            # Define the crop box (exclude header and footer)
            # Origin is top-left
            crop_rect = fitz.Rect(0, self.header_height, width, height - self.footer_height)
            
            # Extract text only from the cropped area
            # text_content = page.get_text("text", clip=crop_rect)
            
            # Alternative: Get blocks and filter by position for more control if needed
            # For now, clip is robust enough for general text
            text_content = page.get_text(clip=crop_rect)

            if text_content.strip():
                extracted_pages.append({
                    "text": text_content.strip(),
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })

        doc.close()
        return extracted_pages

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        pages = parser.extract_text(sys.argv[1])
        print(f"Extracted {len(pages)} pages.")
        if pages:
            print("Sample from page 1:")
            print(pages[0]['text'][:200])
