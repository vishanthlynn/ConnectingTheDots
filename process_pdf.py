
import os
import json
import pdfplumber
import fitz  # PyMuPDF

# Extract title using PDF metadata
def extract_title(pdf_path):
    doc = fitz.open(pdf_path)
    title = doc.metadata.get("title")
    return title if title else os.path.basename(pdf_path).replace(".pdf", "")

# Extract outline (headings) from each page
def extract_outline(pdf_path):
    outline = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    clean_line = line.strip()
                    if len(clean_line) < 5:
                        continue

                    # Define rules for heading levels
                    if clean_line.isupper():
                        level = "H1"
                    elif clean_line.istitle():
                        level = "H2"
                    else:
                        continue

                    outline.append({
                        "level": level,
                        "text": clean_line,
                        "page": page_num
                    })
    return outline

# Process one PDF: extract title + outline + save JSON
def process_pdf(pdf_path, output_path):
    title = extract_title(pdf_path)
    outline = extract_outline(pdf_path)

    result = {
        "title": title,
        "outline": outline
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# Main logic for processing all PDFs in /input
def main():
    input_dir = "input"
    output_dir = "output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace(".pdf", ".json")
            output_path = os.path.join(output_dir, output_filename)

            print(f"📄 Processing: {filename}")
            process_pdf(input_path, output_path)

    print("✅ All PDFs processed.")

if __name__ == "__main__":
    main()

