# process_pdf.py
import os
import fitz # This is PyMuPDF

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def process_pdf_file(pdf_path):
    """
    Processes a single PDF file to extract basic info and
    creates a placeholder JSON output.
    """
    print(f"Attempting to process: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        print(f"  - Successfully opened PDF. Pages: {num_pages}")
        doc.close()

        # Placeholder for your actual heading extraction logic
        # You will replace this with real title and outline data
        output_data = {
            "title": "Extracted Title Placeholder",
            "outline": [
                {"level": "H1", "text": "Sample Heading 1", "page": 1},
            ]
        }

        # Define the output JSON file path
        output_filename = os.path.basename(pdf_path).replace('.pdf', '.json')
        output_json_path = os.path.join(OUTPUT_DIR, output_filename)

        import json
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  - Generated placeholder JSON: {output_json_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    print("Starting PDF processing for Adobe Hackathon Round 1A...")

    # Ensure input and output directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files_in_input = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files_in_input:
        print(f"No PDF files found in the input directory: {INPUT_DIR}. Please place sample PDFs there to test.")
    else:
        print(f"Found {len(pdf_files_in_input)} PDF(s) in {INPUT_DIR}.")
        for pdf_file_name in pdf_files_in_input:
            pdf_full_path = os.path.join(INPUT_DIR, pdf_file_name)
            process_pdf_file(pdf_full_path)

    print("All specified PDF processing tasks completed.")