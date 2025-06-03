import json
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

# Configuration
JSON_PATH = "program/hotpot_dev_fullwiki_v1.json"  # Path to your JSON file
OUTPUT_DIR = "./hotpotqa_pdfs"  # Directory for PDF files
MAX_DOCS = 20  # Maximum number of documents
LOG_FILE = "pdf_creation_log.txt"  # Log file

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Logging function
def log_message(message, log_file=LOG_FILE):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
    print(message)


# Function to create a PDF from text
def create_pdf(text, title, output_path):
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))

        # Add text (handling special characters)
        text = text.replace("\n", "<br/>").replace("&", "&").replace("<", "<").replace(">", ">")
        story.append(Paragraph(text, styles['Normal']))

        doc.build(story)
        log_message(f"Created PDF: {output_path}")
    except Exception as e:
        log_message(f"Error while creating PDF for {title}: {e}")


# Main function to create PDFs from HotpotQA dataset
def create_hotpotqa_pdfs(json_path, output_dir, max_docs):
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Read JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_message(f"Successfully loaded {json_path}")
    except FileNotFoundError:
        log_message(f"Error: File {json_path} not found")
        return []
    except json.JSONDecodeError:
        log_message(f"Error: Invalid JSON format in file {json_path}")
        return []

    # Collect unique documents
    documents = {}
    for item in data:
        context = item.get("context", [])
        for title, sentences in context:
            if title not in documents:
                text = "".join(sentences)
                if text.strip():  # Skip empty text
                    documents[title] = text
                else:
                    log_message(f"Empty text for article {title}, skipped")
            if len(documents) >= max_docs:
                break
        if len(documents) >= max_docs:
            break

    log_message(f"Found {len(documents)} unique documents")

    # Mapping for documents
    mapping = []

    # Create PDF for each document
    for title, text in documents.items():
        # Sanitize filename (replace invalid characters)
        safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "_").replace("*", "_")
        output_path = os.path.join(output_dir, f"{safe_title}.pdf")

        # Generate PDF
        create_pdf(text, title, output_path)

        # Add to mapping
        mapping.append({"title": title, "pdf_path": output_path})

    # Save mapping to CSV
    mapping_df = pd.DataFrame(mapping)
    mapping_csv = os.path.join(output_dir, "document_mapping.csv")
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8")
    log_message(f"Mapping saved to {mapping_csv}")

    return mapping


# Execution
if __name__ == "__main__":
    pdf_mapping = create_hotpotqa_pdfs(JSON_PATH, OUTPUT_DIR, MAX_DOCS)
    log_message(f"Created {len(pdf_mapping)} PDF files in {OUTPUT_DIR}")
