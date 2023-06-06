from PyPDF2 import PdfReader

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        total_num_pages = len(reader.pages)

        for page_number in range(total_num_pages):
            text += reader.pages[page_number].extract_text()
        return text