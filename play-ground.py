import PyPDF2
import os
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        total_num_pages = len(reader.pages)

        for page_number in range(total_num_pages):
            text += reader.pages[page_number].extract_text()
        return text

# Path to your PDF file
pdf_path = 'source.pdf'

# Extract text from the PDF
corpus = extract_text_from_pdf(pdf_path)

# Tokenize the corpus
tokens = word_tokenize(corpus)

# Check if model already exists
current_file_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_file_directory,'model/model.bin')

if (os.path.exists(model_path) == False):
    # Train Word2Vec model
    model = Word2Vec([tokens], min_count=1)
    model.save(model_path)
else:
    model = Word2Vec.load(model_path)

# Example usage: Find similar words to a target word
target_word = 'economy'
similar_words = model.wv.most_similar(target_word)

# Print the similar words
print(f"Similar words to '{target_word}':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")