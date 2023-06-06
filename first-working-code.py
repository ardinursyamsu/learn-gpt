import PyPDF2
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
        return text

# Path to your PDF file
pdf_path = 'source.pdf'

# Extract text from the PDF
corpus = extract_text_from_pdf(pdf_path)

# Tokenize the corpus
tokens = word_tokenize(corpus)

# Train Word2Vec model
model = Word2Vec([tokens], min_count=1)

# Example usage: Find similar words to a target word
target_word = 'economy'
similar_words = model.wv.most_similar(target_word)

# Print the similar words
print(f"Similar words to '{target_word}':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")