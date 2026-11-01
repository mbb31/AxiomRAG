import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Bring in the Translator (Ollama) and the Filing Cabinet (ChromaDB)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("1. Opening the PDF and extracting text...")
pdf = pdfplumber.open("report.pdf")
first_page = pdf.pages[0]
page_text = first_page.extract_text()

print("2. Chopping text into chunks...")
shredder = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = shredder.split_text(page_text)

print("3. Waking up the Nomic Translator...")
# Connect to the Nomic model we downloaded in Step 1
translator = OllamaEmbeddings(model="nomic-embed-text")

print("4. Translating chunks and saving to ChromaDB... (This takes a few seconds)")
# This takes our chunks, runs them through the translator, and saves them in a new folder called 'axiomrag_db'
vector_db = Chroma.from_texts(
    texts=chunks, 
    embedding=translator, 
    persist_directory="./axiomrag_db"
)

print("Success! Your PDF is now translated and permanently saved in the database.")