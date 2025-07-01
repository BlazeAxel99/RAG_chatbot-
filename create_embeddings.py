
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
FAISS_PATH = "faiss_index (auto-created)"

# Configure loader for different file types
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*",
    use_multithreading=True,
    show_progress=True,
    loader_cls=TextLoader
)

print("Loading documents...")
documents = loader.load()
if not documents:
    print("No documents found in the data directory.")
    exit()
print(f"Loaded {len(documents)} documents.")


# Split the documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")


# Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
print("Creating FAISS vector store...")
db = FAISS.from_documents(docs, embeddings)
db.save_local(FAISS_PATH)

print("FAISS index has been created and saved.")
