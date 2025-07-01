# chatbot.py

def build_qa_chain():
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import pipeline

    # Load documents
    loader = DirectoryLoader('data/', glob='**/*.*', loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local("embeddings/")
    print("FAISS index saved to ./embeddings")

    # LLM
    local_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256, do_sample=True, temperature=0.3)
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Utility functions for fallback
import re

def is_math_query(query):
    return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", query))

def handle_math(query):
    try:
        return str(eval(query))
    except:
        return "Sorry, I couldn't calculate that."

def handle_local_fallbacks(query):
    if "weather" in query.lower():
        return "This chatbot is offline and cannot fetch live weather data."
    if is_math_query(query):
        return handle_math(query)
    return None

# CLI
def chat_loop():
    qa_chain = build_qa_chain()
    print("\n🔍 Chatbot is ready! Ask questions or type 'exit' to quit.\n")
    while True:
        query = input("Ask me anything: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        fallback = handle_local_fallbacks(query)
        if fallback:
            print("Answer:", fallback)
            continue

        result = qa_chain.invoke({"query": query})
        print("Answer:", result)

if __name__ == "__main__":
    chat_loop()
