# RAG Chatbot

This is a simple Retrieval-Augmented Generation (RAG) chatbot built using Langchain, HuggingFace, and FAISS. It allows you to chat with your documents by leveraging a local language model and document embeddings.

## Features

- **Document Loading:** Loads text documents from a specified directory.
- **Text Splitting:** Splits documents into manageable chunks for processing.
- **Embeddings:** Generates and stores document embeddings using `all-MiniLM-L6-v2` for efficient retrieval.
- **Local LLM:** Utilizes a local `google/flan-t5-base` model for text generation.
- **Fallback Mechanisms:** Includes basic fallback for math queries and predefined responses for certain keywords (e.g., "weather").

## Setup

Follow these steps to set up the project locally:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/BlazeAxel99/RAG_chatbot-.git
    cd rag-chatbot
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\\Scripts\\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your documents:**
    Put your `.txt` or `.md` documents into the `data/` directory.

## Usage

1.  **Run the chatbot:**

    ```bash
    python chatbot.py
    ```

    This will:

    - Load and split your documents.
    - Generate embeddings and save them to the `embeddings/` directory.
    - Start the chat loop.

2.  **Interact with the chatbot:**
    Once the chatbot is ready, you can type your questions in the terminal.

    - Type `exit` or `quit` to end the chat.
    - Try asking math questions like `2+2` or `5 * 3`.
    - Ask about the weather to see a predefined fallback response.
    - Ask questions related to the content of your documents in the `data/` folder.

## Project Structure

- `chatbot.py`: The main script for the chatbot, including document loading, chain building, and the chat loop.
- `create_embeddings.py`: (Implicitly handled by `chatbot.py`'s `build_qa_chain`) Responsible for creating and saving document embeddings.
- `README.md`: This file.
- `requirements.txt`: Lists all Python dependencies.
- `test_chatbot.py`: Contains unit tests for the chatbot's functionalities.
- `data/`: Directory to store your input documents (`.txt`, `.md`).
- `embeddings/`: Directory where FAISS embeddings are saved.
- `__pycache__/`: Python cache directory.
- `.git/`: Git version control directory.
- `.pytest_cache/`: Pytest cache directory.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.
