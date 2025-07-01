import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
import importlib

import chatbot  # Assumes chatbot.py is refactored and doesn't run input() at import


# ----------------------------
# ✅ 1. Test math detection
# ----------------------------
def test_is_math_query():
    assert chatbot.is_math_query("2+2")
    assert chatbot.is_math_query("5 * 3")
    assert chatbot.is_math_query("10 / 2")
    assert chatbot.is_math_query("7 - 4")
    assert not chatbot.is_math_query("hello world")
    assert not chatbot.is_math_query("2 plus 2")


# ----------------------------
# ✅ 2. Test fallback logic
# ----------------------------
def test_handle_local_fallbacks():
    assert chatbot.handle_local_fallbacks("what is the weather like?") == "This chatbot is offline and cannot fetch live weather data."
    assert chatbot.handle_local_fallbacks("2+2") == "4"
    assert chatbot.handle_local_fallbacks("5 * 3") == "15"
    assert chatbot.handle_local_fallbacks("hello world") is None


# ----------------------------
# ✅ 3. Test embedding creation (simulate create_embeddings.py)
# ----------------------------
def test_embedding_creation():
    path = "embeddings/"
    doc_path = "data/test_doc.txt"

    # Setup
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(doc_path, "w") as f:
        f.write("This is a sample test document for embedding.")

    # Run embedding generation via chatbot module
    qa_chain = chatbot.build_qa_chain()

    # Validate FAISS index created
    assert os.path.exists(path)
    assert any(f.startswith("index") for f in os.listdir(path))

    # Cleanup
    os.remove(doc_path)
    shutil.rmtree(path)


# ----------------------------
# ✅ 4. Test QA invocation using mocks
# ----------------------------
# @patch("chatbot.HuggingFacePipeline")
# @patch("chatbot.FAISS")
# @patch("chatbot.HuggingFaceEmbeddings")
# def test_qa_chain_invocation(mock_embed, mock_faiss, mock_llm_pipeline):
@patch("langchain_community.embeddings.HuggingFaceEmbeddings")
@patch("chatbot.FAISS")
@patch("langchain_huggingface.llms.HuggingFacePipeline")
def test_qa_chain_invocation(mock_pipeline, mock_faiss, mock_embed):
    # Mock embedding model
    mock_embed.return_value = MagicMock()

    # Mock FAISS and retriever
    mock_faiss_instance = MagicMock()
    mock_retriever = MagicMock()
    mock_faiss_instance.as_retriever.return_value = mock_retriever
    mock_faiss.from_documents.return_value = mock_faiss_instance

    # Mock LLM pipeline
    mock_pipeline = MagicMock()
    mock_llm_pipeline.return_value = mock_pipeline

    # Mock QA chain and its invoke
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"query": "mock query", "result": "mocked answer"}

    with patch("langchain.chains.RetrievalQA.from_chain_type", return_value=mock_chain):
        qa_chain = chatbot.build_qa_chain()
        query = "What is Python?"
        response = qa_chain.invoke({"query": query})
        assert response["result"] == "mocked answer"
        mock_chain.invoke.assert_called_once_with({"query": query})
