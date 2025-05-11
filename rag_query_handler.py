#!/usr/bin/env python
"""
Handles the RAG (Retrieval Augmented Generation) querying process.

This module is responsible for:
- Loading the existing vector store.
- Initializing the chosen LLM (e.g., Groq, Google Gemini, OpenAI).
- Setting up the Langchain ConversationalRetrievalChain.
- Providing a function to query the conversational RAG chain.
"""

import sys
import os
from typing import List, Tuple, Dict, Any # Added for type hinting chat_history and response

# Add the current workspace to the path to allow imports from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config # For accessing configuration like API keys, model names, etc.
    from ingestion_pipeline import get_embedding_model # To use the same embedding model for loading the store
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure config.py and ingestion_pipeline.py are in the project root and sys.path is correct.")
    sys.exit(1)

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings # Changed from HuggingFaceEmbeddings for broader compatibility
from langchain_core.language_models.chat_models import BaseChatModel
# from langchain.chains import RetrievalQA # Replaced with ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain # Added for conversational context

# Import specific LLM clients
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Added for OpenAI LLM option

def load_existing_vector_store(embedding_model: Embeddings) -> Chroma:
    """
    Loads the persisted Chroma vector store from disk.

    Args:
        embedding_model (Embeddings): The same embedding model instance that was used
                                      to create the vector store.
    Returns:
        Chroma: The loaded Chroma vector store instance.
    Raises:
        FileNotFoundError: If the vector store directory does not exist.
        Exception: For other errors during loading.
    """
    persist_directory = config.VECTOR_STORE_DIRECTORY
    print(f"Loading existing vector store from: {persist_directory}...")

    if not os.path.isdir(persist_directory):
        error_msg = f"Vector store directory '{persist_directory}' not found. Please run the ingestion process first."
        print(f"Error: {error_msg}")
        raise FileNotFoundError(error_msg)
    
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        # Check if the vector store is empty right after loading
        if vector_store._collection.count() == 0:
            print("Warning: Vector store loaded but appears to be empty. Querying may not yield results.")
        else:
            print(f"Vector store loaded successfully with {vector_store._collection.count()} items.")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store from '{persist_directory}': {e}")
        raise

def get_llm_chat_client() -> BaseChatModel:
    """
    Initializes and returns the appropriate Langchain chat model client based on config.
    """
    provider = config.LLM_PROVIDER.lower()
    print(f"Initializing LLM chat client for provider: {provider}...")

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY.")
        print(f"Using Groq model: {config.GROQ_MODEL_NAME}")
        return ChatGroq(api_key=config.GROQ_API_KEY, model_name=config.GROQ_MODEL_NAME)
    
    elif provider == "google_gemini":
        if not config.GOOGLE_API_KEY:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY.")
        print(f"Using Google Gemini model: {config.GEMINI_MODEL_NAME}")
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME, google_api_key=config.GOOGLE_API_KEY)
    
    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
        print(f"Using OpenAI model: {config.OPENAI_CHAT_MODEL_NAME}")
        return ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name=config.OPENAI_CHAT_MODEL_NAME)

    else:
        raise ValueError(f"Unsupported LLM provider: '{config.LLM_PROVIDER}'. Check config.py.")

def setup_conversational_rag_chain(vector_store: Chroma, llm_client: BaseChatModel) -> ConversationalRetrievalChain:
    """
    Sets up the Langchain ConversationalRetrievalChain.

    Args:
        vector_store (Chroma): The loaded Chroma vector store instance.
        llm_client (BaseChatModel): The initialized LLM chat client.

    Returns:
        ConversationalRetrievalChain: The configured conversational chain.
    """
    print("Setting up Conversational RAG chain...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    print(f"Retriever created. Will fetch top {retriever.search_kwargs.get('k', 'N/A')} documents.")

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Optional: for more complex memory management

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_client,
        retriever=retriever,
        return_source_documents=True, # Return source documents for context
        # memory=memory, # If using explicit memory
        # chain_type="stuff" # Default, can be changed if needed
    )
    print("Conversational RAG chain setup complete.")
    return conversational_chain

def query_conversational_rag_chain(chain: ConversationalRetrievalChain, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Queries the configured ConversationalRetrievalChain with a question and chat history.

    Args:
        chain (ConversationalRetrievalChain): The chain instance.
        question (str): The user's current question.
        chat_history (List[Tuple[str, str]]): A list of past interactions, where each tuple is (user_query, ai_response).

    Returns:
        Dict[str, Any]: A dictionary containing the answer and source documents.
                         Example: {'question': str, 'chat_history': list, 'answer': str, 'source_documents': list[Document]}
    """
    print(f"\nQuerying Conversational RAG chain with: '{question}'")
    if chat_history:
        print(f"Using chat history with {len(chat_history)} turns.")
        # for i, (q, a) in enumerate(chat_history):
        #     print(f"  History {i+1}: Q: {q[:50]}... A: {a[:50]}...")
            
    response = chain.invoke({"question": question, "chat_history": chat_history})
    print("Query processed by Conversational RAG chain.")
    return response

if __name__ == '__main__':
    print("--- Testing rag_query_handler.py (Conversational) ---")
    
    try:
        print("Step 1: Initializing embedding model...")
        emb_model = get_embedding_model()
        
        print("\nStep 2: Loading vector store...")
        vs = load_existing_vector_store(embedding_model=emb_model)
        
        if vs._collection.count() == 0:
            print("Vector store is empty. Test cannot proceed with querying. Please run ingestion first.")
            sys.exit(0)
            
        print("\nStep 3: Initializing LLM client...")
        llm = get_llm_chat_client()
        
        print("\nStep 4: Setting up Conversational RAG chain...")
        conv_rag_chain = setup_conversational_rag_chain(vector_store=vs, llm_client=llm)
        
        print("\nStep 5: Simulating a conversation...")
        
        # First question
        q1 = "What is the penalty for a shortfall in cash or cash equivalents?"
        chat_history_for_chain: List[Tuple[str, str]] = []
        print(f"\n--- Test Query 1 ---")
        response1 = query_conversational_rag_chain(conv_rag_chain, q1, chat_history_for_chain)
        print(f"Q1: {q1}")
        print(f"A1: {response1.get('answer')}")
        
        # Update chat history for the chain
        chat_history_for_chain.append((q1, response1.get('answer', '')))

        # Second question (follow-up)
        q2 = "And what if the shortfall exceeds 5% of the requirement?"
        print(f"\n--- Test Query 2 (Follow-up) ---")
        response2 = query_conversational_rag_chain(conv_rag_chain, q2, chat_history_for_chain)
        print(f"Q2: {q2}")
        print(f"A2: {response2.get('answer')}")
        
        # print("\nSource Documents for last query (Q2):")
        # if response2.get('source_documents'):
        #     for i, doc in enumerate(response2['source_documents']):
        #         print(f"  --- Source {i+1} (Q2) ---")
        #         print(f"  Content (first 100 chars): {doc.page_content[:100]}...")
        #         print(f"  Metadata: {doc.metadata}")
        # else:
        #     print("  No source documents returned for Q2.")

    except FileNotFoundError as fnf_error:
        print(f"Test run failed: {fnf_error}")
    except ValueError as val_error:
        print(f"Test run failed due to configuration issue: {val_error}")
    except Exception as e:
        print(f"An unexpected error occurred during the test run: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

    print("\n--- rag_query_handler.py (Conversational) test finished ---") 