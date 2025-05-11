#!/usr/bin/env python
# Placeholder for app.py
import argparse
import ingestion_pipeline
import streamlit as st
from typing import List, Tuple, Dict, Any

# Assuming rag_query_handler.py and ingestion_pipeline.py are in the same directory or accessible via PYTHONPATH
try:
    import config # To ensure it's loaded, though direct use here might be minimal
    from rag_query_handler import (
        load_existing_vector_store,
        get_llm_chat_client,
        setup_conversational_rag_chain,
        query_conversational_rag_chain
    )
    from ingestion_pipeline import get_embedding_model
except ImportError as e:
    st.error(f"Error importing necessary modules: {e}. Ensure all required files (config.py, rag_query_handler.py, ingestion_pipeline.py) are present and configured.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during imports: {e}")
    st.stop()

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the document ingestion process to load and embed documents."
    )
    # Future arguments for querying can be added here, e.g.:
    # parser.add_argument("--query", type=str, help="Ask a question to the RAG system.")

    args = parser.parse_args()

    if args.ingest:
        print("Ingestion process triggered from app.py...")
        ingestion_pipeline.run_full_ingestion_process()
    # elif args.query:
    #     print(f"Query received: {args.query}")
    #     # Call query handling logic here
    else:
        # Default behavior if no specific action is called, or print help
        print("No specific action requested. Use --ingest to process documents or --query \"Your question\" to ask.")
        parser.print_help()

# --- Cached Resource Loading ---
@st.cache_resource(show_spinner="Initializing Groww ComplianceGPT, please wait...")
def load_resources():
    """Loads all necessary resources for the RAG chain: embedding model, vector store, LLM, and the chain itself."""
    try:
        embedding_model = get_embedding_model()
        vector_store = load_existing_vector_store(embedding_model=embedding_model)
        
        if vector_store._collection.count() == 0:
            st.error("Vector store is empty. Please run the ingestion pipeline (`python ingestion_pipeline.py`) to populate it.")
            return None # Indicate failure to load completely

        llm_client = get_llm_chat_client()
        conversational_rag_chain = setup_conversational_rag_chain(vector_store=vector_store, llm_client=llm_client)
        return conversational_rag_chain
    except FileNotFoundError as fnf_error:
        st.error(f"Failed to load resources: {fnf_error}. Please ensure the vector store has been created by running the ingestion pipeline.")
        return None
    except ValueError as val_error: # Catch API key errors or unsupported provider from get_llm_chat_client
        st.error(f"Configuration error: {val_error}. Please check your .env file and config.py.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}") # More detailed error for debugging
        return None

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Groww ComplianceGPT", layout="wide")
    st.title("Groww ComplianceGPT")

    # Load the conversational RAG chain
    rag_chain = load_resources()

    if not rag_chain:
        st.warning("RAG chain could not be initialized. Please check the error messages above and ensure the ingestion process has been completed successfully.")
        st.stop() # Stop execution if chain isn't loaded

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Optional: Add a welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm Groww ComplianceGPT. How can I help you with your compliance questions today?"
        })

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "source_documents" in message and message["source_documents"]:
                with st.expander("View Sources", expanded=False):
                    for i, doc in enumerate(message["source_documents"]):
                        st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source_pdf', 'N/A')}` (Page {doc.metadata.get('page', 'N/A') if 'page' in doc.metadata else doc.metadata.get('page_number', 'N/A')}) ")
                        st.text(f"{doc.page_content[:300]}...") # Display a snippet

    # Clear chat button
    if st.button("Clear Chat Thread"):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat cleared! Ask me a new question about your compliance documents."
        })
        st.rerun() # Rerun the script to reflect the cleared messages immediately

    # Accept user input
    if prompt := st.chat_input("Ask your compliance question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare chat history for the chain
        # The chain expects a list of tuples: (user_query, ai_response)
        chat_history_for_chain: List[Tuple[str, str]] = []
        # Iterate through messages, two at a time, to form (user, assistant) pairs
        # We only consider messages up to the one before the current user prompt
        history_to_process = st.session_state.messages[:-1] 
        
        idx = 0
        while idx < len(history_to_process) - 1: # Ensure there's a next message to form a pair
            current_msg = history_to_process[idx]
            next_msg = history_to_process[idx+1]
            
            if current_msg["role"] == "user" and next_msg["role"] == "assistant":
                user_q = current_msg["content"]
                ai_a = next_msg["content"]
                chat_history_for_chain.append((user_q, ai_a))
                idx += 2 # Move past this pair
            else:
                # If the sequence is not user then assistant (e.g. two user msgs, or assistant first),
                # just move to the next message to try and find a valid pair start.
                # This also handles the initial assistant welcome message correctly by skipping it.
                idx += 1

        # Display thinking spinner and process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_data = query_conversational_rag_chain(
                        chain=rag_chain,
                        question=prompt,
                        chat_history=chat_history_for_chain
                    )
                    answer = response_data.get("answer", "Sorry, I couldn't find an answer.")
                    source_documents = response_data.get("source_documents", [])
                    
                    st.markdown(answer)
                    if source_documents:
                        with st.expander("View Sources", expanded=False):
                             for i, doc in enumerate(source_documents):
                                st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source_pdf', 'N/A')}` (Page {doc.metadata.get('page', 'N/A') if 'page' in doc.metadata else doc.metadata.get('page_number', 'N/A')}) ")
                                st.text(f"{doc.page_content[:300]}...") # Display a snippet
                    
                    # Add assistant response to chat history (including sources for potential later use, though not directly displayed from history yet)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "source_documents": source_documents # Store sources with the message
                    })

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main() 