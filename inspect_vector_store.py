#!/usr/bin/env python
"""
Script to inspect the contents of the persisted Chroma vector store.
It loads the store, fetches a sample of items, and prints their content and metadata.
"""

import sys
import os

# Add the current workspace to the path to allow imports from config
# This assumes the script is run from the project root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
except ImportError:
    print("Error: Could not import config.py. Make sure it's in the same directory or sys.path is set correctly.")
    sys.exit(1)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def inspect_store():
    """
    Loads the Chroma vector store and prints a sample of its contents.
    """
    print("--- Vector Store Inspection Script ---")

    # Initialize the same embedding model used during ingestion
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}  # Assuming CPU, change if you used GPU for ingestion
        )
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # Define the path to the persisted vector store
    persist_directory = config.VECTOR_STORE_DIRECTORY
    if not os.path.isdir(persist_directory):
        print(f"Error: Vector store directory '{persist_directory}' not found.")
        print("Please ensure the ingestion process has run successfully and the directory exists.")
        return
        
    print(f"Attempting to load vector store from: {persist_directory}...")

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        print("Vector store loaded successfully.")

        # Fetch a sample of documents (e.g., first 5)
        # The get() method can retrieve documents.
        # We can specify include=["metadatas", "documents"] to get both.
        print("\nFetching first 5 items (or fewer if not available)...")
        retrieved_items = vector_store.get(limit=5, include=["metadatas", "documents"])

        documents = retrieved_items.get('documents')
        metadatas = retrieved_items.get('metadatas')
        ids = retrieved_items.get('ids')

        if not documents:
            print("No documents found in the vector store sample. The store might be empty or an issue occurred.")
        else:
            print(f"Retrieved {len(documents)} items.\n")
            for i in range(len(documents)):
                item_id = ids[i] if ids and i < len(ids) else 'N/A'
                print(f"--- Item {i+1} (ID: {item_id}) ---")
                
                print("Metadata:")
                if metadatas and i < len(metadatas) and metadatas[i]:
                    for key, value in metadatas[i].items():
                        print(f"  {key}: {value}")
                else:
                    print("  No metadata for this item.")
                
                print("\nDocument Chunk Content (first 500 characters):")
                content_snippet = documents[i][:500] + ("..." if len(documents[i]) > 500 else "")
                print(content_snippet)
                print("------------------------------------\n")

        # You can also get the total count of items
        try:
            total_items = vector_store._collection.count()
            print(f"Total items in the vector store: {total_items}")
        except Exception as e:
            print(f"Could not retrieve total item count: {e}")
            print("This might happen if the collection is empty or not properly initialized.")

    except Exception as e:
        print(f"An error occurred while loading or querying the vector store: {e}")
        print("Please ensure that the vector store exists at the specified path,")
        print("was created with the same embedding model, and is not corrupted.")
        print("If you recently updated ChromaDB or Langchain, there might be compatibility issues.")

if __name__ == '__main__':
    inspect_store()
    print("\n--- Inspection script finished ---") 