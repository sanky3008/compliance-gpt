#!/usr/bin/env python
# Ingestion pipeline for processing PDF documents and storing embeddings.

import os
import glob
import json # Added for status file management
from typing import List, Set # Added Set for status tracking

# Import Langchain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import configuration variables
import config

# Helper functions for ingestion status management
def load_ingestion_status(status_file_path: str) -> Set[str]:
    """Loads the set of successfully processed PDF file paths from the status JSON file."""
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path, 'r') as f:
                processed_files = json.load(f)
                return set(processed_files)
        except json.JSONDecodeError:
            print(f"Warning: Status file {status_file_path} is corrupted. Starting fresh.")
            return set()
        except Exception as e:
            print(f"Warning: Could not read status file {status_file_path}: {e}. Starting fresh.")
            return set()
    return set()

def save_ingestion_status(status_file_path: str, processed_files: Set[str]):
    """Saves the given set of processed PDF file paths to the status JSON file."""
    try:
        with open(status_file_path, 'w') as f:
            json.dump(list(processed_files), f, indent=4)
    except Exception as e:
        print(f"Error: Could not write to status file {status_file_path}: {e}")

# Core ingestion functions
def load_and_split_single_pdf(pdf_path: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Loads a single PDF document, splits it into chunks, and assigns the PDF path as metadata.

    Args:
        pdf_path (str): The path to the PDF file.
        text_splitter (RecursiveCharacterTextSplitter): The text splitter instance.

    Returns:
        List[Document]: A list of Langchain Document objects (chunks) for the PDF.
    """
    print(f"Processing PDF: {pdf_path}...")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print(f"Warning: No content loaded from {pdf_path}.")
            return []
        
        # Add pdf_path to metadata of each document before splitting
        # This helps in tracking the origin of chunks if needed later
        for doc in documents:
            doc.metadata = doc.metadata if doc.metadata is not None else {}
            doc.metadata['source_pdf'] = pdf_path 
            
        chunks = text_splitter.split_documents(documents)
        print(f"Successfully loaded and split {pdf_path} into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error processing PDF file {pdf_path}: {e}")
        return []

def get_embedding_model() -> Embeddings:
    """
    Initializes and returns an embedding model based on the EMBEDDING_PROVIDER in config.
    """
    provider = config.EMBEDDING_PROVIDER.lower()
    print(f"Initializing embedding model from provider: {provider}...")

    if provider == "huggingface":
        print(f"Using HuggingFace model: {config.HUGGINGFACE_EMBEDDING_MODEL_NAME}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.HUGGINGFACE_EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Explicitly use CPU; change to 'cuda' for GPU
        )
    elif provider == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        print(f"Using OpenAI model: {config.OPENAI_EMBEDDING_MODEL_NAME}")
        # This 'chunk_size' is for batching documents to the API, NOT the text splitter chunk_size.
        openai_batch_size = 512 
        print(f"OpenAIEmbeddings will use an internal batch size (chunk_size parameter) of: {openai_batch_size}")
        embedding_model = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            chunk_size=openai_batch_size 
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{config.EMBEDDING_PROVIDER}'. Choose 'huggingface' or 'openai'.")
    
    print("Embedding model initialized successfully.")
    return embedding_model

def run_full_ingestion_process():
    """
    Orchestrates the resumable, file-batched document ingestion pipeline:
    1. Loads ingestion status to identify already processed files.
    2. Finds new PDF files to process.
    3. Initializes the embedding model and vector store.
    4. Processes new PDFs in batches: loads, chunks, embeds, and stores them.
    5. Updates ingestion status after each successful batch.
    """
    print("Starting the resumable document ingestion process...")

    # Step 1: Load ingestion status
    processed_pdf_files = load_ingestion_status(config.INGESTION_STATUS_FILE)
    print(f"Loaded {len(processed_pdf_files)} already processed PDF paths from status file.")

    # Step 2: Identify new PDF files
    if not os.path.isdir(config.DOCUMENTS_PATH):
        print(f"Error: Documents directory '{config.DOCUMENTS_PATH}' not found. Aborting.")
        return
    
    all_pdf_files_in_docs = glob.glob(os.path.join(config.DOCUMENTS_PATH, "**/*.pdf"), recursive=True)
    new_pdf_files_to_process = [f for f in all_pdf_files_in_docs if f not in processed_pdf_files]

    if not new_pdf_files_to_process:
        print("No new PDF files to process. Ingestion is up-to-date.")
        return
    
    print(f"Found {len(new_pdf_files_to_process)} new PDF files to process.")

    # Step 3: Initialize embedding model and text splitter
    print("Initializing embedding model and text splitter...")
    try:
        embedding_model = get_embedding_model()
    except ValueError as e:
        print(f"Error initializing embedding model: {e}. Aborting.")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    # Step 4: Initialize Chroma vector store
    # This will load an existing store if present, or create it on first addition.
    print(f"Initializing Chroma vector store. Persisting to: {config.VECTOR_STORE_DIRECTORY}")
    vector_store = Chroma(
        persist_directory=config.VECTOR_STORE_DIRECTORY,
        embedding_function=embedding_model
    )

    # Step 5: Process new PDFs in batches
    file_batch_size = config.FILE_PROCESSING_BATCH_SIZE
    total_new_files = len(new_pdf_files_to_process)
    
    # Define a new batch size for adding documents to Chroma, safely below its limit
    CHROMA_ADD_DOCUMENTS_BATCH_SIZE = 4000 # Max was 5461, using 4000 to be safe
    
    for i in range(0, total_new_files, file_batch_size):
        current_file_batch_paths = new_pdf_files_to_process[i : i + file_batch_size]
        batch_number = (i // file_batch_size) + 1
        total_batches = (total_new_files + file_batch_size - 1) // file_batch_size

        print(f"--- Processing File Batch {batch_number}/{total_batches} (Files {i+1}-{min(i+file_batch_size, total_new_files)} of {total_new_files}) ---")
        
        all_chunks_for_current_file_batch: List[Document] = []
        successfully_processed_files_in_this_batch: Set[str] = set()

        for pdf_path in current_file_batch_paths:
            chunks = load_and_split_single_pdf(pdf_path, text_splitter)
            if chunks: # Only add if chunks were successfully created
                all_chunks_for_current_file_batch.extend(chunks)
                # Mark this file as provisionally successful for this batch
                # It will be confirmed if add_documents succeeds
            # If load_and_split_single_pdf fails, it prints an error and returns [], so pdf_path isn't added here

        if not all_chunks_for_current_file_batch:
            print(f"No chunks generated for file batch {batch_number}. Skipping to next batch.")
            # Potentially mark files that yielded no chunks as processed here if desired,
            # for now, they will be re-scanned but should yield no chunks again.
            # If a file in the batch *failed* to load/split, it's not in `successfully_processed_files_in_this_batch`
            # and will be retried. If it loaded but was empty, it also won't be added.
            # This logic ensures only files that are successfully embedded update the status.
            
            # If some files in the batch *did* produce chunks but others failed,
            # we proceed with the chunks we have. The failed files won't be in `source_pdf` metadata of these chunks.
            # However, our current loop for `pdf_path in current_file_batch_paths` means `all_chunks_for_current_file_batch`
            # will contain chunks from all successfully processed files in this *file batch*.
            # The check `if not all_chunks_for_current_file_batch:` handles the case where *NO* files in the batch yielded chunks.
            # If *some* files yield chunks, we proceed.
            continue

        try:
            print(f"Preparing to add {len(all_chunks_for_current_file_batch)} chunks from {len(current_file_batch_paths)} file(s) for file batch {batch_number} to vector store...")
            
            num_chunks_to_add = len(all_chunks_for_current_file_batch)
            for chunk_idx_start in range(0, num_chunks_to_add, CHROMA_ADD_DOCUMENTS_BATCH_SIZE):
                chunk_idx_end = min(chunk_idx_start + CHROMA_ADD_DOCUMENTS_BATCH_SIZE, num_chunks_to_add)
                current_chunk_sub_batch = all_chunks_for_current_file_batch[chunk_idx_start:chunk_idx_end]
                
                if not current_chunk_sub_batch: # Should not happen if num_chunks_to_add > 0
                    continue

                sub_batch_num = (chunk_idx_start // CHROMA_ADD_DOCUMENTS_BATCH_SIZE) + 1
                total_sub_batches = (num_chunks_to_add + CHROMA_ADD_DOCUMENTS_BATCH_SIZE - 1) // CHROMA_ADD_DOCUMENTS_BATCH_SIZE
                
                print(f"  Adding chunk sub-batch {sub_batch_num}/{total_sub_batches} (chunks {chunk_idx_start+1}-{chunk_idx_end} of {num_chunks_to_add}) to vector store...")
                vector_store.add_documents(documents=current_chunk_sub_batch)
                print(f"  Successfully added chunk sub-batch {sub_batch_num}/{total_sub_batches}.")

            print(f"All chunks for file batch {batch_number} successfully added to vector store.")
            
            # If all sub-batches were successful, confirm the files that contributed chunks as processed
            for chunk_doc in all_chunks_for_current_file_batch:
                if 'source_pdf' in chunk_doc.metadata:
                    successfully_processed_files_in_this_batch.add(chunk_doc.metadata['source_pdf'])

            processed_pdf_files.update(successfully_processed_files_in_this_batch)
            save_ingestion_status(config.INGESTION_STATUS_FILE, processed_pdf_files)
            print(f"Updated ingestion status for {len(successfully_processed_files_in_this_batch)} files in batch {batch_number}.")
            
            # Persist changes to ChromaDB after each successful batch addition
            # vector_store.persist() # Chroma client typically handles persistence automatically on add/update with a persist_directory. Explicit persist might be needed for older versions or specific configurations.
            # For robust saving, especially with OpenAI billing or rate limits, persisting explicitly might be safer.
            print("Persisting vector store changes...")
            vector_store.persist() # Call persist to ensure data is written
            print("Vector store changes persisted.")

        except Exception as e:
            print(f"MAJOR ERROR: Failed to add chunks for batch {batch_number} to vector store: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Full error details: {str(e)}")
            print(f"Batch {batch_number} (files: {', '.join(current_file_batch_paths)}) will be retried on the next run as status was not updated for these files.")
            print("Stopping ingestion process. Please check the error and logs.")
            return # Stop the process, next run will pick up from this batch

    print("--- All new PDF files processed. ---")
    print("Full document ingestion process completed successfully!")

if __name__ == '__main__':
    print("Running ingestion pipeline directly...")
    run_full_ingestion_process()
    print("Ingestion pipeline script finished.") 