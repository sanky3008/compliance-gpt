---
title: Groww ComplianceGPT
emoji: 📚 # You can choose another emoji if you like, e.g., 🔎 or 🤖
colorFrom: indigo
colorTo: green
sdk: streamlit
python_version: 3.12
app_file: app.py
pinned: false
---


# Groww ComplianceGPT

## Overview

Groww ComplianceGPT is a Retrieval Augmented Generation (RAG) system designed to answer questions about PDF circulars and compliance documents. Users can ingest a corpus of PDF documents, and then interact with a conversational AI through a web interface to ask questions and receive answers based on the content of these documents. The system leverages Langchain for its core RAG capabilities, supports various embedding models (OpenAI, HuggingFace) and LLM providers (Groq, OpenAI, Google Gemini), and uses Streamlit for the user interface.

## Features

*   **Resumable Document Ingestion:** Efficiently processes PDF documents, creating and updating a vector store. If the ingestion process is interrupted, it can resume from where it left off, saving time and resources.
*   **Configurable Models:** Easily switch between different embedding providers (OpenAI, local HuggingFace Sentence Transformers) and LLM providers (Groq, OpenAI, Google Gemini) via a configuration file.
*   **Conversational Chat Interface:** A user-friendly web UI built with Streamlit allows for natural, multi-turn conversations with the RAG system.
*   **Source Document Display:** The chat interface can display snippets and metadata (source PDF, page number) of the documents used to generate an answer, allowing for verification.
*   **Chat History Management:** Remembers the context of the conversation for more relevant follow-up answers.
*   **Clear Chat Functionality:** Users can easily clear the current conversation thread and start anew.
*   **Persistent Vector Store:** Uses ChromaDB to store document embeddings locally.

## System Architecture

The system is primarily composed of the following Python scripts:

*   `config.py`: Manages all configurable parameters for the system, including API keys, model names, and paths.
*   `ingestion_pipeline.py`: Handles the loading of PDF documents from a specified folder, splits them into manageable chunks, generates embeddings, and stores them in a ChromaDB vector store. It features a resumable, file-batched processing mechanism.
*   `rag_query_handler.py`: Contains the logic for loading the existing vector store, initializing the chosen LLM, and setting up/querying the `ConversationalRetrievalChain` from Langchain.
*   `app.py`: A Streamlit application that provides the "Groww ComplianceGPT" web-based chat interface. It orchestrates the loading of resources and interaction with the `rag_query_handler`.

## Setup and Installation

**1. Prerequisites:**

*   Python 3.9 or higher.
*   Git (optional, for cloning).
*   Access to a terminal or command prompt.

**2. Clone the Repository (Optional):**

   If the project is hosted on Git, clone it:
   ```bash
   git clone <repository_url>
   cd <project_directory_name>
   ```
   If not, ensure all project files (`.py`, `requirements.txt`, etc.) are in a single project directory.

**3. Create and Activate a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   ```
   Activate it:
   *   On macOS and Linux:
       ```bash
       source venv/bin/activate
       ```
   *   On Windows:
       ```bash
       venv\Scripts\activate
       ```

**4. Install Dependencies:**

   Install all required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

**5. Set Up Environment Variables:**

   *   Create a file named `.env` in the root of your project directory.
   *   Add your API keys to this file. For example:
       ```env
       OPENAI_API_KEY="your_openai_api_key_here"
       GROQ_API_KEY="your_groq_api_key_here"
       GOOGLE_API_KEY="your_google_api_key_here"
       ```
   *   The `config.py` file is set up to load these variables.

## Usage

**1. Run the Document Ingestion Pipeline:**

   *   Place all your PDF circulars/documents into the `Documents` folder (or the folder specified by `DOCUMENTS_PATH` in `config.py`).
   *   Run the ingestion script from your terminal:
       ```bash
       python ingestion_pipeline.py
       ```
   *   This will process the PDFs and create/update the `vector_store_db` directory. This step only needs to be run once initially, or when you add new documents or change embedding settings.

**2. Run the Chat Application:**

   Once the ingestion is complete, start the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
   This will open the "Groww ComplianceGPT" chat interface in your web browser (usually at `http://localhost:8501`). You can then start asking questions.

## Configuration

Most of the system's behavior can be configured by modifying the variables in `config.py`. This includes:

*   Paths for the vector store and document sources.
*   Choice of embedding provider and model names (OpenAI, HuggingFace).
*   Text splitting parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
*   Choice of LLM provider and model names (Groq, Google Gemini, OpenAI).
*   Ingestion settings like `FILE_PROCESSING_BATCH_SIZE` and `INGESTION_STATUS_FILE` path.

Refer to the comments within `config.py` for details on each variable.

## File Structure (Simplified)

```
.env.example                 # Example environment file
.gitignore
Documents/                   # Folder for your PDF documents
README.md
RAG_SYSTEM_DESIGN.md         # Detailed system design document
app.py                       # Streamlit UI application
config.py                    # Configuration settings
ingestion_pipeline.py        # PDF ingestion and vector store creation
ingestion_status.json        # (Generated) Tracks processed PDFs
rag_query_handler.py         # RAG chain and query logic
requirements.txt             # Python dependencies
vector_store_db/             # (Generated) ChromaDB vector store
venv/                        # (Optional) Python virtual environment
```

## Potential Future Enhancements

*   User authentication for the web app.
*   More sophisticated source document display (e.g., highlighting relevant text).
*   Option to upload documents directly through the UI.
*   Integration with other data sources beyond PDFs.
*   Advanced error handling and logging for production deployment.
*   UI for managing configuration settings. 