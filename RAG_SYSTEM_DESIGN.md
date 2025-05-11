# RAG System Design for PDF Circulars

This document outlines the proposed file structure and key components for building a Retrieval Augmented Generation (RAG) system to query PDF circulars. The system will heavily leverage Langchain, use configurable local or API-based embeddings, and an API-based LLM for answer generation.

**Project Root:** `/Users/sankalp.phadnis/Documents/Coding/ComplianceGPT/`

---

## 1. Configuration File

*   **File:** `config.py`
*   **Purpose:** To store all configurable parameters, making it easy to adjust settings without digging into the code.
*   **Key Variables (Examples):
    *   `VECTOR_STORE_DIRECTORY: str`: Local path to save/load the ChromaDB vector store (e.g., `"vector_store_db"`).
    *   `EMBEDDING_PROVIDER: str`: Choice of embedding provider (e.g., `"openai"`, `"huggingface"`).
    *   `HUGGINGFACE_EMBEDDING_MODEL_NAME: str`: Identifier for local Sentence Transformers model if provider is `"huggingface"`.
    *   `OPENAI_API_KEY: str`: API key for OpenAI, loaded from `.env`.
    *   `OPENAI_EMBEDDING_MODEL_NAME: str`: OpenAI embedding model name if provider is `"openai"`.
    *   `CHUNK_SIZE: int`: Target size for text chunks (e.g., `1000`).
    *   `CHUNK_OVERLAP: int`: Overlap between text chunks (e.g., `200`).
    *   `LLM_PROVIDER: str`: To choose between LLM APIs (e.g., `"groq"`, `"google_gemini"`, `"openai"`).
    *   `GROQ_API_KEY: str`, `GROQ_MODEL_NAME: str`: Groq settings.
    *   `GOOGLE_API_KEY: str`, `GEMINI_MODEL_NAME: str`: Google Gemini settings.
    *   `OPENAI_CHAT_MODEL_NAME: str`: OpenAI chat model name if LLM provider is `"openai"`.
    *   `INGESTION_STATUS_FILE: str`: Path to the JSON file tracking processed PDFs (e.g., `"ingestion_status.json"`).
    *   `FILE_PROCESSING_BATCH_SIZE: int`: Number of PDF files to process in one batch during ingestion (e.g., `10`).

---

## 2. Document Ingestion and Vector Store Management

*   **File:** `ingestion_pipeline.py`
*   **Purpose:** Handles loading PDF documents, splitting them, generating embeddings, and creating/updating the vector store in a **resumable, file-batched manner**.
*   **Key Helper Functions (Conceptual):
    *   `load_ingestion_status(status_file_path: str) -> set[str]`:
        *   Loads a set of successfully processed PDF file paths from the status JSON file.
        *   Returns an empty set if the file doesn't exist.
    *   `save_ingestion_status(status_file_path: str, processed_files: set[str])`:
        *   Saves the given set of processed PDF file paths to the status JSON file.
*   **Core Functions:**
    *   `load_and_split_single_pdf(pdf_path: str, text_splitter: RecursiveCharacterTextSplitter) -> list[langchain_core.documents.Document]`:
        *   Loads a single PDF using `PyMuPDFLoader`.
        *   Splits it into chunks using the provided `text_splitter`.
        *   Returns a list of `Document` chunks for that PDF.
    *   `get_embedding_model() -> langchain_core.embeddings.Embeddings`:
        *   Initializes and returns an embedding model (`HuggingFaceEmbeddings` or `OpenAIEmbeddings`) based on `config.EMBEDDING_PROVIDER`.
        *   For `OpenAIEmbeddings`, configures its internal batch size (`chunk_size` parameter) for API calls.
    *   `run_full_ingestion_process()`:
        *   **Orchestration Logic:**
            1.  Loads the set of already processed PDF file paths from `config.INGESTION_STATUS_FILE`.
            2.  Identifies new PDF files in the `DOCUMENTS_PATH` (hardcoded or from config) that are not in the processed set.
            3.  If no new files, exits.
            4.  Initializes the embedding model once using `get_embedding_model()`.
            5.  Initializes a `Chroma` vector store instance, configured to persist to `config.VECTOR_STORE_DIRECTORY` and using the initialized embedding function. This will load an existing store if present, or create it on first addition.
            6.  Iterates through the new PDF files in batches (e.g., `config.FILE_PROCESSING_BATCH_SIZE` files at a time).
            7.  For each batch of PDF files:
                a.  Collects all text chunks from these files by calling `load_and_split_single_pdf` for each.
                b.  If chunks are collected for the current file batch:
                    i.  Attempts to add these chunks to the `Chroma` vector store using `vector_store.add_documents(chunks_for_this_file_batch)`. The embedding model (e.g., `OpenAIEmbeddings`) will handle its own internal batching for API calls efficiently.
                    ii. If successful, adds the paths of the files in the current batch to the set of processed files and immediately saves this updated set to `config.INGESTION_STATUS_FILE`.
                    iii. Calls `vector_store.persist()` if necessary to ensure data is written to disk (may depend on Chroma version and usage).
                    iv. If `add_documents` fails (e.g., API error), logs the error and stops or skips to the next batch (depending on error handling strategy). The status file is *not* updated for the failed batch, so it will be retried on the next run.
            8.  Reports completion.

*   **Note on `create_and_store_embeddings`:** This function (as previously designed for a one-shot full store creation) will likely be refactored. Its core logic of adding documents to Chroma will be integrated into the new batched loop within `run_full_ingestion_process()` using `vector_store.add_documents()`.

---

## 3. LLM and Retrieval-Augmented Generation

*   **File:** `rag_query_handler.py`
*   **Purpose:** Manages loading the existing vector store, initializing the chosen API-based LLM, and constructing/running a **conversational RAG chain** to support chat history.
*   **Key Functions:**
    *   `load_existing_vector_store(embedding_model: langchain_core.embeddings.Embeddings) -> langchain_community.vectorstores.Chroma`:
        *   Loads the persisted Chroma vector store.
    *   `get_llm_chat_client() -> langchain_core.language_models.chat_models.BaseChatModel`:
        *   Initializes the chosen LLM (Groq, Gemini, OpenAI, etc.) based on `config.py`.
    *   `setup_conversational_rag_chain(vector_store: langchain_community.vectorstores.Chroma, llm_client: langchain_core.language_models.chat_models.BaseChatModel) -> langchain.chains.ConversationalRetrievalChain`:
        *   Constructs and returns a `ConversationalRetrievalChain` using the LLM and the vector store retriever.
        *   This chain is designed to take chat history into account for contextual responses.
    *   `query_conversational_rag_chain(chain: langchain.chains.ConversationalRetrievalChain, question: str, chat_history: list) -> dict`:
        *   Queries the `ConversationalRetrievalChain` with the current `question` and formatted `chat_history`.
        *   Returns the response, which typically includes the 'answer' and potentially source documents.

---

## 4. Main Application / User Interface

*   **File:** `app.py`
*   **Purpose:** Provides a web-based chat interface titled "Groww ComplianceGPT" using Streamlit. It allows users to ask questions about the ingested documents and receive answers in a conversational manner.
*   **Key Features/Logic:**
    *   **UI:** Built with Streamlit, featuring a chat input box, a message display area, and a "Clear Chat Thread" button.
    *   **Title:** "Groww ComplianceGPT".
    *   **Initialization (Cached):**
        *   Uses `ingestion_pipeline.get_embedding_model()` to load the correct embedding model.
        *   Uses `rag_query_handler.load_existing_vector_store()` to load the vector store.
        *   Uses `rag_query_handler.get_llm_chat_client()` to initialize the LLM.
        *   Uses `rag_query_handler.setup_conversational_rag_chain()` to get the conversational chain.
        *   These components are loaded once using Streamlit's caching (`@st.cache_resource`).
    *   **Chat State Management:**
        *   Maintains chat history (user messages and assistant responses) in `st.session_state.messages`.
        *   Displays the conversation history in the UI.
    *   **Clear Chat Button:**
        *   A button labeled "Clear Chat Thread" allows the user to reset the conversation.
        *   On click, `st.session_state.messages` is cleared, an initial assistant message is re-added, and `st.rerun()` is called to refresh the UI.
    *   **Query Processing:**
        *   When the user submits a question:
            1.  Appends the user's question to the session state chat history.
            2.  Formats the chat history appropriately for the `ConversationalRetrievalChain`.
            3.  Calls `rag_query_handler.query_conversational_rag_chain()` with the question and formatted history.
            4.  Appends the assistant's answer to the session state chat history.
            5.  Displays the new messages in the UI.
    *   **No Ingestion:** This application does not run the ingestion pipeline. Ingestion is handled by running `ingestion_pipeline.py` separately.

---

## 5. Dependencies

*   **File:** `requirements.txt`
*   **Content (Example):** (Includes `langchain-openai` and `streamlit`)
    ```
    langchain
    langchain-community
    langchain-core
    langchain-groq
    langchain-google-genai
    langchain-openai 
    PyMuPDF
    sentence-transformers
    chromadb
    python-dotenv
    streamlit # For the web UI
    # faiss-cpu
    ```

---

## 6. Environment Variables

*   **File:** `.env`
*   **Content (Example):** (Includes `OPENAI_API_KEY`)
    ```
    OPENAI_API_KEY="your_openai_api_key"
    GROQ_API_KEY="your_groq_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    ```

---

## 7. Git Ignore

*   **File:** `.gitignore`
*   **Content (Example):** (Should include `ingestion_status.json`)
    ```
    __pycache__/
    *.pyc
    .env
    vector_store_db/
    venv/
    *.log
    Documents/
    ingestion_status.json
    ``` 