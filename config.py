# Configuration settings for the RAG system
import os
from dotenv import load_dotenv

load_dotenv()

# --- Document Ingestion Settings ---
# DOCUMENTS_PATH = "Documents"  # Relative path to the folder containing PDFs - This will be hardcoded in ingestion_pipeline.py
VECTOR_STORE_DIRECTORY = "vector_store_db"  # Directory to store ChromaDB

# --- Embedding Model Settings ---
# Choose "huggingface" for local Sentence Transformers or "openai" for OpenAI API embeddings
EMBEDDING_PROVIDER = "openai"  # Options: "huggingface", "openai"

# Settings for HuggingFace Sentence Transformers (if EMBEDDING_PROVIDER is "huggingface")
HUGGINGFACE_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Example Sentence Transformer model

# Settings for OpenAI Embeddings (if EMBEDDING_PROVIDER is "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Loaded from .env file
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Recommended default, good balance of cost/performance
# Other OpenAI embedding models: "text-embedding-ada-002" (older, cheaper), "text-embedding-3-large" (more powerful, more expensive)


# --- Text Splitting Settings ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# --- LLM Provider Settings ---
# Choose between "groq", "google_gemini", "openai" or add your own
LLM_PROVIDER = "groq" # Example: use "groq"

# API Keys & Model Names for LLMs - Loaded from .env file where applicable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL_NAME = "mixtral-8x7b-32768" # Decommissioned
GROQ_MODEL_NAME = "llama3-70b-8192" # Updated to a current model on Groq

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# OPENAI_API_KEY is already defined above for embeddings, can be reused for ChatOpenAI if LLM_PROVIDER is "openai"
OPENAI_CHAT_MODEL_NAME = "gpt-3.5-turbo" # Default, can be changed

# --- Ingestion Settings ---
DOCUMENTS_PATH = "Documents"  # Path to the folder containing PDF circulars
INGESTION_STATUS_FILE = "ingestion_status.json" # Tracks processed PDFs
FILE_PROCESSING_BATCH_SIZE = 5 # Number of PDFs to process in one batch

# --- Sanity Checks (Optional but Recommended) ---
if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY not found. Please set it in your .env file.")

# Example check for LLM provider (can be expanded)
# if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
#     raise ValueError("LLM_PROVIDER is 'groq' but GROQ_API_KEY not found. Please set it in your .env file.")
# elif LLM_PROVIDER == "google_gemini" and not GOOGLE_API_KEY:
#     raise ValueError("LLM_PROVIDER is 'google_gemini' but GOOGLE_API_KEY not found. Please set it in your .env file.")
# elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY: # Assuming same key for chat
#     raise ValueError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY not found for chat. Please set it in your .env file.")

print(f"config.py loaded. Embedding provider: {EMBEDDING_PROVIDER}, LLM provider: {LLM_PROVIDER}") 