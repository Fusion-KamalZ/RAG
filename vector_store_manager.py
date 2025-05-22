# vector_store_manager.py
import os
import shutil
# Remove streamlit import if not strictly needed for UI messages here
# import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging # Use logging for background tasks

# Configure logging (ensure it's configured only once, maybe move to main app if needed)
# If running this file independently, this is fine. If imported, ensure main app configures root logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger for this module

# --- Configuration ---
INDEX_DIR = "faiss_indexes"

class VectorStoreManager:
    """Manages creation, loading, and retrieval from FAISS vector stores persisted on disk."""

    def __init__(self, index_base_dir, embeddings_model):
        logger.info(f"Initializing VectorStoreManager for directory: {index_base_dir}")
        self.index_base_dir = index_base_dir
        self.embeddings = embeddings_model
        self._loaded_stores = {}
        if not os.path.exists(self.index_base_dir):
            os.makedirs(self.index_base_dir)
            logger.info(f"Created base index directory: {self.index_base_dir}")

    def _create_and_save_index(self, index_name, source_content):
        """Internal method to create and save a new FAISS index."""
        folder_path = os.path.join(self.index_base_dir, index_name)
        logger.info(f"Starting index creation process for: {index_name} in {folder_path}")

        if not source_content:
             logger.error(f"Source content for {index_name} is empty. Cannot create index.")
             return None

        if not os.path.exists(folder_path):
             os.makedirs(folder_path)
             logger.info(f"Created folder for index: {folder_path}")

        try:
            logger.info(f"[{index_name}] Preparing document...")
            docs = [Document(page_content=source_content)]

            logger.info(f"[{index_name}] Initializing text splitter...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

            logger.info(f"[{index_name}] Splitting documents...")
            split_docs = text_splitter.split_documents(docs)
            logger.info(f"[{index_name}] Split into {len(split_docs)} chunks.")

            if not split_docs:
                logger.error(f"[{index_name}] Failed to split documents (0 chunks).")
                if os.path.exists(folder_path) and not os.listdir(folder_path):
                     shutil.rmtree(folder_path)
                return None

            # --- Critical Embedding Step ---
            logger.info(f"[{index_name}] Starting FAISS.from_documents (embedding documents)...")
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            logger.info(f"[{index_name}] FAISS.from_documents completed successfully.")
            # --------------------------------

            logger.info(f"[{index_name}] Saving FAISS index to local disk at {folder_path}...")
            vectorstore.save_local(folder_path)
            logger.info(f"[{index_name}] Index saved successfully.")
            return vectorstore

        except Exception as e:
            logger.exception(f"[{index_name}] Error during index creation/saving: {e}") # Log full traceback
            if os.path.exists(folder_path):
                 try:
                    # Attempt cleanup only if saving failed, embedding might have partially succeeded
                    if 'vectorstore' not in locals() or not os.path.exists(os.path.join(folder_path, "index.faiss")):
                         logger.warning(f"[{index_name}] Attempting to remove potentially incomplete folder: {folder_path}")
                         shutil.rmtree(folder_path)
                 except OSError as rm_err:
                     logger.error(f"[{index_name}] Failed to remove folder {folder_path} after creation error: {rm_err}")
            return None

    def load_index(self, index_name):
        """Loads a specific FAISS index from disk."""
        if index_name in self._loaded_stores:
             logger.debug(f"Returning cached index: {index_name}")
             return self._loaded_stores[index_name]

        folder_path = os.path.join(self.index_base_dir, index_name)
        index_path = os.path.join(folder_path, "index.faiss")

        if os.path.exists(folder_path) and os.path.exists(index_path):
            try:
                logger.info(f"Loading existing index from disk: {index_name}")
                vectorstore = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
                self._loaded_stores[index_name] = vectorstore
                logger.info(f"Index '{index_name}' loaded successfully.")
                return vectorstore
            except Exception as e:
                logger.exception(f"Error loading index {index_name} from {folder_path}: {e}")
                return None
        else:
            logger.warning(f"Index '{index_name}' not found at {folder_path}")
            return None

    def create_new_knowledge_base(self, index_name, source_content):
        """Public method to explicitly create a new knowledge base index."""
        logger.info(f"Request received to create new knowledge base: {index_name}")
        folder_path = os.path.join(self.index_base_dir, index_name)
        if os.path.exists(folder_path):
            logger.warning(f"Index '{index_name}' already exists at {folder_path}. Creation aborted.")
            # Use st.error in the main app based on the return value, not here.
            # st.error(f"Knowledge base '{index_name}' already exists...")
            return None # Indicate failure due to existence

        logger.info(f"Calling internal _create_and_save_index for: {index_name}")
        vectorstore = self._create_and_save_index(index_name, source_content)

        if vectorstore:
            logger.info(f"Caching newly created vector store for: {index_name}")
            self._loaded_stores[index_name] = vectorstore
            logger.info(f"Successfully created and cached knowledge base: {index_name}")
        else:
            logger.error(f"Failed to create knowledge base: {index_name}")

        return vectorstore # Return the store or None if creation failed

    def get_retriever(self, index_name, k=2):
        """Gets a retriever for the specified index name (loads if not cached)."""
        logger.debug(f"Requesting retriever for index: {index_name}")
        vectorstore = self.load_index(index_name)
        if vectorstore:
            logger.debug(f"Returning retriever for index: {index_name} with k={k}")
            return vectorstore.as_retriever(search_kwargs={'k': k})
        else:
            logger.error(f"Could not get vector store for index '{index_name}' to create retriever.")
            return None

    def get_available_indexes(self):
        """Returns a sorted list of available index names based on directories."""
        logger.debug(f"Checking for available indexes in: {self.index_base_dir}")
        if not os.path.exists(self.index_base_dir):
            logger.warning(f"Index base directory not found: {self.index_base_dir}")
            return []
        try:
            indexes = sorted([
                d for d in os.listdir(self.index_base_dir)
                if os.path.isdir(os.path.join(self.index_base_dir, d))
            ])
            logger.debug(f"Found available indexes: {indexes}")
            return indexes
        except FileNotFoundError:
            logger.error(f"FileNotFoundError while listing indexes in: {self.index_base_dir}")
            return []

# Keep format_docs here or move to chatbot_app.py, doesn't matter much
def format_docs(docs):
    """Combines page content of retrieved documents."""
    if not docs:
        return "No relevant context found."
    # Add a separator
    return "\n\n---\n\n".join(doc.page_content for doc in docs)