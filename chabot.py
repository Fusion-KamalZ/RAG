# chatbot_app.py
import streamlit as st
import os
import subprocess
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage # Import message types

# Assuming vector_store_manager.py is in the same directory
from vector_store_manager import VectorStoreManager, INDEX_DIR, format_docs, logger as manager_logger

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add a handler to also log to Streamlit's console area for easier debugging during development
# sh = logging.StreamHandler()
# sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logging.getLogger().addHandler(sh) # Add handler to root logger
app_logger = logging.getLogger(__name__)


# --- Available Models ---
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    # Add other models as needed and available
    # "gemini-2.5-flash-preview-04-17", # Example - check availability
    # "gemini-2.5-pro-preview-03-25",  # Example - check availability
]

# --- Initialization ---
load_dotenv()
app_logger.info("Streamlit app started.")

# --- Initialize Session State ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GOOGLE_API_KEY")
    app_logger.info(f"Initialized 'api_key' in session state (from .env: {'Yes' if st.session_state.api_key else 'No'}).")

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS[0]
    app_logger.info(f"Initialized 'selected_model' in session state to: {st.session_state.selected_model}")

if 'settings_saved' not in st.session_state:
    st.session_state.settings_saved = bool(st.session_state.api_key)

# --- Initialize Chat History (NEW) ---
if 'messages' not in st.session_state:
    st.session_state.messages = [] # Stores chat history: {'role': 'user'/'assistant', 'content': '...'}
    app_logger.info("Initialized 'messages' in session state.")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Contextual RAG Chatbot", layout="wide")
st.title("🤖 Contextual RAG Chatbot")
st.caption("Ask questions based on selected knowledge bases, with conversation history.")

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")

# --- API Key and Model Selection ---
st.sidebar.subheader("API Settings")
api_key_input = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    value=st.session_state.api_key or "",
    key="api_key_input_widget"
)

selected_model_input = st.sidebar.selectbox(
    "Select Gemini Chat Model:",
    options=AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in AVAILABLE_MODELS else 0,
    key="model_select_widget"
)

if st.sidebar.button("Save Settings"):
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.selected_model = selected_model_input
        st.session_state.settings_saved = True
        app_logger.info(f"Settings saved. API Key {'set'}. Model: {selected_model_input}")
        st.sidebar.success("Settings saved!")
        # Clear chat history when settings change? Optional, but might make sense.
        # st.session_state.messages = []

        # Clear all resource caches since API key/model changed
        st.cache_resource.clear()
        app_logger.info("Cleared all cached resources (@st.cache_resource).")

        st.rerun() # Rerun to re-initialize with new settings
    else:
        st.sidebar.error("API Key cannot be empty.")
        st.session_state.settings_saved = False

st.sidebar.divider()

# --- Conditional Initialization & Main App Logic ---
if st.session_state.get("api_key") and st.session_state.get("settings_saved"):

    # --- Initialize Models (Cached based on API Key and Model) ---
    @st.cache_resource
    def get_embeddings_model(key):
        app_logger.info(f"Initializing Embeddings Model for session...")
        try:
            # Ensure API key is passed correctly if needed by the constructor
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        except Exception as e:
            st.error(f"Failed to initialize Embeddings Model: {e}")
            app_logger.error(f"Embeddings initialization failed: {e}", exc_info=True)
            return None

    @st.cache_resource
    def get_llm(key, model_name):
        app_logger.info(f"Initializing LLM '{model_name}' for session...")
        try:
            # Consider adding max_output_tokens if needed
            # convert_system_message_to_human=True might be needed depending on Langchain/Gemini version interaction
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.7, convert_system_message_to_human=True)
        except Exception as e:
            st.error(f"Failed to initialize LLM '{model_name}': {e}")
            app_logger.error(f"LLM initialization failed for model {model_name}: {e}", exc_info=True)
            return None

    # --- Initialize Vector Store Manager (Cached based on embeddings instance) ---
    @st.cache_resource(show_spinner="Initializing Knowledge Base Manager...")
    def get_vector_store_manager(_embeddings_instance):
        if _embeddings_instance is None:
             app_logger.error("Cannot initialize VectorStoreManager: Embeddings model is None.")
             return None
        app_logger.info("Initializing Vector Store Manager for session...")
        try:
            manager = VectorStoreManager(
                index_base_dir=INDEX_DIR,
                embeddings_model=_embeddings_instance
            )
            return manager
        except Exception as e:
            st.error(f"Failed to initialize Vector Store Manager: {e}")
            app_logger.error(f"Vector Store Manager initialization failed: {e}", exc_info=True)
            return None

    # --- Get models and manager ---
    current_api_key = st.session_state.api_key
    current_model = st.session_state.selected_model

    embeddings = get_embeddings_model(current_api_key)
    llm = get_llm(current_api_key, current_model)
    vector_manager = get_vector_store_manager(embeddings)

    # --- Proceed only if all core components initialized ---
    if embeddings and llm and vector_manager:
        app_logger.info("Core components (Embeddings, LLM, Manager) initialized successfully.")

        # --- Section: Add New Knowledge Base (Sidebar) ---
        st.sidebar.subheader("Add New Source")
        new_kb_url = st.sidebar.text_input("Enter URL to crawl:", key="kb_url_input")
        new_kb_name = st.sidebar.text_input("Enter a unique name for this source:", key="kb_name_input")

        if st.sidebar.button("Create Knowledge Base", key="create_kb_button"):
            app_logger.info(f"Create KB button clicked. URL: {new_kb_url}, Name: {new_kb_name}")
            if new_kb_url and new_kb_name:
                # Basic validation for name (avoiding path traversal issues)
                if not all(c.isalnum() or c in ('_', '-') for c in new_kb_name) or new_kb_name in ('.', '..'):
                     st.sidebar.error("Invalid name. Use letters, numbers, underscores, hyphens only.")
                     app_logger.warning(f"Invalid KB name attempt: {new_kb_name}")
                else:
                    app_logger.info(f"Checking existence of KB: {new_kb_name}")
                    existing_indexes = vector_manager.get_available_indexes()
                    if new_kb_name in existing_indexes:
                         st.sidebar.error(f"Knowledge base '{new_kb_name}' already exists.")
                         app_logger.warning(f"Attempted to create existing KB: {new_kb_name}")
                    else:
                        # Use an expander for progress in the sidebar
                        with st.sidebar.expander(f"Creating '{new_kb_name}'...", expanded=True):
                            st.info(f"Processing {new_kb_name} from {new_kb_url}...")
                            markdown_content = None
                            error_message = None
                            try:
                                process_env = os.environ.copy()
                                process_env["PYTHONIOENCODING"] = "utf-8"
                                st.info(f"[{new_kb_name}] Starting crawler...")
                                app_logger.info(f"Executing crawler script: {sys.executable} crawler_script.py {new_kb_url}")
                                # Ensure crawler_script.py path is correct if not in the same dir
                                crawler_script_path = os.path.join(os.path.dirname(__file__), "crawler_script.py")
                                if not os.path.exists(crawler_script_path):
                                     raise FileNotFoundError(f"crawler_script.py not found at {crawler_script_path}")

                                process = subprocess.run(
                                    [sys.executable, crawler_script_path, new_kb_url],
                                    capture_output=True, text=True, encoding='utf-8',
                                    check=True, timeout=180, env=process_env # Increased timeout
                                )
                                markdown_content = process.stdout
                                st.info(f"[{new_kb_name}] Crawling finished.")
                                app_logger.info(f"Crawler script successful for {new_kb_name}. Output length: {len(markdown_content)}")
                            except subprocess.CalledProcessError as e:
                                error_message = f"Crawler script failed for {new_kb_name}.\nError:\n{e.stderr}"
                                st.error(error_message); app_logger.error(f"Crawler CalledProcessError: {e.stderr}")
                            except subprocess.TimeoutExpired:
                                error_message = f"Crawler script timed out for {new_kb_name} after 180 seconds."
                                st.error(error_message); app_logger.error("Crawler TimeoutExpired")
                            except FileNotFoundError as e:
                                 error_message = f"Error: {e}."; st.error(error_message); app_logger.error(str(e))
                            except Exception as e:
                                error_message = f"Unexpected error running crawler: {e}"; st.error(error_message); app_logger.exception("Crawler subprocess error")

                            if markdown_content:
                                st.info(f"[{new_kb_name}] Creating vector index...")
                                app_logger.info(f"Calling create_new_knowledge_base for {new_kb_name}")
                                try:
                                    created_store = vector_manager.create_new_knowledge_base(new_kb_name, markdown_content)
                                    if created_store:
                                        st.success(f"Knowledge base '{new_kb_name}' created!")
                                        app_logger.info(f"Vector store creation successful for {new_kb_name}")
                                        st.rerun() # Rerun to update the list of KBs
                                    else:
                                        # create_new_knowledge_base should log the specific error
                                        st.error(f"Failed to create vector store for '{new_kb_name}'. Check logs."); app_logger.error("Vector store creation failed (returned None).")
                                except Exception as e:
                                     st.error(f"Error during index creation process: {e}")
                                     app_logger.exception(f"Error calling create_new_knowledge_base for {new_kb_name}")

                            elif not error_message:
                                # Handle case where crawler succeeded but returned no content
                                st.warning(f"[{new_kb_name}] Crawler ran successfully but returned no content. Index not created.")
                                app_logger.warning(f"Crawler returned no content for {new_kb_name}.")
                            else:
                                # Error message already shown from crawler exception block
                                st.warning(f"[{new_kb_name}] Skipping index creation due to crawler error.")
                                app_logger.warning(f"Skipping vector store creation for {new_kb_name} due to crawler error.")
            else:
                st.sidebar.warning("Please enter both a URL and a unique name.")
                app_logger.warning("Create KB clicked with missing URL or Name.")

        # --- Section: Select Existing Knowledge Bases (Sidebar) ---
        st.sidebar.divider()
        st.sidebar.subheader("Query Sources")
        st.sidebar.write("Select sources to query:")

        available_indexes = vector_manager.get_available_indexes()
        selected_kb_names = [] # Track selected KBs for the current query

        if not available_indexes:
            st.sidebar.info("No knowledge bases created yet. Add one above.")
        else:
            # Use session state to remember selections across reruns if desired
            # For simplicity here, we just read the checkboxes each time
            for index_name in available_indexes:
                # Default to False, user must check them each time or manage state
                if st.sidebar.checkbox(index_name, key=f"cb_{index_name}"):
                    selected_kb_names.append(index_name)

        # --- Define RAG Prompt Template (Hybrid Approach) ---
        # Note: Formatting history directly into the prompt string might be simpler
        # than relying on ChatPromptTemplate's message handling for some models/versions.
        # However, using the message format is generally preferred.
        rag_template_str = """You are a helpful assistant answering questions based on the provided context and chat history.

Use the following pieces of retrieved context from documentation to answer the question.
Prioritize information from the context if it directly addresses the question.
Consider the ongoing conversation history for context.
If the context does not contain the answer or is not relevant to the question, rely on your general knowledge and the chat history to provide a helpful response.
Synthesize information if it comes from multiple sources or context/history. Keep the answer concise and relevant to the user's query.

Chat History:
{chat_history_str}

Retrieved Context:
{context}

Question:
{question}

Answer:"""
        rag_prompt = ChatPromptTemplate.from_template(rag_template_str)


        # --- Main Chat Area ---
        st.markdown("---")

        # Display existing messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get new user input via chat interface
        if user_question := st.chat_input("Ask your question..."):
            app_logger.info(f"User question: '{user_question}' with selected KBs: {selected_kb_names}")

            # Add user message to history and display it immediately
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # --- Determine Response Strategy ---
            if not selected_kb_names:
                # Scenario 1: No KBs selected - Use direct LLM call
                app_logger.info("No KBs selected. Using direct LLM call.")
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking... 🤔 (using general knowledge)")
                    try:
                        # --- CORRECTED DIRECT LLM CALL ---
                        # Manually construct the message list for the LLM

                        # 1. Start with the system prompt (optional, can be part of history formatting)
                        # messages_for_llm = [
                        #     SystemMessage(content="You are a helpful assistant. Answer the following question based on the chat history and your general knowledge.")
                        # ]
                        messages_for_llm = [] # Start empty, history will include context

                        # 2. Add past messages from history
                        # Ensure we only add history if it exists, excluding the current user question already added to state
                        history_to_add = st.session_state.messages[:-1]
                        for msg in history_to_add:
                            if msg["role"] == "user":
                                messages_for_llm.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant": # Check role explicitly
                                messages_for_llm.append(AIMessage(content=msg["content"]))
                            # You might want to handle other potential roles or skip them

                        # 3. Add the current user question
                        messages_for_llm.append(HumanMessage(content=user_question))

                        app_logger.debug(f"Messages sent to LLM (direct): {[m.pretty_repr() for m in messages_for_llm]}") # Log the messages

                        # 4. Invoke the LLM directly with the message list
                        ai_response = llm.invoke(messages_for_llm)

                        # 5. Extract content from the AIMessage response
                        response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

                        # --- END CORRECTION ---

                        message_placeholder.markdown(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                        app_logger.info("Direct LLM call successful.")

                    except Exception as e:
                        error_msg = f"Error generating answer with LLM (general): {e}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"}) # Add error to history
                        app_logger.exception("Direct LLM execution error.")

            else:
                # Scenario 2: KBs selected - Attempt RAG
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown(f"Searching in: {', '.join(selected_kb_names)}... 🤔")

                    all_retrieved_docs = []
                    retrieval_errors = False
                    app_logger.info(f"Starting retrieval for question: '{user_question}'")
                    for name in selected_kb_names:
                        try:
                            retriever = vector_manager.get_retriever(name, k=3) # Retrieve slightly more docs
                            if retriever:
                                docs = retriever.invoke(user_question)
                                app_logger.info(f"Retrieved {len(docs)} docs from KB: {name}")
                                all_retrieved_docs.extend(docs)
                            else:
                                 # This case should ideally not happen if get_retriever handles errors
                                 st.warning(f"Could not get retriever for '{name}'.")
                                 app_logger.error(f"Failed to get retriever for {name} (returned None).")
                                 retrieval_errors = True # Non-fatal error for this KB
                        except Exception as e:
                             # Display error specific to this KB, but continue if others selected
                             st.warning(f"Error retrieving from '{name}': {e}")
                             app_logger.exception(f"Retrieval error from {name}")
                             retrieval_errors = True # Potentially more serious error

                    # --- Decide based on retrieval results ---
                    if all_retrieved_docs:
                        # Scenario 2a: RAG - Docs found
                        message_placeholder.markdown("Found relevant documents. Generating answer...")
                        app_logger.info(f"Generating answer using {len(all_retrieved_docs)} docs and chat history.")
                        try:
                            formatted_context = format_docs(all_retrieved_docs)

                            # Prepare history string for the prompt template
                            history_str = "\n".join(
                                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]]
                            )

                            # Define RAG chain using the string-based prompt
                            rag_chain = (
                                {
                                    "context": lambda x: formatted_context,
                                    "chat_history_str": lambda x: history_str,
                                    "question": RunnablePassthrough() # Passes the original question string
                                }
                                | rag_prompt # Use the string-based RAG prompt
                                | llm
                                | StrOutputParser()
                            )

                            app_logger.info("Invoking RAG chain with history...")
                            response = rag_chain.invoke(user_question) # Pass the user question string
                            app_logger.info("RAG chain invocation successful.")
                            message_placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                            # Optionally show context used
                            with st.expander("Show Context Used"):
                                st.markdown(formatted_context)

                        except Exception as e:
                            error_msg = f"Error generating answer with LLM (RAG): {e}"
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                            app_logger.exception("RAG chain/LLM execution error.")

                    elif not retrieval_errors:
                        # Scenario 2b: Fallback - No docs found, but no retrieval errors
                        message_placeholder.markdown("No specific info in selected documents. Trying general knowledge...")
                        app_logger.warning(f"No documents retrieved for: '{user_question}'. Attempting direct LLM call.")
                        try:
                            # --- Use the same direct call logic as Scenario 1 ---
                            messages_for_llm = []
                            history_to_add = st.session_state.messages[:-1]
                            for msg in history_to_add:
                                if msg["role"] == "user":
                                    messages_for_llm.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    messages_for_llm.append(AIMessage(content=msg["content"]))
                            messages_for_llm.append(HumanMessage(content=user_question))

                            app_logger.debug(f"Messages sent to LLM (fallback): {[m.pretty_repr() for m in messages_for_llm]}")
                            ai_response = llm.invoke(messages_for_llm)
                            response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
                            # --- End direct call logic ---

                            message_placeholder.markdown(response_content)
                            st.session_state.messages.append({"role": "assistant", "content": response_content})
                            app_logger.info("Direct LLM call successful (fallback).")

                        except Exception as e:
                            error_msg = f"Error generating answer with LLM (fallback): {e}"
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                            app_logger.exception("Direct LLM execution error (fallback).")
                    else:
                        # Scenario 2c: Error - Retrieval errors occurred for *all* selected KBs
                        # (or user wants to be notified even if some succeeded but others failed)
                        error_msg = "Could not retrieve information from one or more sources. Answering based on general knowledge and any successful retrievals (if any)."
                        # Decide if you want to proceed with fallback or just show error.
                        # Let's proceed with fallback for now.
                        message_placeholder.warning(error_msg + " Trying general knowledge...")
                        app_logger.error("Retrieval completed with errors. Attempting direct LLM call as fallback.")
                        try:
                            # --- Use the same direct call logic as Scenario 1 ---
                            messages_for_llm = []
                            history_to_add = st.session_state.messages[:-1]
                            for msg in history_to_add:
                                if msg["role"] == "user":
                                    messages_for_llm.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    messages_for_llm.append(AIMessage(content=msg["content"]))
                            messages_for_llm.append(HumanMessage(content=user_question))

                            app_logger.debug(f"Messages sent to LLM (error fallback): {[m.pretty_repr() for m in messages_for_llm]}")
                            ai_response = llm.invoke(messages_for_llm)
                            response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
                            # --- End direct call logic ---

                            message_placeholder.markdown(response_content) # Show the fallback answer
                            st.session_state.messages.append({"role": "assistant", "content": response_content}) # Add fallback answer to history
                            app_logger.info("Direct LLM call successful (error fallback).")

                        except Exception as e:
                            final_error_msg = f"Error generating answer with LLM (error fallback): {e}"
                            message_placeholder.error(final_error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": f"Error: {final_error_msg}"})
                            app_logger.exception("Direct LLM execution error (error fallback).")


    else:
         # Handle case where core components failed to initialize
         st.error("Core components (LLM, Embeddings, Manager) failed to initialize. Please check API key, model selection, and console logs. Restart the app after fixing.")
         app_logger.error("Core components failed to initialize. Halting main app execution.")
         # Add button to retry initialization?
         if st.button("Retry Initialization"):
             st.cache_resource.clear()
             st.rerun()


else:
    # Prompt user to enter API Key if not set
    st.info("👈 Please enter your Google AI API Key and select a model in the sidebar, then click 'Save Settings' to begin.")
    app_logger.info("API key not found in session state or settings not saved. Prompting user.")


# --- Footer ---
st.markdown("---")
st.caption("Contextual Multi-Source RAG Chatbot")