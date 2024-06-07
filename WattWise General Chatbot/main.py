import os
import logging
from flask import Flask, render_template
import openai
import core_functions
import assistant

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from pathlib import Path
from pprint import pprint

from flask import Flask, request, jsonify
#retry settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import signal
import sys
#from  integrations.Telegram_V2 import start_bot  # Import the start_bot function from telegramV2

retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(Exception)
}

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check OpenAI version compatibility
core_functions.check_openai_version()

# Create Flask app
app = Flask(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found in environment variables")


@retry(**retry_settings)
def initialize_openai_client(api_key):
    return openai.OpenAI(api_key=api_key)


try:
    client = initialize_openai_client(OPENAI_API_KEY)
    logging.info("OpenAI client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Initialize all available tools
tool_data = core_functions.load_tools_from_directory('tools')

#added for the vector storage
# Define the path to the resources folder & vectors
resources = 'resources'
vector_store_path = 'docs/static'

# Load documents
raw_documents = core_functions.load_json_files(resources)

# Print the raw documents for debugging
print("Raw Documents:")
for doc in raw_documents:
    print(
        doc.page_content[:200])  # Print first 200 characters of each document

# Directly use raw documents without further splitting
documents = raw_documents

# Verify the documents
print("Split Documents:")
for doc in documents[:5]:  # Display first 5 chunks for verification
    print(doc.page_content[:200])  # Print first 200 characters of each chunk

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create and save vector store locally
vector_store = core_functions.initialize_vector_store(documents, embeddings,
                                                      vector_store_path)
print(f"Vector store saved locally at: {vector_store_path}")

# Load the vector store for verification
loaded_vector_store = core_functions.load_vector_store(vector_store_path,
                                                       embeddings)
print("Vector store loaded successfully")

# Create or load assistant
assistant_id = assistant.create_assistant(client, tool_data,
                                          loaded_vector_store)

if not assistant_id:
    raise ValueError(f"No assistant found by id: {assistant_id}")

# Import integrations
available_integrations = core_functions.import_integrations()

requires_db = False

# Dynamically set up routes for active integrations
for integration_name in available_integrations:
    integration_module = available_integrations[integration_name]
    integration_module.setup_routes(app, client, tool_data, assistant_id)

    #Checks whether or not a DB mapping is required
    if integration_module.requires_mapping():
        requires_db = True

# Maybe initialize the SQLite DB structure
if requires_db:
    core_functions.initialize_mapping_db()


# Display a simple web page for simplicity
@app.route('/')
def home():
    return render_template('index.html')


# Ensure graceful shutdown
def signal_handler(sig, frame):
    logging.info("Signal received, shutting down")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    # Start keep-alive mechanism
    app.run(host='0.0.0.0', port=8080)
    #start_bot()
