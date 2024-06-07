import json
import hashlib
from datetime import datetime
import importlib.util
from flask import request, abort
import time
import logging
import openai
import os
import sqlite3
import assistant
from packaging import version

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from pathlib import Path
from pprint import pprint
from langchain_openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langdetect import detect

import nltk
from nltk.tokenize import word_tokenize
# Download the NLTK sentence tokenizer model
nltk.download('punkt')

## test message log
import sqlite3
from datetime import datetime

# Define the path to the resources folder
resources = 'resources'


CUSTOM_API_KEY = os.environ.get('CUSTOM_API_KEY')
mappings_db_path = '.storage/chat_mappings.db'

# Retry settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(Exception)
}

# Initialize a mapping db for chat mappings
def initialize_mapping_db():
  conn = sqlite3.connect(mappings_db_path)
  cursor = conn.cursor()
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS chat_mappings (
          integration TEXT,
          assistant_id TEXT,
          chat_id TEXT,
          thread_id TEXT,
          date_of_creation TIMESTAMP,
          PRIMARY KEY (integration, chat_id)
      )
  ''')
  cursor.execute('''
      CREATE TABLE IF NOT EXISTS chat_logs (
          chat_id TEXT,
          assistant_id TEXT,
          thread_id TEXT,
          message_id TEXT,
          message_content TEXT,
          timestamp TIMESTAMP
      )
  ''')
  conn.commit()
  conn.close()


# Save chat mapping to the db
def get_chat_mapping(integration, chat_id=None, assistant_id=None):
  conn = sqlite3.connect(mappings_db_path)
  cursor = conn.cursor()

  query = "SELECT * FROM chat_mappings WHERE integration = ?"
  params = [integration]

  if chat_id:
    query += " AND chat_id = ?"
    params.append(chat_id)
  elif assistant_id:
    query += " AND assistant_id = ?"
    params.append(assistant_id)

  # Add order by and limit to fetch the last 10 rows
  query += " ORDER BY date_of_creation DESC LIMIT 10"

  cursor.execute(query, params)
  rows = cursor.fetchall()

  conn.close()

  return [
      dict(
          zip([
              "integration", "assistant_id", "chat_id", "thread_id",
              "date_of_creation"
          ], row)) for row in rows
  ]

# Gets a specific value from the db result
def get_value_from_mapping(data, key):
  if data and isinstance(data, list) and isinstance(data[0], dict):
    return data[0].get(key)
  return None


# Adds or updates a chat mapping
def update_chat_mapping(integration, chat_id, assistant_id, thread_id):
  conn = sqlite3.connect(mappings_db_path)
  cursor = conn.cursor()

  date_of_creation = datetime.now()

  cursor.execute(
      '''
      INSERT OR REPLACE INTO chat_mappings (integration, chat_id, assistant_id, thread_id, date_of_creation)
      VALUES (?, ?, ?, ?, ?)
  ''', (integration, chat_id, assistant_id, thread_id, date_of_creation))

  conn.commit()
  conn.close()


def delete_chat_mapping(integration, chat_id):
  conn = sqlite3.connect(mappings_db_path)
  cursor = conn.cursor()

  cursor.execute(
      'DELETE FROM chat_mappings WHERE integration = ? AND chat_id = ?',
      (integration, chat_id))

  conn.commit()
  conn.close()


# Function to check API key
def check_api_key():
  api_key = request.headers.get('X-API-KEY')
  if api_key != CUSTOM_API_KEY:
    abort(401)  # Unauthorized access


# Check the current OpenAI version
def check_openai_version():
  required_version = version.parse("1.1.1")
  current_version = version.parse(openai.__version__)
  if current_version < required_version:
    raise ValueError(
        f"Error: OpenAI version {openai.__version__} is less than the required version 1.1.1"
    )
  else:
    logging.info("OpenAI version is compatible.")

@retry(**retry_settings)
def process_tool_call(client, thread_id, run_id, tool_call, tool_data):
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        logging.error(
            f"JSON decoding failed: {e.msg}. Input: {tool_call.function.arguments}"
        )
        arguments = {}  # Set to default value

    if function_name in tool_data["function_map"]:
        function_to_call = tool_data["function_map"][function_name]
        output = function_to_call(arguments)
        client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run_id, tool_outputs=[{
            "tool_call_id": tool_call.id,
            "output": json.dumps(output)
        }])
    else:
        logging.warning(f"Function {function_name} not found in tool data.")

# Process the actions that are initiated by the assistants API
def process_tool_calls(client, thread_id, run_id, tool_data):
    start_time = time.time()  # Record the start time

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > 20:
            logging.info("Breaking loop after 20 seconds timeout")
            break

        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id,
                                                       run_id=run_id)
        if run_status.status == 'completed':
            break
        elif run_status.status == 'requires_action':
            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                try:
                    process_tool_call(client, thread_id, run_id, tool_call, tool_data)
                except Exception as e:
                    logging.error(f"Failed to process tool call: {e}")
            time.sleep(2)
  
# Function to load tools from a file
def load_tools_from_directory(directory):
  tool_data = {"tool_configs": [], "function_map": {}}

  for filename in os.listdir(directory):
    if filename.endswith('.py'):
      module_name = filename[:-3]
      module_path = os.path.join(directory, filename)
      spec = importlib.util.spec_from_file_location(module_name, module_path)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)

      # Load tool configuration
      if hasattr(module, 'tool_config'):
        tool_data["tool_configs"].append(module.tool_config)

      # Map functions
      for attr in dir(module):
        attribute = getattr(module, attr)
        if callable(attribute) and not attr.startswith("__"):
          tool_data["function_map"][attr] = attribute

  return tool_data


# Function to dynamically import modules from a folder
def import_integrations():
  directory = 'integrations'
  modules = {}
  for filename in os.listdir(directory):
    if filename.endswith('.py') and not filename.startswith('__'):
      module_name = filename[:-3]
      module_path = os.path.join(directory, filename)
      spec = importlib.util.spec_from_file_location(module_name, module_path)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)

      # Setup routes for each integration
      if hasattr(module, 'setup_routes'):
        modules[module_name] = module
  return modules


# Creates a hashsum based on a given folder
def generate_hashsum(path, hash_func=hashlib.sha256):
  """
  Generates a hashsum for a file or all the files in a directory.
  
  :param path: Path to the file or folder.
  :param hash_func: Hash function to use, default is sha256.
  :return: Hexadecimal hashsum.
  """
  if not os.path.exists(path):
    raise ValueError("Path does not exist.")

  hashsum = hash_func()

  if os.path.isfile(path):
    # If it's a file, read and hash the file content
    with open(path, 'rb') as f:
      while chunk := f.read(8192):
        hashsum.update(chunk)
  elif os.path.isdir(path):
    # If it's a folder, iterate through all files and update hash
    for subdir, dirs, files in os.walk(path):
      for file in sorted(files):
        filepath = os.path.join(subdir, file)

        # Read file in binary mode and update hash
        with open(filepath, 'rb') as f:
          while chunk := f.read(8192):
            hashsum.update(chunk)
  else:
    raise ValueError("Path is not a file or directory.")

  return hashsum.hexdigest()


# Compares two given checksumssav
def compare_checksums(checksum1, checksum2):
  """
    Compares two checksums.

    :param checksum1: First checksum to compare.
    :param checksum2: Second checksum to compare.
    :return: Boolean indicating if checksums are identical.
    """
  return checksum1 == checksum2


# Function to load JSON files and extract text data
def load_json_files(folder):
  documents = []
  for filename in os.listdir(folder):
      if filename.endswith('.json'):
          file_path = os.path.join(folder, filename)
          # Use JSONLoader to load and extract content from JSON files
          loader = JSONLoader(
              file_path=file_path,
              jq_schema=".[]",  # Extract each object in the array
              text_content=False
          )
          documents.extend(loader.load())
  return documents

def split_documents(documents, max_chunk_size=1000):
  # Ensure JSON is parsed correctly before splitting
  parsed_jsons = [json.loads(doc.page_content) for doc in documents]
  splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size)
  return splitter.create_documents(json_data=parsed_jsons)

def initialize_vector_store(documents, embeddings, save_path):
  vector_store = FAISS.from_documents(documents, embeddings)
  vector_store.save_local(save_path)
  return vector_store
  

def load_vector_store(load_path, embeddings):
  return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

def detect_language(text):
    try:
        return str(detect(text))
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

      # Example usage #dutch: "Hallo, waar kan ik een leuke cafe vinden in het buurt van el         clot? "
      #sample_text = "I si tinguéssim algú que pogués recopilar i conèixer tota la         informació sobre el veïnat que no està a Google, i a qui poguéssim preguntar o explicar qualsevol cosa sempre que volguéssim? "
      ##language = detect_language(sample_text)
      #print(f"The detected language is: {language}")


def chunk_input_message(input_text, chunk_size=3):
  """
  Tokenize and chunk input text into smaller meaningful chunks (every 3 words as a separate chunk).
  """
  words = word_tokenize(input_text)
  chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
  return chunks

def chunk_input_message2(input_text, chunk_size=8):
  """
  Tokenize and chunk input text into smaller meaningful chunks (every 3 words as a separate chunk).
  """
  words = word_tokenize(input_text)
  chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
  return chunks
