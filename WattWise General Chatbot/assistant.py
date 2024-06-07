# get_assistant_instructions():

#  This function reads the contents of the assistant/instructions.txt file and returns the instructions for the assistant.

#create_assistant(client, tool_data):

# This function is responsible for creating or loading an assistant based on certain conditions.
# It checks if the assistant.json file exists. If it does, it loads the assistant.
#It generates hash sums for tools, resources, assistant's code, and instructions.
#It compares the current assistant data hash sums with the saved data hash sums to determine if an update is needed.
#If an update is required, it fetches resource file IDs, creates a vector store, and updates the assistant with the new information.
#If no update is needed or if the assistant.json file doesn't exist, it creates a new assistant, generates hash sums, and saves the assistant data to assistant.json.

#save_assistant_data(assistant_data, file_path):

#This function saves the assistant data (such as assistant ID,   hash sums) into a JSON file located at the specified file path.

#is_valid_assistant_data(assistant_data):

#This function checks if the assistant data contains valid values for all the required keys (assistant_id, tools_sum, resources_sum, assistant_sum).

#compare_assistant_data_hashes(current_data, saved_data):

#This function compares the current assistant data hash sums with the saved data hash sums loaded from a JSON file.
#It first checks if the saved data is valid.
#It then compares the hash sums for tools, resources, assistant's code, and instructions to determine if they match."

import os
import core_functions
import json
from langsmith import traceable
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = "WattWise AI Chatbot"

# This is the storage path for the new assistant.json file
assistant_file_path = '.storage/assistant.json'
assistant_name = "WattWise AI"
assistant_instructions_path = 'assistant/instructions.txt'

# Retry settings
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(Exception)
}

# Get the instructions for the assistant
def get_assistant_instructions():
  # Open the file and read its contents
  with open(assistant_instructions_path, 'r') as file:
    return file.read()


@retry(**retry_settings)
def create_or_update_assistant(client, assistant_data, tool_data, is_update=False):
    if is_update:
        return client.beta.assistants.update(
            assistant_id=assistant_data['assistant_id'],
            name=assistant_name,
            instructions=get_assistant_instructions(),
            model="gpt-4o",
            tools=[{"type": "file_search"}] + tool_data["tool_configs"]
        )
    else:
        return client.beta.assistants.create(
            instructions=get_assistant_instructions(),
            name=assistant_name,
            model="gpt-4o",
            tools=[{"type": "file_search"}] + tool_data["tool_configs"]
        )

# Create or load assistant
def create_assistant(client, tool_data, context):
    if os.path.exists(assistant_file_path):
        with open(assistant_file_path, 'r') as file:
            assistant_data = json.load(file)
            assistant_id = assistant_data['assistant_id']

            # Generate current hash sums
            current_tool_hashsum = core_functions.generate_hashsum('tools')
            current_resource_hashsum = core_functions.generate_hashsum('resources')
            current_assistant_hashsum = core_functions.generate_hashsum('assistant.py')
            current_instructions_hashsum = core_functions.generate_hashsum('assistant/instructions.txt')

            current_assistant_data = {
                'tools_sum': current_tool_hashsum,
                'resources_sum': current_resource_hashsum,
                'assistant_sum': current_assistant_hashsum,
                'instructions_sum': current_instructions_hashsum
            }

            if compare_assistant_data_hashes(current_assistant_data, assistant_data):
                print("Assistant is up-to-date. Loaded existing assistant ID.")
                return assistant_id
                
            else:
                print("Changes detected. Updating assistant...")
                try:
                    assistant = create_or_update_assistant(client, assistant_data, tool_data, is_update=True)
                    assistant_data = {
                        'assistant_id': assistant.id,
                        'tools_sum': current_tool_hashsum,
                        'resources_sum': current_resource_hashsum,
                        'assistant_sum': current_assistant_hashsum,
                        'instructions_sum': current_instructions_hashsum
                    }
                    save_assistant_data(assistant_data, assistant_file_path)
                    print(f"Assistant (ID: {assistant_id}) updated successfully.")
                except Exception as e:
                    print(f"Error updating assistant: {e}")
    else:
        try:
            assistant = create_or_update_assistant(client, {}, tool_data, is_update=False)
            print(f"Assistant ID: {assistant.id}")

            tool_hashsum = core_functions.generate_hashsum('tools')
            resource_hashsum = core_functions.generate_hashsum('resources')
            assistant_hashsum = core_functions.generate_hashsum('assistant.py')
            instructions_hashsum = core_functions.generate_hashsum('assistant/instructions.txt')

            assistant_data = {
                'assistant_id': assistant.id,
                'tools_sum': tool_hashsum,
                'resources_sum': resource_hashsum,
                'assistant_sum': assistant_hashsum,
                'instructions_sum': instructions_hashsum
            }

            save_assistant_data(assistant_data, assistant_file_path)
            print(f"Assistant has been created with ID: {assistant.id}")

            assistant_id = assistant.id
        except Exception as e:
            print(f"Error creating assistant: {e}")
            raise

    return assistant_id

# Save the assistant to a file
def save_assistant_data(assistant_data, file_path):
  try:
    with open(file_path, 'w') as file:
      json.dump(assistant_data, file)
  except Exception as e:
    print(f"Error saving assistant data: {e}")


# Checks if the Assistant JSON has all required fields
def is_valid_assistant_data(assistant_data):
  required_keys = [
      'assistant_id', 'tools_sum', 'resources_sum', 'assistant_sum',
      'instructions_sum'
  ]
  return all(key in assistant_data and assistant_data[key]
             for key in required_keys)


# Compares if all of the fields match with the current hashes
def compare_assistant_data_hashes(current_data, saved_data):
  if not is_valid_assistant_data(saved_data):
    return False

  return (current_data['tools_sum'] == saved_data['tools_sum']
          and current_data['resources_sum'] == saved_data['resources_sum']
          and current_data['assistant_sum'] == saved_data['assistant_sum'] and
          current_data['instructions_sum'] == saved_data['instructions_sum'])
