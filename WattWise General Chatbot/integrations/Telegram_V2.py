"""
Setting Up Telegram Integration

To integrate a Telegram bot into your application, you'll need to first create the bot on Telegram and then configure your environment (like Replit) to use it. Here's a step-by-step guide:

1. Create a Telegram Bot:
   - Start by searching for 'BotFather' on Telegram. This is an official bot that Telegram uses to create and manage other bots.
   - Send the '/newbot' command to BotFather. It will guide you through creating your bot. You'll need to choose a name and a username for your bot.
   - Upon successful creation, BotFather will provide you with an API token. This token is essential for your bot's connection to the Telegram API.

2. Add the API Key to Replit:
   - Go to your Replit project where you intend to use the Telegram bot.
   - Open the 'Secrets' tab (usually represented by a lock icon).
   - Create a new secret with the key as `TELEGRAM_TOKEN` and the value as the API token provided by BotFather.
"""

import os
import logging
import telebot
import core_functions
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from pathlib import Path
from pprint import pprint
from langchain_openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langsmith import client, traceable

import signal
import tempfile
import sys
from threading import Thread, Event, current_thread

from flask import Flask, request, jsonify
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = "WattWise-AI-Chatbot"

# Initialize OpenAI client
client = os.environ.get('OPENAI_API_KEY')

## for chunk splitting incoming message

import nltk
from nltk.tokenize import word_tokenize
# Download the NLTK sentence tokenizer model
nltk.download('punkt')

# translator
from deep_translator import GoogleTranslator

#for image recognition
import base64
import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

#for audio transcription
import soundfile as sf
import transcribe

# path for vector store
vector_store_path = 'docs/static'
# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the global variable
language_status = None
messageEN = None
telegram_chat_id = None

# Define event for stopping threads gracefully
stop_event = Event()


# Defines if a DB mapping is required
def requires_mapping():
    return True


@traceable
def setup_routes(app, client, tool_data, assistant_id):
    # Ensure this is declared as global
    global language_status
    global messageEN
    global telegram_chat_id

    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    if not TELEGRAM_TOKEN:
        raise ValueError("No Telegram token found in environment variables")
    global bot
    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    # Ensure no webhooks are set
    #bot.remove_webhook()

    greetings = [
        'hello', 'hello!', 'hi', 'Hi!'
        'hi there!', 'hey', 'greetings', 'hola', 'bonjour'
    ]

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        telegram_chat_id = message.chat.id
        user = message.from_user
        logging.info("Starting a new conversation...")

        chat_mapping = core_functions.get_chat_mapping("telegram",
                                                       telegram_chat_id,
                                                       assistant_id)

        # Check if this chat ID already has a thread ID
        if not chat_mapping:
            thread = client.beta.threads.create()

            # Save the mapping
            core_functions.update_chat_mapping("telegram", telegram_chat_id,
                                               assistant_id, thread.id)

            logging.info(f"New thread created with ID: {thread.id}")

        welcome_message = f"Hi {user.first_name}! How can I help you today?"
        bot.reply_to(message, welcome_message)

    @bot.message_handler(
        func=lambda message: message.text.lower() in greetings)
    def greet_user(message):
        telegram_chat_id = message.chat.id
        user = message.from_user
        welcome_message = f"Hi {user.first_name}! How can I help you today?"
        bot.reply_to(message, welcome_message)

    @bot.message_handler(content_types=['voice'])
    def handle_audio(message):
        telegram_chat_id = message.chat.id
        # Get the OpenAI API key from the environment variable
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        # Initialize the OpenAI client
        openai_transcription_client = OpenAI(api_key=openai_api_key)

        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.ogg') as temp_ogg:
            temp_ogg_path = temp_ogg.name

        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.wav') as temp_wav:
            temp_wav_path = temp_wav.name

        try:
            file_info = bot.get_file(message.voice.file_id)
            logging.info(f"Retrieved file_info: {file_info}")
            downloaded_file = bot.download_file(file_info.file_path)

            with open(temp_ogg_path, 'wb') as f:
                f.write(downloaded_file)

            temp_wav_path = temp_ogg_path.replace('.ogg', '.wav')
            convert_ogg_to_wav(temp_ogg_path, temp_wav_path)

            result_transcription = transcribe.audio_transcribe(temp_wav_path)
            user_input = str(result_transcription)

            # Cleanup temporary files
            os.remove(temp_ogg_path)
            os.remove(temp_wav_path)

            language = core_functions.detect_language(user_input)
            logging.info(f"Detected Language: {language}")
            global language_status
            language_status = language

            messageEN = GoogleTranslator(
                source='auto', target='en').translate(text=user_input)
            logging.info(f"English Message: {messageEN}")

            chunks = core_functions.chunk_input_message2(messageEN)
            logging.info(f"Chunks: {chunks}")

            echo_all(telegram_chat_id, messageEN, chunks)

        except telebot.apihelper.ApiTelegramException as e:
            logging.error(f"Telegram API error: {e}")
            bot.send_message(
                telegram_chat_id,
                "Failed to process audio. The file might be temporarily unavailable. Please try again later."
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download audio: {e}")
            bot.send_message(
                telegram_chat_id,
                "Failed to process audio. Please try again later.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            bot.send_message(
                telegram_chat_id,
                "An unexpected error occurred. Please try again later.")

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        telegram_chat_id = message.chat.id
        # Get the file ID and download the image
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}"

        #get additional text input with photo and add
        Usermessage = message.caption
        if Usermessage:
            user_message = GoogleTranslator(
                source='auto', target='en').translate(text=Usermessage)
        else:
            user_message = " "

        try:
            image_data = base64.b64encode(
                download_file(image_url)).decode("utf-8")
            print("Image_Data:", image_data)

            #process to text
            model = ChatOpenAI(model="gpt-4o")

            message = HumanMessage(content=[
                {
                    "type":
                    "text",
                    "text":
                    "describe the contents of the this image. If it is food please identify the ingredients and amounts based on the image and give a list of quantaties in kg+name, use estimations/assumptions if needed. Include a quick introduction saying today I used ... Please keep the answer short, factual and based purely on the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ], )

            response = model.invoke([message])
            print("photo Description is:", response.content)

            # Combine user message and AI-generated description
            user_input = f"{user_message}\n\n{response.content}"

            # Detect language and translate if necessary
            language = core_functions.detect_language(user_input)
            print("Detected Language:", language)
            global language_status
            language_status = language

            messageEN = GoogleTranslator(
                source='auto', target='en').translate(text=user_input)
            print("English Message:", messageEN)

            # Tokenize and chunk the input message
            chunks = []
            chunks = core_functions.chunk_input_message2(messageEN)
            print("chunks are:", chunks)

            # Proceed with the common processing logic
            echo_all(telegram_chat_id, messageEN, chunks)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download image: {e}")
            bot.send_message(
                telegram_chat_id,
                "Failed to process image. Please try again later.")

    @bot.message_handler(func=lambda message: True)
    def handle_text(message):
        telegram_chat_id = message.chat.id
        user_input = message.text
        print("Message is:", user_input)

        # Detect language and translate if necessary
        language = core_functions.detect_language(user_input)
        print("Detected Language:", language)
        global language_status
        language_status = language

        messageEN = GoogleTranslator(source='auto',
                                     target='en').translate(text=user_input)
        print("English Message:", messageEN)

        # Tokenize and chunk the input message
        chunks = []
        chunks = core_functions.chunk_input_message(messageEN)
        print("chunks are:", chunks)

        # Proceed with the common processing logic
        echo_all(telegram_chat_id, messageEN, chunks)

    @bot.message_handler(func=lambda message: True)
    def echo_all(telegram_chat_id, messageEN, chunk):

        # Tokenize and chunk the input message
        chunks = []
        chunks = chunk

        # will give the most relevant chunks of data to the LLM to let it answer your question
        relevant_info = core_functions.load_vector_store(
            vector_store_path, embeddings)
        all_retrieved_docs = []

        # run through chunks input message chunks and grabs 4 kwars
        for chunk in chunks:
            retriever = relevant_info.as_retriever(search_type="similarity",
                                                   search_kwargs={"k": 2})
            retrieved_docs = retriever.invoke(chunk)
            all_retrieved_docs.extend(retrieved_docs)

        # Remove duplicates
        all_retrieved_docs = list(
            {doc.page_content: doc
             for doc in all_retrieved_docs}.values())

        # Check the number of retrieved documents
        print(f"Number of retrieved documents: {len(all_retrieved_docs)}")

        # Prepare the context from retrieved documents
        context_docs = "\n".join(
            [doc.page_content for doc in all_retrieved_docs])
        print("Context to send:", context_docs)

        # Threading
        db_entry = core_functions.get_chat_mapping("telegram",
                                                   telegram_chat_id,
                                                   assistant_id)
        print(f"DB entry: {db_entry}")

        thread_id = core_functions.get_value_from_mapping(
            db_entry, "thread_id")
        print(f"Thread ID: {thread_id}")

        if not thread_id:

            thread = client.beta.threads.create()

            # Save the mapping
            core_functions.update_chat_mapping("telegram", telegram_chat_id,
                                               assistant_id, thread.id)

            thread_id = thread.id

            logging.info(f"XXXXXX: {thread_id}")

        if not thread_id:
            logging.error("Error: Missing OpenAI thread_id")
            return

        prompt_with_context = f"User Input: {messageEN}\n\nContext from Documents: {context_docs}\n\nIMPORTANT: Provide a response under 100 words and format this all for a text message, avoid backend language."

        logging.info(
            f"Message for AI: {prompt_with_context} for OpenAI thread ID: {thread_id}"
        )

        client.beta.threads.messages.create(thread_id=thread_id,
                                            role="user",
                                            content=prompt_with_context)
        print("Message sent to OpenAI")

        #test truncation
        # Define the truncation strategy
        truncation_strategy = {"type": "last_messages", "last_messages": 10}

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            truncation_strategy=truncation_strategy)
        print("Run created")

        # This processes any possible action requests
        core_functions.process_tool_calls(client, thread_id, run.id, tool_data)
        print("Tool calls processed"
              )  # this is where issue is due to high data storaf

        #list messages received from the ai an grab the last
        message = client.beta.threads.messages.list(thread_id=thread_id,
                                                    limit=10)
        print("Message grabbed")

        AIresponse = message.data[0].content[0].text.value
        AIresponse = str(AIresponse)

        ResponseL = GoogleTranslator(
            source='en', target=language_status).translate(text=AIresponse)
        print("AI Answer:", ResponseL)

        response = ResponseL

        # Use the original Telegram chat ID here, not the OpenAI thread ID
        bot.send_message(telegram_chat_id, response, parse_mode='Markdown')
    
    # Remove any existing webhook before starting polling
    bot.remove_webhook()

    # Start polling in a separate thread
    from threading import Thread
    def start_polling():
        bot.infinity_polling(none_stop=True)
    Thread(target=start_polling).start()

def convert_ogg_to_wav(ogg_path, wav_path):
    """Convert an OGG file to WAV format."""
    try:
        data, samplerate = sf.read(ogg_path)
        sf.write(wav_path, data, samplerate)
        logging.info(f"Successfully converted {ogg_path} to {wav_path}")
    except Exception as e:
        logging.error(f"Error converting {ogg_path} to {wav_path}: {e}")

# Retry settings
retry_settings = {
    'stop': stop_after_attempt(3),
    'wait': wait_exponential(multiplier=1, min=4, max=10),
    'retry': retry_if_exception_type(requests.exceptions.RequestException)
}

@retry(**retry_settings)
def download_audio(bot, file_path):
    file_info = bot.get_file(file_path)
    return bot.download_file(file_info.file_path)

@retry(**retry_settings)
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content