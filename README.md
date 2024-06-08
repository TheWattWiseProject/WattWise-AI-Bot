# WattWise AI Bot

## Introduction

**WattWise** is your energy-awareness companion, blending AI technology and personalized advice to help you visualize your daily energy usage across food, transport, and the home. By combining available data and your input on daily habits, Wattwise provides real-time insights and advice to make small changes with a big collective impact.

## Core Values

- **Sustainable Living**: Designing solutions that enable people to adopt more sustainable habits.
- **Knowledge Empowerment**: Promoting the growth of knowledge through open-source design.
- **Energy Awareness**: Helping people understand energy sources and their environmental impact.

## About Us
We are two masters students studying Design for Emergent Futures at IAAC and ELISAVA. We believe that together we can start to build towards the changes that need to happen in the world. As two design students with a passion for sustainable design, we have worked hard to better understand what the problem of energy is and how we can create a sense of awareness for people. 

**Carlotta**: An industrial design engineer with a passion for human-centered, sustainable design.  
**Oliver**: A product designer focused on planet-centered design across physical and digital realms.

## Getting Started

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TheWattWiseProject/WattWise-AI-Bot.git
   cd WattWise-AI-Bot/WattWise\ General\ Chatbot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes:

   ```plaintext
   flask
   openai
   langchain
   tenacity
   pytelegrambotapi
   nltk
   deep-translator
   SoundFile
   requests
   langdetect
   faiss-cpu
   packaging
   langsmith
   ```

   - `flask`: For creating the web application.
   - `openai`: To interact with OpenAI's API.
   - `langchain`: For document processing and embeddings.
   - `tenacity`: To handle retries for API calls.
   - `pytelegrambotapi`: For integrating with Telegram.
   - `nltk`: For natural language processing tasks.
   - `deep-translator`: For translating text.
   - `SoundFile`: For audio processing.
   - `requests`: For making HTTP requests.
   - `langdetect`: For language detection.
   - `faiss-cpu`: For efficient similarity search and clustering.
   - `packaging`: To handle version comparisons.
   - `langsmith`: For tracing and managing LangChain operations.

### Setup Environment Variables

For a Replit environment or local setup, replace `os.environ.get` with your actual API keys or use Replit's secrets management:

1. Open the 'Secrets' tab in Replit (represented by a lock icon).
2. Add the following secrets:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `TELEGRAM_TOKEN`: Your Telegram bot token.
   - `LANGCHAIN_API_KEY`: Your LangChain API key.

Alternatively, you can set these variables in your local environment.

### Creating a Telegram Bot

1. Open Telegram and search for [BotFather](https://telegram.me/BotFather).
2. Start a chat with BotFather and send the command `/newbot`.
3. Follow the prompts to set up your bot, including choosing a name and username.
4. Once created, BotFather will provide you with an API token. This token is the `TELEGRAM_TOKEN` variable you need to add to your environment variables.

### Running the Application

1. Start the Flask application:
   ```bash
   python main.py
   ```

2. The application is accessed through the Telegram app using your bot. Ensure you have set up your bot with BotFather and added the `TELEGRAM_TOKEN`.

## Project Structure

- **assistant.py**: Handles creating and managing the AI assistant, including instructions and hash sum comparisons for updates. The model used is GPT-4o called through OpenAI API. You can switch to another model, but it requires code restructuring.
- **core_functions.py**: Stores most functions that are called elsewhere in the code. If there's an error with `core_function.XYZ`, check this file to resolve the issue.
- **main.py**: Sets up all other files to ensure the bot is live via `telegramV2.py`.
- **telegramV2.py**: Manages waiting, formatting, and responding to messages. It differentiates between audio, image, and text inputs, and works with multiple languages. This file contains the structure for messages sent to the OpenAI API and retrieves relevant data for inputs. If you want to change the number of data points retrieved, adjust the `kwargs`.
- **requirements.txt**: Lists Python dependencies.
- **resources**: Contains data files used by the chatbot.
- **integrations**: Integration modules for external services.
- **docs/static**: Location where the vector store is saved.

### Instructions for the Assistant

The `assistant/instructions.txt` file contains the instructions for the assistant, detailing how it should interact and respond to user inputs.

### Vector Store

The data from the `resources` folder is used to create the vector store, which is stored locally in the `docs/static` directory. This allows efficient retrieval of document embeddings for generating responses.

## Contributing

We welcome contributions! Feel free to reach out to us through our [social media](https://www.instagram.com/thewattwiseproject/?igsh=MXg0MjV4NXpsc3R0eQ%3D%3D&utm_source=qr) or check out our website and get in touch that way! [WattWise Website](https://thewattwiseproject.github.io/WattWise/index.html)
