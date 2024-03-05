# README

This code implements a Streamlit web application for a chatbot integrated with document retrieval and analysis functionalities. Users can interact with the chatbot by asking questions, and the bot responds based on the provided context and historical conversation.

## Features

- **Authentication**: Simple authentication is implemented to ensure secure access to the application.
- **Chat Interface**: Users can ask questions via a chat interface, and the chatbot provides responses.
- **Document Processing**: Users can upload PDF documents, which are then processed to extract text for further analysis.
- **Document Retrieval**: The application uses Elasticsearch for storing and retrieving document vectors, enabling efficient retrieval of relevant information.
- **Question Answering**: The chatbot leverages OpenAI's GPT-3.5 model for question answering, providing concise answers to user queries.

## Requirements

- Python 3.x
- Streamlit
- Streamlit Authenticator
- PyYAML
- Langchain Community Library
- Langchain Core Library
- Langchain OpenAI Library
- Elasticsearch
- Python-dotenv
- st-mui-dialog
- dotenv

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your/repository.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
   
   Create a `.env` file in the root directory and add the following variables:

    ```plaintext
    CHAT_HISTORY_INDEX=<Your Elasticsearch chat history index>
    ELASTICSEARCH_INDEX_NAME=<Your Elasticsearch index name>
    ELASTICSEARCH_URL=<Your Elasticsearch URL>
    ```

4. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Launch the application by running the Streamlit command.
2. Log in with your credentials.
3. Upload PDF documents for processing.
4. Interact with the chatbot by asking questions in the chat interface.
5. View responses from the chatbot and related references.

## Contributors

- Ever Cristian Coarite Vasquez.
