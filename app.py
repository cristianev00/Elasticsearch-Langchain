import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from st_mui_dialog import st_mui_dialog
import os
from dotenv import load_dotenv

#
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# elasticsearch
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain.memory import ElasticsearchChatMessageHistory

# prompt template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter

# cache
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain_community.callbacks import get_openai_callback

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

load_dotenv()

CHAT_HISTORY_INDEX = os.getenv("CHAT_HISTORY_INDEX")
ELASTICSEARCH_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX_NAME")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")

# Configuration of simple authentication
with open("config/auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)


def get_session_history(session_id: str):
    chat_history = ElasticsearchChatMessageHistory(
        es_url=ELASTICSEARCH_URL,
        session_id=session_id,
        index=CHAT_HISTORY_INDEX,
    )
    response = chat_history
    return response


def format_message(message: str):
    formatted_text = ""
    # Iterate through the context list
    for document in message:
        # Extract page content and metadata
        page_content = document.page_content.strip()
        metadata = document.metadata

        # Add page number as title
        formatted_text += f"Pagina No.{metadata['page']+ 1}\n\n --->    "

        # Add content as paragraph
        formatted_text += f"{page_content}\n\n"

        # Add metadata details as subtitles
        formatted_text += "Metadata:\n"
        for key, value in metadata.items():
            formatted_text += f"- {key}: {value}\n"
            formatted_text += "\n"

    return formatted_text


def save_pdfs(pdf_docs):
    saved_paths = []
    docs_folder = "docs"
    os.makedirs(docs_folder, exist_ok=True)
    existing_files = set(os.listdir(docs_folder))
    for idx, pdf_file in enumerate(pdf_docs):
        filename = pdf_file.name
        if filename in existing_files:
            st.warning(f"El archivo '{filename}' ya existe. Continuando...")
            continue
        pdf_path = os.path.join(docs_folder, filename)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        saved_paths.append(pdf_path)
        existing_files.add(filename)
    return saved_paths


# Load documents from directory
def get_pdf_loaders(pdf_docs, expediente):
    for pdf in pdf_docs:
        loader = PyPDFLoader(pdf)
        documents = loader.load()
        for doc in documents:
            # Add metadata to the document
            doc.metadata["expediente"] = expediente
    return documents


def get_text_chunks_elasticSearch(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks


def get_vectorstore_elasticSearch(text_chunks):
    with get_openai_callback() as cb:

        embeddings = OpenAIEmbeddings()
        vectorstore = ElasticsearchStore.from_documents(
            text_chunks,
            embeddings,
            es_url=ELASTICSEARCH_URL,
            index_name=ELASTICSEARCH_INDEX_NAME,
        )
    print(cb)
    vectorstore.client.indices.refresh(index=ELASTICSEARCH_INDEX_NAME)

    return vectorstore


def chatbot(prompt, metadata):

    embeddings = OpenAIEmbeddings()
    db = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name=ELASTICSEARCH_INDEX_NAME,
        embedding=embeddings,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
    )
    db.client.indices.refresh(index=ELASTICSEARCH_INDEX_NAME)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    if len(metadata) > 0:
        # Split the metadata string into an array separated by commas,
        # strip each element, and add "add this" to each element
        metadata_array = [data.strip() for data in metadata.split(",")]
        metadata_array.append("Ley N° 254 CODIGO PROCESAL CONSTITUCIONAL")

        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {
                        "terms": {"metadata.expediente.keyword": metadata_array}
                    },
                }
            ),
            llm=llm,
        )
    else:
        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            llm=llm,
        )

    session_id = str(st.session_state["name"])
    chat_history = get_session_history(session_id)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": contextualized_question | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).assign(answer=rag_chain_from_docs)

    with get_openai_callback() as cb:
        results = rag_chain_with_source.invoke(
            {"question": prompt, "chat_history": chat_history.messages}
        )
        print(cb)

    chat_history.add_user_message(prompt)
    chat_history.add_ai_message(results["answer"])

    return results


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():

    # First we will get back the chat history
    session = str(st.session_state["name"])
    history = get_session_history(session)

    # The version of the prototype
    st.title("Prototipo v0.3")
    st.session_state.messages = []
    # Assuming your list is named `message_list`

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.write("Hola 👋")

    # we have a chat list with different class names so we will order them
    for message in history.messages:
        if type(message).__name__ == "HumanMessage":
            st.session_state.messages.append(
                {"role": "user", "content": message.content}
            )
        elif type(message).__name__ == "AIMessage":
            st.session_state.messages.append(
                {"role": "assistant", "content": message.content}
            )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        if st.button("Borrar Historial"):
            history.clear()
            st.session_state.messages = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        authenticator.logout("Salir")
        st.write(f'Bienvenido *{st.session_state["name"]}*')
        st.subheader("Tus documentos")
        pdf_docs = st.file_uploader(
            "Carga tus archivos PDF's y luego presiona 'Procesar'",
            accept_multiple_files=True,
            type="pdf",
        )

        expediente = st.text_input("Ingresa el No. Expediente")

        if pdf_docs is not None and st.button("Procesar") and len(expediente) > 0:
            with st.spinner("Procesando..."):
                # Save PDFs and get their paths
                pdf_paths = save_pdfs(pdf_docs)
                st.write("PDFs guardado en las siguientes rutas:")
                for path in pdf_paths:
                    st.write(path)
                # empty the file uploader
                pdf_docs = None
                # get pdf text
                documents = get_pdf_loaders(pdf_paths, expediente=expediente)
                # get the text chunks
                text_chunks = get_text_chunks_elasticSearch(documents)
                # create vector store
                vectorstore = get_vectorstore_elasticSearch(text_chunks)
        else:
            st.write("Introduce No. Expediente")

    # React to user input

    if prompt := st.chat_input("Pregunta algo..."):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = chatbot(prompt, expediente)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": str(response)}
        )

        # Format context text message
        formatted_text = format_message(answer["context"])
        dialog = st_mui_dialog(
            title="Referencias",
            content=formatted_text,
            button_txt="Referencias",
            abortlabel="Cancelar",
            agreelabel="ok",
            width_dialog="xl",
        )


if __name__ == "__main__":
    authenticator.login()
    if st.session_state["authentication_status"]:
        main()
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")
