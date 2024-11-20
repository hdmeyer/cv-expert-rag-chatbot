import streamlit as st
import os
import tempfile

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from cassandra.cluster import Cluster
from langchain_community.vectorstores import Cassandra
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    """
    StreamHandler is a callback handler that processes new tokens from a language model (LLM) and updates a container with the generated text.

    Attributes:
        container: The container object where the generated text will be displayed.
        text (str): The initial text to be displayed in the container.

    Methods:
        __init__(self, container, initial_text=""):
            Initializes the StreamHandler with a container and optional initial text.
        
        on_llm_new_token(self, token: str, **kwargs):
            Appends a new token to the text and updates the container with the current text followed by a cursor symbol.
    """
    #
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into DataStax Enterprise
def vectorize_text(uploaded_file, vector_store):
    """
    Vectorizes the text content of an uploaded PDF file and loads it into a vector store.
    Args:
        uploaded_file (UploadedFile): The uploaded PDF file to be processed.
        vector_store (VectorStore): The vector store where the vectorized text will be stored.
    Returns:
        None
    Notes:
        - The function writes the uploaded file to a temporary directory.
        - It then loads the PDF content and splits it into chunks using a text splitter.
        - The chunks are vectorized and added to the provided vector store.
        - Displays an info message with the number of pages loaded.
    """
    if uploaded_file is not None:
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap  = 100
        )

        # Vectorize the PDF and load it into the DataStax Enterprise Vector Store
        pages = text_splitter.split_documents(docs)
        vector_store.add_documents(pages)  
        st.info(f"{len(pages)} pages loaded.")

# Allow uploading files
def process_file(uploaded_file):
    """
    Process the uploaded file and return its content.
    Args:
        uploaded_file (UploadedFile): The uploaded file to be processed.
    Returns:
        str: The content of the uploaded file.
    """
    #
    if uploaded_file is not None:
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Load the PDF
        pages = []
        loader = PyPDFLoader(temp_filepath)
        for page in loader.lazy_load():
            pages.append(page)
        return " ".join([page.page_content for page in pages])
    
# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """
    You are a curriculum advisor that is specialized in the field of product management, technical writing, fund acquisition and food science. 
    You advise people on how to improve their CVs and what to include in them. Use the provided context as the basis for your 
    answers and do not make up new reasoning paths just mix-and-match what you are given. You should focus on the context given 
    for a position and the CV provided. You provide in-depth suggestions to change the uploaded CV to match the position description mentioned in the 
    question of the user. 

    CONTEXT: {context}

    CV: {cv}
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "QUESTION: {question}"),
        ]
    )


# Cache Mistral Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    # parameters for ollama see: https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.ollama.ChatOllama.html
    # num_ctx is the context window size
    return ChatOllama(model="mistral:latest", num_ctx=18192, base_url=st.secrets['OLLAMA_ENDPOINT'])

# Cache the DataStax Enterprise Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Datastax Enterprise v7 with Vector Support')
def load_vector_store():
    """
    Loads and returns a vector store from a Cassandra database.
    This function connects to a DataStax Enterprise (DSE) cluster using the endpoint
    specified in the secrets, and then connects to a Cassandra vector store using
    the provided keyspace and table name. The vector store is configured to use
    HuggingFace embeddings.
    Returns:
        vector_store (Cassandra): The connected vector store instance.
    """

    # Connect to DSE
    cluster = Cluster([st.secrets['DSE_ENDPOINT']])
    session = cluster.connect()

    # Connect to the Vector Store
    vector_store = Cassandra(
        session=session,
        embedding=HuggingFaceEmbeddings(),
        keyspace=st.secrets['DSE_KEYSPACE'],
        table_name=st.secrets['DSE_TABLE']
    )
    return vector_store


# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever(_vector_store):
    # Get the retriever for the Chat Model
    retriever = _vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever

prompt = load_prompt()
chat_model = load_chat_model()
vector_store = load_vector_store()
retriever = load_retriever(_vector_store=vector_store)

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'cv_content' not in st.session_state:
    st.session_state.cv_content = "default"
# Draw a title and some markdown
st.markdown("""# Your Enterprise Co-Pilot 🚀
Generative AI is considered to bring the next Industrial Revolution.  
Why? Studies show a **37% efficiency boost** in day to day work activities!

### Security and safety
This Chatbot is safe to work with sensitive data. Why?
- First of all it makes use of [Ollama, a local inference engine](https://ollama.com);
- On top of the inference engine, we're running [Mistral, a local and open Large Language Model (LLM)](https://mistral.ai/);
- Also the LLM does not contain any sensitive or enterprise data, as there is no way to secure it in a LLM;
- Instead, your sensitive data is stored securely within the firewall inside [DataStax Enterprise v7 Vector Database](https://www.datastax.com/blog/get-started-with-the-datastax-enterprise-7-0-developer-vector-search-preview);
- And lastly, the chains are built on [RAGStack](https://www.datastax.com/products/ragstack), an enterprise version of Langchain and LLamaIndex, supported by [DataStax](https://www.datastax.com/).""")
st.divider()

# Include the upload form for new data to be Vectorized
with st.sidebar:
    st.image("assets/chatbot_2.png", width=250)
    
    with st.form('upload_context'):
        uploaded_file = st.file_uploader('Upload a document for additional context. A position description maybe.', type=['pdf'])
        submitted = st.form_submit_button('Save to DataStax Enterprise')
        if submitted:
            vectorize_text(uploaded_file, vector_store)
    with st.form('upload_cv'):
        uploaded_cv = st.file_uploader('Upload your CV to get recommendations. ', type=['pdf'])
        submitted_cv = st.form_submit_button('Upload CV')
        if submitted_cv:
            st.session_state.cv_content = process_file(uploaded_cv)
            st.info(f"CV loaded.")
 
# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("What's up?"):
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()
    # Generate the answer by calling Mistral's Chat Model
    value=st.session_state.cv_content
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'cv': lambda x: value,
        'question': lambda x: x['question']
    })
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "ai", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)