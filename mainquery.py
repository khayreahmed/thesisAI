import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import streamlit as st
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from langchain import OpenAI

doc_path = './data/'
index_file = 'index.json'

if 'response' not in st.session_state:
    st.session_state.response = ''

def send_click():
    query_engine = index.as_query_engine()
    st.session_state.response  = query_engine.query(st.session_state.prompt)

index = None

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@500&display=swap');
    body {
        color: #fff;
        background-color: #000;
        font-family: 'Josefin Sans', serif;
        margin: 0;
        padding: 0;
    }
    h1 {
        color: #fff;
    }
    a {
        color: #2196f3;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000;
        color: white;
        text-align: center;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("ThesisAI")

st.markdown(""" 
<footer>
    By Khayre Ali | 
    <a href="https://github.com/khayreahmed" style="background-color: white; border-radius: 50%; display: inline-block; width: 30px; height: 30px; text-align: center;">
    <img src="https://github.com/favicon.ico" width="24" style="padding-top: 3px;">
</a> |
    <a href="https://khayreali.substack.com/"><img src="https://substack.com/favicon.ico" width="24"></a> |
    <a href="https://www.linkedin.com/in/khayreali/"><img src="https://linkedin.com/favicon.ico" width="24"></a>
</footer>
""", unsafe_allow_html=True)


sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    doc_files = os.listdir(doc_path)
    for doc_file in doc_files:
        os.remove(doc_path + doc_file)

    bytes_data = uploaded_file.read()
    with open(f"{doc_path}{uploaded_file.name}", 'wb') as f: 
        f.write(bytes_data)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(uploaded_file.name)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    index.set_index_id(index_file)

elif os.path.exists(index_file):
    index = GPTVectorStoreIndex.load_from_disk(index_file)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    doc_filename = os.listdir(doc_path)[0]
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(doc_filename)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

if index is not None:
    with st.form(key='my_form'):
        st.text_input("Ask something: ", key='prompt')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button:
        query_engine = index.as_query_engine()
        st.session_state.response  = query_engine.query(st.session_state.prompt)

        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")
