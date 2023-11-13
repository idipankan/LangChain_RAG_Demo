import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader,TextLoader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

st.title('LangChain RAG Demo App')

#openai_api_key = st.sidebar.text_input('OpenAI API Key')

loader = TextLoader("dump.txt")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Store splits
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# LLM
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def getLLMResponse(query):
  return qa_chain({"query": query})

def generate_response(input_text):
  result = getLLMResponse(input_text)
  st.info(result['result'])

with st.form('my_form'):
  text = st.text_area('Enter text:', 'Ask away...')
  submitted = st.form_submit_button('Submit')
  # if not openai_api_key.startswith('sk-'):
  #   st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    generate_response(text)
