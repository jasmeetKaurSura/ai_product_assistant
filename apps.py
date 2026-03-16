import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

DOC_FOLDER = "C:\Projects\Product-assistant\docs"

# -------------------------
# Load documents
# -------------------------
def load_documents():

    docs = []

    for file in os.listdir(DOC_FOLDER):

        path = os.path.join(DOC_FOLDER, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())

    return docs


# -------------------------
# Create Vector DB
# -------------------------
@st.cache_resource
def build_vector_store():

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# -------------------------
# Build QA System
# -------------------------
@st.cache_resource
def build_qa_chain():

    vectorstore = build_vector_store()

    llm = Ollama(
        model="tinyllama"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    return qa


# -------------------------
# Streamlit UI
# -------------------------
st.title("Private AI Document Assistant")

st.write(
    "Ask questions from internal documents."
)

qa_chain = build_qa_chain()

query = st.text_input(
    "Ask your question"
)

if query:

    with st.spinner("Thinking..."):

        answer = qa_chain.run(query)

    st.success(answer)