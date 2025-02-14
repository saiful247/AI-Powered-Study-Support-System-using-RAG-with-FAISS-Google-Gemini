import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=.6)

st.title("Study Support System")
st.sidebar.title("Select the pdf file")

pdf_file = st.sidebar.file_uploader("Upload a pdf file", type=['pdf'])

process_pdf = st.sidebar.button("Process PDF")

file_path = "models/qa_with_pdf"

main_placeholder = st.empty()

if process_pdf and pdf_file is not None:

    pdf_path = os.path.join("uploaded_docs", pdf_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader(pdf_path)

    main_placeholder.text("Loading data...")

    data = loader.load()

    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=520, chunk_overlap=50)

    main_placeholder.text("Splitting data...")

    doc_chunks = doc_splitter.split_documents(data)

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorindex_gemini = FAISS.from_documents(doc_chunks, embedding)

    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    vectorindex_gemini.save_local(file_path)

query = main_placeholder.text_input("Enter your question here")

if query:
    if os.path.exists(file_path):
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorindex_gemini = FAISS.load_local(
            file_path, embedding, allow_dangerous_deserialization=True)

        from langchain.chains import RetrievalQA

        chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorindex_gemini.as_retriever(), chain_type="stuff")

        result = chain({"query": query})

        st.header("Answer")
        st.subheader(result["result"])
