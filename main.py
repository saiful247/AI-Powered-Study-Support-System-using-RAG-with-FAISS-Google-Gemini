import os
import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=.6)

st.title("Study Support System")
st.sidebar.title("Upload PDF Files")

# Allow multiple PDF uploads
pdf_files = st.sidebar.file_uploader("Upload up to 2 PDFs", type=[
                                     'pdf'], accept_multiple_files=True)

process_pdf = st.sidebar.button("Process PDFs")

file_path = "models/qa_with_pdf"

main_placeholder = st.empty()

if process_pdf and pdf_files:
    doc_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join("uploaded_docs", pdf_file.name)
        os.makedirs("uploaded_docs", exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader(pdf_path)
        main_placeholder.text(f"Loading data from {pdf_file.name}...")

        data = loader.load()

        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=520, chunk_overlap=50)
        main_placeholder.text(f"Splitting data from {pdf_file.name}...")

        doc_chunks.extend(doc_splitter.split_documents(data))  # Append chunks

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

        retriever = vectorindex_gemini.as_retriever()
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        chain = RetrievalQA(
            combine_documents_chain=qa_chain, retriever=retriever)

        result = chain({"query": query})

        st.header("Answer")
        st.subheader(result["result"])
