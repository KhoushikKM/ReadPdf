import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile

st.title("AI PDF Analyst")

groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Extract & Chunk
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split() # Simple split for learning
    
    # 3. Use FREE Embeddings (Runs on your CPU)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Create Local Vector Store
    vectorstore = Chroma.from_documents(pages, embeddings)
    
    # 5. Connect to Free LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    
    # Q&A Feature
    user_query = st.text_input("Ask anything about the PDF:")
    if user_query:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        response = qa_chain.invoke(user_query)
        st.write(response["result"])
