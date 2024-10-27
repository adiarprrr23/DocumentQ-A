import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Set up environment keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Streamlit app title
st.title("Gemma Model Document Report Generator")

# Language Model with Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# Adjusted prompt template to guide model in writing a deep report
report_prompt = ChatPromptTemplate.from_template(
    """
        Based strictly on the provided context, write a detailed report that spans at least 5 pages.
        The report should analyze the context thoroughly and be structured with an introduction, 
        body sections with key findings, examples, and insights, and a conclusion.
        
        <context>
        {context}
        <context>
        
        Please structure the report as follows:
        - Title: [Your Report Title]
        - Introduction: Summarize the main themes and objectives based on the context
        - Analysis: Discuss relevant points with examples and structured arguments
        - Insights: Provide specific insights, drawing from the context
        - Conclusion: Summarize key findings and implications
        
        Report:
    """
)

# Function to generate embeddings and set up vector storage
def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        st.session_state.loader = PyPDFDirectoryLoader('./papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embedding=st.session_state.embeddings)

# User input for generating report
input_prompt = st.text_input("Describe the type of report you need from the document:")

# Button to create vector store if it doesn’t already exist
if st.button("Create Vector Store"):
    vector_embeddings()
    st.write("Vector Store DB is Ready")

# Generate report based on input and context
if input_prompt:
    documents_chain = create_stuff_documents_chain(llm, report_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, documents_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input": input_prompt})
    st.write(response['answer'])

    # Show relevant document sections under the "Document Similarity Search" expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
