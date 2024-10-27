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
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')

st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma-7b-it")

# prompt=ChatPromptTemplate.from_template(
#     """
#         Answer the questions based on the provided context only.Please provide the most accurate response based on the question 
#         <context>
#         {context}
#         <context>
#         Questions:{input}
#     """
# )

prompt = ChatPromptTemplate.from_template(
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

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        st.session_state.loader=PyPDFDirectoryLoader('./papers')
        
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,embedding=st.session_state.embeddings)

input_prompt=st.text_input("What you want to ask from the document?")

if st.button("Create Vector Store"):
    vector_embeddings()
    st.write("Vector Store DB is Ready")
import time

if input_prompt:
    documents_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,documents_chain)

    start=time.process_time()
    response=retriever_chain.invoke({"input":input_prompt})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
