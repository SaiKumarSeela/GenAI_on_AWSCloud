import json 
import os 
import sys 
import boto3
import streamlit as st 
from langchain_community.llms import Bedrock
import os
import numpy as np 
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
load_dotenv()
# from langchain.embeddings import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings
# from langchain.llms.bedrock import Bedrock
# from langchain_aws.embeddings import BedrockEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
# Bedrock clients
# print("AWS Access Key ID:", os.getenv("AWS_ACCESS_KEY_ID"))
# print("AWS Secret Access Key:", os.getenv("AWS_SECRET_ACCESS_KEY"))


bedrock = boto3.client(service_name="bedrock-runtime",region_name='ap-south-1' )
# bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client= bedrock)
# embeddings = GooglePalmEmbeddings()
google_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # in our testing character split works
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs

# Vector embeddings and vector store


def get_vector_store(docs):
    vectorestore_faiss = FAISS.from_documents(
        docs,
        GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    )
    vectorestore_faiss.save_local("faiss_index")

def get_llama3_llm():
    # create the anthropic 

    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock,
                  model_kwargs = {
                      "max_gen_len":512})
    return llm


prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast summarize with 250 words
with detailed explanation. If you don't know the answer , just say that you don't know, don't try to make up an answer.
{context}
</context

Question: {question}

Assistant: 

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context","question"]
)

def get_response_llm(llm, vectorestore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vectorestore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer = qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question from the PDF files")
    
    with st.sidebar:
        st.title("Update :")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", google_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()

            st.write(get_response_llm(llm,faiss_index, user_question))
            st.success("Done")



if __name__=="__main__":
    main()

