import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import wandb_tracing_enabled
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import asyncio


os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-proj"

def ollamaCalls () -> str:
    llm = Ollama(model="llama3")
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_template("""You are an expert in MLOps tooling. You have an opinion on how to best build an MLOps stack. 
    Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

    LSdocs = WebBaseLoader("https://docs.smith.langchain.com/").load()
    WBdocs = WebBaseLoader("https://wandb.github.io/weave/").load()
    embeddings = OllamaEmbeddings(model="llama3")

    text_splitter = RecursiveCharacterTextSplitter()
    LSdocuments = text_splitter.split_documents(LSdocs)
    WBdocuments = text_splitter.split_documents(WBdocs)
    vector = FAISS.from_documents(LSdocuments, embeddings)
    vector.aadd_documents(WBdocuments)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "What is better for LLMOPs? W&B Weave or LangSmith?"})

    return response["answer"]


print(ollamaCalls())

