import os
import json
import bs4
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.load import dumps, loads

def load_config(config_path="config.json"):
    """ Load configuration from a JSON file """
    with open(config_path, "r") as config_file:
        return json.load(config_file)

def setup_environment(config):
    """ Set up API keys """
    os.environ['LANGCHAIN_API_KEY'] = config.get("LANGCHAIN_API_KEY", "")
    os.environ['OPENAI_API_KEY'] = config.get("OPENAI_API_KEY", "")

def load_documents(config):
    """ Load documents from a web source """
    loader = WebBaseLoader(web_paths=[config["DOCUMENT_SOURCE"]])
    return loader.load()

def process_documents(docs, config):
    """ Split documents into smaller chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["CHUNK_SIZE"], chunk_overlap=config["CHUNK_OVERLAP"])
    return text_splitter.split_documents(docs)

def create_vectorstore(splits, config):
    """ Create vector store using multiple embeddings """
    vectorstore_dict = {}
    for emb_type in config["EMBEDDINGS"]:
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model=emb_type))
        vectorstore_dict[emb_type] = vectorstore.as_retriever()
    return vectorstore_dict

def multi_representation_retrieval(retrievers):
    """ Perform retrieval using multiple embeddings """
    query_template = """You are an AI assistant. Generate five different versions of the question:\n\nOriginal question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(query_template)
    
    generate_queries = (
        prompt_perspectives | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
    )
    
    def get_unique_union(documents):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]
    
    retrieval_chains = {emb: generate_queries | retrievers[emb].map() | get_unique_union for emb in retrievers}
    return retrieval_chains

def reciprocal_rank_fusion(results, k=60):
    """ Apply Reciprocal Rank Fusion """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    return [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

def final_rag_chain(retrieval_chains):
    """ Define the final RAG chain using multiple representations """
    final_template = """Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(final_template)
    
    llm = ChatOpenAI(temperature=0)
    return (
        {"context": retrieval_chains, "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    config = load_config()
    setup_environment(config)
    docs = load_documents(config)
    splits = process_documents(docs, config)
    retrievers = create_vectorstore(splits, config)
    retrieval_chains = multi_representation_retrieval(retrievers)
    
    question = "What is multi-representation retrieval for RAG?"
    retrieved_docs = {emb: retrieval_chains[emb].invoke({"question": question}) for emb in retrieval_chains}
    for emb, docs in retrieved_docs.items():
        print(f"Retrieved {len(docs)} documents using {emb}")
    
    fusion_chain = retrieval_chains | reciprocal_rank_fusion
    fused_docs = fusion_chain.invoke({"question": question})
    print(f"RAG Fusion retrieved {len(fused_docs)} documents")
    
    final_chain = final_rag_chain(fusion_chain)
    response = final_chain.invoke({"question": question})
    print(response)

if __name__ == "__main__":
    main()
