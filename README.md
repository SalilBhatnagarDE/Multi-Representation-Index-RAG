# Multi-Representation Indexing and RAG

This repository explores advanced techniques in Retrieval Augmented Generation (RAG) using multi-representation indexing. It aims to improve the accuracy and relevance of generated responses by leveraging multiple embedding models and indexing strategies.

![Multi-Representation RAG](multi-rag.png)

## Overview

Traditional RAG systems often rely on a single embedding model to index and retrieve relevant documents. This approach can be limited, as different embedding models capture different semantic aspects of the text. This repository demonstrates how to combine multiple embedding models to create a more comprehensive and robust indexing system.

## Key Features

* **Multi-Representation Indexing:** Utilizes multiple embedding models (e.g., OpenAIEmbeddings, Sentence Transformers) to create diverse representations of the same documents.
* **Hybrid Retrieval:** Implements hybrid retrieval strategies, combining vector search with other techniques (e.g., keyword search, metadata filtering).
* **Advanced Reranking:** Integrates reranking algorithms to refine the retrieved documents and improve relevance.
* **Customizable Pipelines:** Provides flexible and modular components that can be easily customized and extended.
* **LangChain Integration:** Leverages LangChain for seamless integration with various language models and tools.
* **Configurable Parameters:** Uses a `config.json` file for easy configuration of embedding models, chunk sizes, and other parameters.
* **Clear and well documented code:** Python code with comments and docstrings.
* **Example Notebooks:** Jupyter notebooks demonstrating the usage of the library.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd multi-representation-rag
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API keys:**

    Create a `config.json` file with your API keys and other configuration parameters. Example:

    ```json
    {
      "OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
      "DOCUMENT_SOURCE": "[https://example.com/document.html](https://www.google.com/search?q=https://example.com/document.html)",
      "CHUNK_SIZE": 1000,
      "CHUNK_OVERLAP": 200,
      "EMBEDDING_MODELS": ["openai", "sentence_transformers"],
      "SENTENCE_TRANSFORMERS_MODEL": "all-mpnet-base-v2"
    }
    ```

4.  **Run the example notebooks:**

    Open the Jupyter notebooks in the `notebooks` directory and follow the instructions.

## Directory Structure
