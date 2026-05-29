"""End-to-end RAG: documents -> OpenAI embeddings -> CRDB -> query.

Requires: ``pip install llama-index-embeddings-openai llama-index-llms-openai``
and ``OPENAI_API_KEY`` in the environment.
"""

import os

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


def main() -> None:
    assert "OPENAI_API_KEY" in os.environ, "Set OPENAI_API_KEY first."

    embed = OpenAIEmbedding(model="text-embedding-3-small")  # 1536-d
    llm = OpenAI(model="gpt-4o-mini")

    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="rag_demo",
        embed_dim=1536,
        distance_metric="cosine",
        cspann_kwargs={"min_partition_size": 16, "max_partition_size": 128},
        sslmode="disable",
    )

    docs = [
        Document(text="CockroachDB v25.2 ships a native VECTOR type with the C-SPANN index."),
        Document(text="C-SPANN partitions the vector space so ANN search scales horizontally."),
        Document(text="LlamaIndex retrievers can be plugged into any vector store implementation."),
    ]

    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, embed_model=embed
    )

    engine = index.as_query_engine(llm=llm, similarity_top_k=3)
    answer = engine.query("How does CockroachDB index vectors?")
    print(answer)


if __name__ == "__main__":
    main()
