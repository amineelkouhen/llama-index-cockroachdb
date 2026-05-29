"""Use CockroachDBRetriever standalone with C-SPANN beam-size tuning."""

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.retrievers.cockroachdb import CockroachDBRetriever
from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


def main() -> None:
    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="rag_demo",
        embed_dim=1536,
        sslmode="disable",
    )

    # Higher beam size = better recall at slightly more latency.
    # Set per-retriever instead of globally to keep cheap queries fast.
    retriever = CockroachDBRetriever(
        vector_store=store,
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
        similarity_top_k=5,
        vector_search_beam_size=128,
    )

    for node_with_score in retriever.retrieve("how does C-SPANN compare to HNSW?"):
        print(f"{node_with_score.score:.4f}  {node_with_score.node.get_content()[:80]}...")


if __name__ == "__main__":
    main()
