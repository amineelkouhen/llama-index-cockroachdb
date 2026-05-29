"""Basic usage: create a store, insert TextNodes, query by embedding."""

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


def main() -> None:
    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="basic_demo",
        embed_dim=4,
        distance_metric="cosine",
        cspann_kwargs={"min_partition_size": 8, "max_partition_size": 32},
        sslmode="disable",
    )

    nodes = []
    for nid, text, vec in [
        ("a", "cockroachdb supports native vector search", [1.0, 0.0, 0.1, 0.0]),
        ("b", "postgres uses pgvector for ANN", [0.9, 0.1, 0.0, 0.1]),
        ("c", "the moon orbits the earth", [0.0, 1.0, 0.0, 1.0]),
    ]:
        node = TextNode(id_=nid, text=text)
        node.embedding = vec
        nodes.append(node)

    store.add(nodes)

    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.1, 0.0],
        similarity_top_k=2,
    )
    result = store.query(query)
    for node, score in zip(result.nodes, result.similarities, strict=False):
        print(f"{score:.4f}  {node.id_}  {node.get_content()}")


if __name__ == "__main__":
    main()
