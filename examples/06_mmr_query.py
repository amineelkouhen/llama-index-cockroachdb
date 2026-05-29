"""MMR mode for diverse top-k results."""

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


def main() -> None:
    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="mmr_demo",
        embed_dim=4,
        sslmode="disable",
    )

    seeds = [
        ("a", "alpha-1", [1.0, 0.0, 0.0, 0.0]),
        ("b", "alpha-2 near duplicate", [0.99, 0.01, 0.0, 0.0]),
        ("c", "alpha-3 near duplicate", [0.98, 0.02, 0.0, 0.0]),
        ("d", "different topic 1", [0.0, 1.0, 0.0, 0.0]),
        ("e", "different topic 2", [0.0, 0.0, 1.0, 0.0]),
    ]
    nodes = []
    for nid, txt, vec in seeds:
        n = TextNode(id_=nid, text=txt)
        n.embedding = vec
        nodes.append(n)
    store.add(nodes)

    # mmr_threshold=0 maximizes diversity, =1 maximizes relevance
    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.3,
    )
    res = store.query(q, mmr_prefetch_factor=4.0)
    for nid, sim in zip(res.ids, res.similarities, strict=False):
        print(f"{sim:.4f}  {nid}")


if __name__ == "__main__":
    main()
