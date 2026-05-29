"""Async add + query with asyncpg under the hood."""

import asyncio

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


async def main() -> None:
    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="async_demo",
        embed_dim=4,
        sslmode="disable",
    )

    nodes = []
    for nid, text, vec in [
        ("a", "alpha", [1.0, 0.0, 0.0, 0.0]),
        ("b", "beta", [0.9, 0.1, 0.0, 0.0]),
    ]:
        n = TextNode(id_=nid, text=text)
        n.embedding = vec
        nodes.append(n)

    await store.async_add(nodes)
    res = await store.aquery(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=2)
    )
    for nid, sim in zip(res.ids, res.similarities, strict=False):
        print(f"{sim:.4f}  {nid}")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
