"""Metadata filters: EQ, IN, range, IS_EMPTY, nested AND/OR."""

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore


def main() -> None:
    store = CockroachDBVectorStore.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        table_name="filter_demo",
        embed_dim=4,
        sslmode="disable",
        indexed_metadata_keys={("category", "text"), ("year", "int")},
    )

    rows = [
        ("a", "doc A", [1.0, 0.0, 0.0, 0.0], {"category": "ai", "year": 2024}),
        ("b", "doc B", [0.9, 0.1, 0.0, 0.0], {"category": "db", "year": 2025}),
        ("c", "doc C", [0.8, 0.2, 0.0, 0.0], {"category": "ai", "year": 2023}),
        ("d", "doc D", [0.7, 0.3, 0.0, 0.0], {"category": "ops", "year": 2024}),
    ]
    nodes = []
    for nid, text, vec, meta in rows:
        n = TextNode(id_=nid, text=text, metadata=meta)
        n.embedding = vec
        nodes.append(n)
    store.add(nodes)

    # category IN ("ai", "db") AND year >= 2024
    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
            MetadataFilter(key="category", value=["ai", "db"], operator=FilterOperator.IN),
            MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
        ],
    )
    res = store.query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            similarity_top_k=5,
            filters=filters,
        )
    )
    print("matched:", res.ids)  # expect ['a', 'b']


if __name__ == "__main__":
    main()
