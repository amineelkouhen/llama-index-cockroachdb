# llama-index-cockroachdb

LlamaIndex integration for [CockroachDB](https://www.cockroachlabs.com/)'s native `VECTOR` column type and the [C-SPANN](https://www.cockroachlabs.com/docs/stable/vector.html) approximate nearest neighbor index. Ships three classes:

- `CockroachDBVectorStore` (`llama_index.vector_stores.cockroachdb`): drop-in `BasePydanticVectorStore` backed by CRDB.
- `CockroachDBReader` (`llama_index.readers.cockroachdb`): load existing CRDB rows as LlamaIndex `Document` objects.
- `CockroachDBRetriever` (`llama_index.retrievers.cockroachdb`): standalone retriever with C-SPANN beam-size tuning.

Requires **CockroachDB v25.2+** and Python 3.10+.

## Why not just use the pgvector store?

`llama-index-vector-stores-postgres` depends on the [pgvector](https://github.com/pgvector/pgvector) extension. CockroachDB ships its own native `VECTOR(n)` type and the C-SPANN distributed ANN index, so the pgvector store cannot be used as-is against a CRDB cluster: the DDL (`CREATE EXTENSION vector`, `USING hnsw`) and the per-session tuning vars (`hnsw.ef_search`, `ivfflat.probes`) don't exist there. This package targets CRDB's actual API surface.

| Concern | pgvector store | this package |
| --- | --- | --- |
| Column type | `pgvector.sqlalchemy.Vector` | native `VECTOR(dim)` |
| Index DDL | `CREATE INDEX ... USING hnsw (...) WITH (m, ef_construction)` | `CREATE VECTOR INDEX ... WITH (min_partition_size, max_partition_size)` |
| Search-time knob | `SET hnsw.ef_search` | `SET vector_search_beam_size` |
| Setup prereq | `CREATE EXTENSION vector` | `SET CLUSTER SETTING feature.vector_index.enabled = true` |
| Dialect | `postgresql+psycopg2` / `+asyncpg` | `cockroachdb+psycopg2` / `cockroachdb+asyncpg` (retry-aware) |
| `HALFVEC` | yes | not yet |
| Hybrid / sparse (`tsvector`) | yes | not yet (CRDB has no `tsvector` equivalent) |

## Install

```bash
pip install llama-index-cockroachdb
```

You also need a CRDB v25.2+ cluster with the vector feature enabled once at the cluster level:

```sql
SET CLUSTER SETTING feature.vector_index.enabled = true;
```

The store will attempt to set this on first initialization if `enable_feature_setting=True` (the default) and the connected user has permission.

## Quick start

```python
from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document

store = CockroachDBVectorStore.from_params(
    host="localhost",
    port=26257,
    user="root",
    password="",
    database="defaultdb",
    table_name="my_index",
    embed_dim=1536,
    distance_metric="cosine",                # or "l2", "inner_product"
    cspann_kwargs={
        "min_partition_size": 16,
        "max_partition_size": 128,
    },
    sslmode="disable",                       # local dev only
)

index = VectorStoreIndex.from_documents(
    [Document(text="...")],
    storage_context=StorageContext.from_defaults(vector_store=store),
)
print(index.as_query_engine().query("..."))
```

See [`examples/`](./examples/) for seven runnable scripts covering async, MMR, metadata filters, the standalone retriever, and the reader.

## Tuning C-SPANN

Two levers, both optional:

1. **Build-time partitioning**: pass `cspann_kwargs={"min_partition_size": ..., "max_partition_size": ...}` to `from_params()`. These map directly to the `WITH` clause on `CREATE VECTOR INDEX`.
2. **Query-time beam size**: pass `vector_search_beam_size=N` to the store constructor (applies to every query) or as a kwarg on individual `query()` calls. Higher = better recall, slightly more latency. Issued as `SET LOCAL vector_search_beam_size` per session.

```python
store.query(query, vector_search_beam_size=128)
```

## Reader

```python
from llama_index.readers.cockroachdb import CockroachDBReader

reader = CockroachDBReader.from_params(
    host="localhost", port=26257, database="defaultdb", user="root", sslmode="disable",
)
docs = reader.load_data(
    table="articles",
    text_column="body",
    metadata_columns=["id", "author", "tag"],
    id_column="id",
)
# or pass a query=... with :named params
```

## Retriever

```python
from llama_index.retrievers.cockroachdb import CockroachDBRetriever
from llama_index.embeddings.openai import OpenAIEmbedding

retriever = CockroachDBRetriever(
    vector_store=store,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    similarity_top_k=5,
    vector_search_beam_size=128,
)
nodes_with_scores = retriever.retrieve("What is C-SPANN?")
```

## Supported query modes

| Mode | Supported | Notes |
| --- | --- | --- |
| `DEFAULT` | yes | Vector ANN through C-SPANN |
| `MMR` | yes | Client-side rerank, configurable `mmr_threshold`, `mmr_prefetch_factor`, `mmr_prefetch_k` |
| `HYBRID` / `SPARSE` / `TEXT_SEARCH` | no | CRDB has no `tsvector` yet; raises `NotImplementedError` |

## Supported metadata filter operators

`EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`, `CONTAINS`, `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`, `IS_EMPTY`. Filters are AND/OR/NOT nestable via `MetadataFilters`.

For frequently filtered keys, declare `indexed_metadata_keys={("category", "text"), ("year", "int")}` on the store to get JSONB-extracted BTREE indices.

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run pytest         # spins up a CRDB testcontainer per session
uv run ruff check .
```

Tests can target an existing cluster instead of a container by exporting `CRDB_TEST_URL=postgresql://user:pass@host:port/db?sslmode=disable`.

## License

Apache 2.0
