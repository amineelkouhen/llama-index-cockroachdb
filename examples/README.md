# Examples

Each script is runnable. They assume a CockroachDB v25.2+ instance reachable at
`localhost:26257` (insecure for local dev). For a managed cluster, override
`host`/`port`/`user`/`password`/`sslmode` accordingly.

```bash
# Local single-node CRDB
cockroach start-single-node --insecure --listen-addr=localhost:26257 &
cockroach sql --insecure -e "SET CLUSTER SETTING feature.vector_index.enabled = true;"
```

| Script | What it shows |
| --- | --- |
| `01_basic_usage.py` | Create a store, add nodes, run a default vector query |
| `02_with_openai_embeddings.py` | End-to-end RAG with OpenAI embeddings + `VectorStoreIndex` |
| `03_load_from_existing_table.py` | `CockroachDBReader` over an existing CRDB table |
| `04_custom_retriever.py` | `CockroachDBRetriever` with C-SPANN beam-size tuning |
| `05_async_usage.py` | `aadd` / `aquery` |
| `06_mmr_query.py` | MMR mode for diverse results |
| `07_metadata_filters.py` | `MetadataFilters` with EQ / IN / range operators |
| `08_walkthrough.ipynb` | Shareable Jupyter notebook covering every public surface end-to-end |
