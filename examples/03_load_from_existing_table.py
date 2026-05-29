"""Load existing CRDB rows as LlamaIndex Documents via CockroachDBReader.

Assumes an ``articles`` table:
    CREATE TABLE articles (id INT PRIMARY KEY, body STRING, author STRING, tag STRING);
"""

from llama_index.readers.cockroachdb import CockroachDBReader


def main() -> None:
    reader = CockroachDBReader.from_params(
        host="localhost",
        port=26257,
        user="root",
        password="",
        database="defaultdb",
        sslmode="disable",
    )

    docs = reader.load_data(
        table="articles",
        text_column="body",
        metadata_columns=["id", "author", "tag"],
        id_column="id",
    )
    for d in docs:
        print(f"id={d.id_}  author={d.metadata.get('author')}  text={d.text[:60]}...")

    # Or with a custom SQL and parameters:
    filtered = reader.load_data(
        query="SELECT id, body, author FROM articles WHERE tag = :tag LIMIT 10",
        text_column="body",
        metadata_columns=["id", "author"],
        id_column="id",
        params={"tag": "cockroachdb"},
    )
    print(f"\nfiltered count: {len(filtered)}")


if __name__ == "__main__":
    main()
