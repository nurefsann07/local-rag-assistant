"""
ingest.py - Doküman İşleme Scripti

'documents' klasöründeki .txt dosyalarını okur, paragraflara böler,
her paragrafı embedding modeliyle (qwen3-embedding-0.6b) vektöre
çevirir ve SQLite veritabanına (knowledge.db) kaydeder.

Bu script sadece bir kez, ya da dokümanlar değiştiğinde çalıştırılmalıdır.
"""

import json
import os
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

DOCS_FOLDER = "documents"
DB_FILE = "knowledge.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    content TEXT,
    embedding TEXT
)
"""


def load_embedding_model():
    """Foundry Local'ı başlatır ve embedding modelini yükler."""
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()

    return embedding_model


def create_database(db_file):
    """Veritabanı bağlantısını açar ve documents tablosunu oluşturur (yoksa)."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn, cursor


def split_into_chunks(text):
    """Metni boş satırla ayrılmış paragraflara (chunk) böler."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def ingest_documents(docs_folder, cursor, embedding_client):
    """documents klasöründeki her .txt dosyasını okuyup veritabanına kaydeder."""
    for filename in os.listdir(docs_folder):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(docs_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_into_chunks(text)
        print(f"{filename}: {len(chunks)} parça bulundu")

        for chunk in chunks:
            result = embedding_client.generate_embedding(chunk)
            embedding_json = json.dumps(result.data[0].embedding)

            cursor.execute(
                "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
                (filename, chunk, embedding_json),
            )


def main():
    embedding_model = load_embedding_model()
    embedding_client = embedding_model.get_embedding_client()

    conn, cursor = create_database(DB_FILE)

    try:
        ingest_documents(DOCS_FOLDER, cursor, embedding_client)
        conn.commit()
        print("Tüm dokümanlar başarıyla veritabanına kaydedildi!")
    finally:
        conn.close()
        embedding_model.unload()


if __name__ == "__main__":
    main()