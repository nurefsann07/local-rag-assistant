"""
retrieve.py - Retrieval Test Scripti

Embedding tabanlı arama (retrieval) fonksiyonunu tek başına test
etmek için kullanılan bir geliştirme scriptidir. Örnek bir soru
sorup, veritabanındaki en alakalı doküman parçalarını ve
benzerlik skorlarını ekrana yazdırır.

Ana uygulamanın (app.py) çalışması için gerekli değildir.
"""

import json
import sqlite3

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_FILE = "../knowledge.db"
TOP_K = 3
PREVIEW_LENGTH = 150
TEST_QUERY = "Prompt injection nedir?"


def load_embedding_client():
    """Foundry Local'ı başlatır ve embedding modelini yükler."""
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\rİndiriliyor: {p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()

    return embedding_model, embedding_model.get_embedding_client()


def cosine_similarity(vec1, vec2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (-1 ile 1 arası)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_chunks(query, embedding_client, db_file=DB_FILE, top_k=TOP_K):
    """Sorguya en alakalı top_k doküman parçasını, skoruyla birlikte döndürür."""
    result = embedding_client.generate_embedding(query)
    query_vector = result.data[0].embedding

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT source, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    scored = []
    for source, content, embedding_json in rows:
        doc_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, doc_vector)
        scored.append((score, source, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def print_results(query, results):
    print(f"\nSoru: {query}\n")
    for score, source, content in results:
        preview = content[:PREVIEW_LENGTH]
        print(f"Skor: {score:.4f} | Kaynak: {source}")
        print(f"{preview}...\n")


def main():
    embedding_model, embedding_client = load_embedding_client()

    try:
        results = get_top_chunks(TEST_QUERY, embedding_client)
        print_results(TEST_QUERY, results)
    finally:
        embedding_model.unload()


if __name__ == "__main__":
    main()