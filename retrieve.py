import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_FILE = "knowledge.db"

# 1. Foundry Local'ı başlat
config = Configuration(app_name="local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Embedding modelini yükle
embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.download(lambda p: print(f"\rİndiriliyor: {p:.1f}%", end="", flush=True))
print()
embedding_model.load()
embedding_client = embedding_model.get_embedding_client()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_chunks(query, top_k=3):
    # Sorunun embedding'ini üret
    result = embedding_client.generate_embedding(query)
    query_vector = result.data[0].embedding

    # Veritabanındaki tüm kayıtları çek
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT source, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    # Her kayıt için benzerlik skorunu hesapla
    scored = []
    for source, content, embedding_json in rows:
        doc_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, doc_vector)
        scored.append((score, source, content))

    # En yüksek skordan en düşüğe doğru sırala
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:top_k]


# Test amaçlı: bu dosyayı direkt çalıştırırsak deneme yapalım
if __name__ == "__main__":
    query = "Prompt injection nedir?"
    results = get_top_chunks(query)

    print(f"\nSoru: {query}\n")
    for score, source, content in results:
        print(f"Skor: {score:.4f} | Kaynak: {source}")
        print(content[:150] + "...\n")

    embedding_model.unload()