"""
app.py - Ana Uygulama

Bu script, yerel RAG (Retrieval-Augmented Generation) asistanının
çalışan arayüzüdür. Kullanıcıdan soru alır, SQLite veritabanındaki
en alakalı doküman parçalarını bulur (retrieval) ve bu parçaları
kullanarak yerel bir dil modeliyle (phi-3.5-mini) cevap üretir.

Çalıştırmadan önce 'python ingest.py' ile veritabanının
oluşturulmuş olması gerekir.
"""

import json
import sqlite3

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_FILE = "knowledge.db"
SIMILARITY_THRESHOLD = 0.55  # Bu skorun altındaki sonuçlar "bilmiyorum" sayılır
TOP_K = 3  # Her soru için getirilecek doküman parçası sayısı

SYSTEM_PROMPT = (
    "Sen bir yapay zeka ve siber güvenlik asistanısın. "
    "SADECE sana verilen bağlamı kullanarak cevap ver. "
    "Bağlamda sorunun cevabı yoksa veya bağlam alakasızsa, "
    "kesinlikle kendi bilginden cevap UYDURMA, sadece "
    "'Bu konuda elimde bilgi yok' de. Kısa ve net cevaplar ver."
)

NO_ANSWER_MESSAGE = (
    "Bu konuda elimde bilgi yok. Sadece yapay zeka ve siber güvenlik "
    "temelleri hakkında sorularınızı cevaplayabilirim."
)


def load_models():
    """Foundry Local'ı başlatır, embedding ve chat modellerini yükler."""
    config = Configuration(app_name="local_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli yükleniyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(lambda p: print(f"\r{p:.1f}%", end="", flush=True))
    print()
    embedding_model.load()

    print("Chat modeli yükleniyor...")
    chat_model = manager.catalog.get_model("phi-3.5-mini")
    chat_model.download(lambda p: print(f"\r{p:.1f}%", end="", flush=True))
    print()
    chat_model.load()

    return embedding_model, chat_model


def cosine_similarity(vec1, vec2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar (-1 ile 1 arası)."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_chunks(query, embedding_client, top_k=TOP_K):
    """Sorguya en alakalı top_k doküman parçasını veritabanından getirir."""
    result = embedding_client.generate_embedding(query)
    query_vector = result.data[0].embedding

    conn = sqlite3.connect(DB_FILE)
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


def answer_query(query, embedding_client, chat_client):
    """
    Kullanıcı sorusuna RAG akışıyla cevap üretir:
    1. En alakalı parçaları bul
    2. Skor çok düşükse modeli hiç çağırmadan "bilmiyorum" de
    3. Aksi halde bağlamı modele gönderip cevap üret
    """
    top_chunks = get_top_chunks(query, embedding_client)

    best_score = top_chunks[0][0]
    if best_score < SIMILARITY_THRESHOLD:
        return NO_ANSWER_MESSAGE, top_chunks

    context = "\n\n".join(
        f"[Kaynak: {source}]\n{content}" for score, source, content in top_chunks
    )
    user_prompt = f"Bağlam:\n{context}\n\nSoru: {query}"

    response = chat_client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    return response.choices[0].message.content, top_chunks


def print_sources(sources):
    """Kullanılan kaynakları, tekrarları eleyerek ekrana yazdırır."""
    print("\nKullanılan kaynaklar:")
    seen_sources = set()
    for score, source, content in sources:
        if source not in seen_sources:
            print(f"  - {source} (en yüksek skor: {score:.3f})")
            seen_sources.add(source)
    print()


def main():
    embedding_model, chat_model = load_models()
    embedding_client = embedding_model.get_embedding_client()
    chat_client = chat_model.get_chat_client()

    print("\n=== Yerel RAG Asistanı ===")
    print("Çıkmak için 'exit' yazın.\n")

    try:
        while True:
            query = input("Soru: ")
            if query.lower() == "exit":
                break

            answer, sources = answer_query(query, embedding_client, chat_client)

            print(f"\nCevap: {answer}")
            print_sources(sources)
    finally:
        embedding_model.unload()
        chat_model.unload()
        print("Görüşürüz!")


if __name__ == "__main__":
    main()