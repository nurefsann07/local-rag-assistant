import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_FILE = "knowledge.db"

# 1. Foundry Local'ı başlat
config = Configuration(app_name="local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. İki modeli de yükle: embedding (arama için) ve chat (cevap üretmek için)
print("Embedding modeli yükleniyor...")
embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.download(lambda p: print(f"\r{p:.1f}%", end="", flush=True))
print()
embedding_model.load()
embedding_client = embedding_model.get_embedding_client()

print("Chat modeli yükleniyor...")
chat_model = manager.catalog.get_model("phi-3.5-mini")
chat_model.download(lambda p: print(f"\r{p:.1f}%", end="", flush=True))
print()
chat_model.load()
chat_client = chat_model.get_chat_client()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_chunks(query, top_k=3):
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


def answer_query(query):
    # 1. En alakalı parçaları bul
    top_chunks = get_top_chunks(query, top_k=3)

    # 2. YENİ: En yüksek skor çok düşükse, modele hiç sormadan cevap ver
    best_score = top_chunks[0][0]
    if best_score < 0.55:
        return "Bu konuda elimde bilgi yok. Sadece yapay zeka ve siber güvenlik temelleri hakkında sorularınızı cevaplayabilirim.", top_chunks

    # 3. Bulunan parçaları tek bir metin haline getir
    context = "\n\n".join([f"[Kaynak: {source}]\n{content}" for score, source, content in top_chunks])

    # 4. Sistem talimatı + bağlam + soru ile modele gönder
    system_prompt = (
        "Sen bir yapay zeka ve siber güvenlik asistanısın. "
        "SADECE sana verilen bağlamı kullanarak cevap ver. "
        "Bağlamda sorunun cevabı yoksa veya bağlam alakasızsa, "
        "kesinlikle kendi bilginden cevap UYDURMA, sadece 'Bu konuda elimde bilgi yok' de. "
        "Kısa ve net cevaplar ver."
    )

    user_prompt = f"Bağlam:\n{context}\n\nSoru: {query}"

    response = chat_client.complete_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    return response.choices[0].message.content, top_chunks

# --- Ana Döngü ---
if __name__ == "__main__":
    print("\n=== Yerel RAG Asistanı ===")
    print("Çıkmak için 'exit' yazın.\n")

    while True:
        query = input("Soru: ")
        if query.lower() == "exit":
            break

        answer, sources = answer_query(query)

        print("\nKullanılan kaynaklar:")
        seen_sources = set()
        for score, source, content in sources:
            if source not in seen_sources:
             print(f"  - {source} (en yüksek skor: {score:.3f})")
             seen_sources.add(source)
        print()

    embedding_model.unload()
    chat_model.unload()
    print("Görüşürüz!")