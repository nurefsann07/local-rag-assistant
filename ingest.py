import os
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

DOCS_FOLDER = "documents" # sabit tanımlamalr bunlar 
DB_FILE = "knowledge.db"  # veri tabanının ismi bu 
  
# 1. Foundry Local'ı başlat
config = Configuration(app_name="local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Embedding modelini yükle
embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embedding_model.download(lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True))
print()
embedding_model.load()
embedding_client = embedding_model.get_embedding_client()

# 3. Veritabanını oluştur (yoksa)
conn = sqlite3.connect(DB_FILE)   #IF NOT EXISTS demek, "tablo zaten varsa hata verme, sadece atla" demek 
cursor = conn.cursor()
cursor.execute("""  
CREATE TABLE IF NOT EXISTS documents (     
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    content TEXT,
    embedding TEXT
)
""")
conn.commit()

# 4. documents klasöründeki her dosyayı oku ve işle
for filename in os.listdir(DOCS_FOLDER):   #os.listdir(...), bir klasördeki tüm dosya isimlerini listeler.
    if not filename.endswith(".txt"):  #Eğer dosya .txt ile bitmiyorsa (örnek: gizli sistem dosyaları), o dosyayı atla (continue = "bu döngü adımını geç, bir sonrakine geç").
        continue

    filepath = os.path.join(DOCS_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Metni paragraflara böl (boş satırla ayrılmış bloklar = ayrı chunk)
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    print(f"{filename}: {len(chunks)} parça bulundu")

    for chunk in chunks:
        result = embedding_client.generate_embedding(chunk)
        embedding_vector = result.data[0].embedding
        embedding_json = json.dumps(embedding_vector)

        cursor.execute(
            "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
            (filename, chunk, embedding_json)
        )

conn.commit()
conn.close()
embedding_model.unload()

print("Tüm dokümanlar başarıyla veritabanına kaydedildi!")