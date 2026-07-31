from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Foundry Local'ı başlat
config = Configuration(app_name="local_rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# 2. Modeli katalogdan seç
model = manager.catalog.get_model("phi-3.5-mini")

# 3. Modelin indirildiğinden emin ol (indirilmediyse indirir)
print("Model kontrol ediliyor / indiriliyor...")
model.download(
    lambda progress: print(f"\rİndiriliyor: {progress:.1f}%", end="", flush=True)
)
print()  # yeni satıra geç

# 4. Modeli yükle
model.load()

# 5. Chat client al
client = model.get_chat_client()

# 6. Soru sor
response = client.complete_chat([
    {"role": "user", "content": "Merhaba, sen kimsin?"}
])

print(response.choices[0].message.content)

# 7. Modeli bellekten kaldır
model.unload()

# --- EMBEDDING TESTİ ---
embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")

print("Embedding modeli indiriliyor...")
embedding_model.download(
    lambda progress: print(f"\rİndiriliyor: {progress:.1f}%", end="", flush=True)
)
print()

embedding_model.load()

embedding_client = embedding_model.get_embedding_client()

result = embedding_client.generate_embedding("Merhaba dünya")
print("Embedding vektörünün ilk 5 sayısı:", result.data[0].embedding[:5])
print("Vektörün toplam uzunluğu:", len(result.data[0].embedding))

embedding_model.unload()