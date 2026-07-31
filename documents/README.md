# Yerel RAG Asistanı (Local RAG Assistant)

Microsoft Foundry Local kullanılarak geliştirilmiş, tamamen çevrimdışı çalışan bir doküman soru-cevap asistanı. Yapay zeka ve siber güvenlik temelleri hakkında 8 doküman üzerinde, RAG (Retrieval-Augmented Generation) yöntemiyle soruları yanıtlar.

## Proje Hakkında

Bu proje, Microsoft Summer School kapsamında geliştirilmiştir. Amaç, bir LLM'in (büyük dil modeli) kendi bilgisi yerine, sağlanan belgelerden aldığı bilgiyle cevap üretmesini sağlamaktır — bu sayede hem daha doğru cevaplar alınır hem de model "halüsinasyon" görmez (bilmediği bir şeyi uydurmaz).

Tüm sistem **internete bağlanmadan**, tamamen kullanıcının kendi bilgisayarında çalışır.

## Nasıl Çalışır

1. **Doküman İşleme (`ingest.py`):** `documents` klasöründeki metin dosyaları paragraflara bölünür, her paragraf embedding modeliyle (qwen3-embedding-0.6b) sayısal vektöre çevrilir ve SQLite veritabanına kaydedilir.
2. **Bilgi Getirme (Retrieval):** Kullanıcı bir soru sorduğunda, soru da embedding'e çevrilir. Kosinüs benzerliği kullanılarak, veritabanındaki en alakalı 3 doküman parçası bulunur.
3. **Cevap Üretme (Generation):** Bulunan parçalar, kullanıcının sorusuyla birlikte yerel bir dil modeline (phi-3.5-mini) gönderilir. Model, sadece bu bağlamı kullanarak cevap üretir.
4. **Güvenlik Katmanı:** Eğer bulunan en alakalı parçanın benzerlik skoru belirli bir eşiğin (0.55) altındaysa, sistem modele hiç sormadan "bu konuda bilgim yok" cevabını döndürür. Bu, modelin alakasız konularda uydurma cevap vermesini engeller.

## Kullanılan Teknolojiler

- **Microsoft Foundry Local** — yerel (offline) LLM çalıştırma altyapısı
- **phi-3.5-mini** — cevap üretme (chat) modeli
- **qwen3-embedding-0.6b** — metin embedding modeli
- **SQLite** — doküman ve embedding'lerin saklandığı yerel veritabanı
- **NumPy** — kosinüs benzerliği hesaplaması için
- **Python 3.14**

## Kurulum

### Gereksinimler
- Windows 10/11 (64-bit)
- Python 3.10+

### Adımlar

1. Foundry Local'ı kurun:
```
winget install Microsoft.FoundryLocal
```

2. Bu repoyu klonlayın ve klasöre girin:
```
git clone <repo-linki>
cd local-rag-assistant
```

3. Sanal ortam oluşturup aktif edin:
```
python -m venv venv
venv\Scripts\activate
```

4. Gerekli kütüphaneleri kurun:
```
pip install foundry-local-sdk openai numpy
```

## Kullanım

1. Önce dokümanları veritabanına işleyin (sadece ilk çalıştırmada gerekli):
```
python ingest.py
```

2. Asistanı başlatın:
```
python app.py
```

3. Sorularınızı yazın, çıkmak için `exit` yazın.

### Örnek Kullanım

```
Soru: Prompt injection nedir?
Cevap: Prompt injection, LLM tabanlı uygulamalarını etkisiz niyetli
talimatlere gizlenerek, onları manipüle etme saldırısıdır.

Kullanılan kaynaklar:
  - 04_prompt_injection.txt (skor: 0.808)
```

## Proje Yapısı

```
local-rag-assistant/
├── documents/              # Kaynak dokümanlar (8 adet .txt dosyası)
├── ingest.py                # Dokümanları işleyip veritabanına kaydeden script
├── retrieve.py               # Retrieval fonksiyonlarını test eden script
├── app.py                    # Ana uygulama (chat arayüzü)
├── check_db.py                # Veritabanını doğrulama scripti
├── knowledge.db                # SQLite veritabanı (embedding'ler burada)
└── README.md
```

## Öğrendiklerim

- RAG (Retrieval-Augmented Generation) mimarisinin nasıl çalıştığı
- Text embedding ve kosinüs benzerliği ile anlamsal arama
- Yerel (on-device) LLM çalıştırmanın pratik zorlukları (GPU/CPU uyumluluğu, model versiyonları)
- LLM halüsinasyonunu, retrieval skoruna dayalı bir eşik mekanizmasıyla azaltma
- SQLite ile basit ama etkili veri saklama

## Geliştirilebilecek Noktalar

- Daha büyük ölçekli veri setleri için gerçek bir vektör veritabanı (ör. ChromaDB) kullanımı
- Web tabanlı bir arayüz (Streamlit) eklenmesi
- Kaynak gösteriminde aynı dosyadan gelen tekrarların birleştirilmesi
- Çoklu dil desteğinin test edilmesi

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.