import sqlite3

conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM documents")
total = cursor.fetchone()[0]
print(f"Toplam kayıt sayısı: {total}")

cursor.execute("SELECT source, content FROM documents LIMIT 3")
rows = cursor.fetchall()

print("\nİlk 3 kayıt:")
for row in rows:
    source, content = row
    print(f"\n--- Kaynak: {source} ---")
    print(content[:100] + "...")  # ilk 100 karakteri göster

conn.close()