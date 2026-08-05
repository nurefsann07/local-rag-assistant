"""
check_db.py - Veritabanı Doğrulama Scripti

knowledge.db veritabanının doğru oluşturulduğunu ve ingest.py
tarafından kaydedilen verilerin beklendiği gibi göründüğünü
kontrol etmek için kullanılan bir geliştirme/hata ayıklama scriptidir.

Ana uygulamanın (app.py) çalışması için gerekli değildir.
"""

import sqlite3

DB_FILE = "../knowledge.db"
PREVIEW_COUNT = 3
PREVIEW_LENGTH = 100


def check_database(db_file, preview_count=PREVIEW_COUNT):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")
    total = cursor.fetchone()[0]
    print(f"Toplam kayıt sayısı: {total}")

    cursor.execute(f"SELECT source, content FROM documents LIMIT {preview_count}")
    rows = cursor.fetchall()

    print(f"\nİlk {preview_count} kayıt:")
    for source, content in rows:
        preview = content[:PREVIEW_LENGTH]
        print(f"\n--- Kaynak: {source} ---")
        print(f"{preview}...")

    conn.close()


if __name__ == "__main__":
    check_database(DB_FILE)