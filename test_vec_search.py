import sqlite3
import sqlite_vec
import numpy as np

conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
sqlite_vec.load(conn)

# Create a virtual table for vectors
conn.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[4])")

# Insert some test vectors
test_vectors = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.707, 0.707, 0.0, 0.0],
]

for i, v in enumerate(test_vectors):
    conn.execute("INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)", (i, str(v)))

# Query with a vector
query_vec = [0.8, 0.6, 0.0, 0.0]
cursor = conn.execute(
    "SELECT rowid, vec_distance_cosine(embedding, ?) as distance FROM vec_items ORDER BY distance LIMIT 3",
    (str(query_vec),)
)
results = cursor.fetchall()
print("Cosine distance results:")
for r in results:
    print(f"  rowid={r[0]}, distance={r[1]}, similarity={1-r[1]}")

# Test with dot product
cursor = conn.execute(
    "SELECT rowid, vec_distance_l2(embedding, ?) as distance FROM vec_items ORDER BY distance LIMIT 3",
    (str(query_vec),)
)
results = cursor.fetchall()
print("L2 distance results:")
for r in results:
    print(f"  rowid={r[0]}, distance={r[1]}")