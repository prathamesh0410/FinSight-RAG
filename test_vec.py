import sqlite3
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
try:
    conn.execute("SELECT load_extension('vec0')")
    print('sqlite-vec loaded')
except Exception as e:
    print(f'Error: {e}')