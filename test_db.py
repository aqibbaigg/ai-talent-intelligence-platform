import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="talent_db",
    user="postgres",
    password="password"  # replace with your PostgreSQL password
)

print("Database Connected Successfully!")

conn.close()