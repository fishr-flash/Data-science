import sqlite3
import os
from string.templatelib import Template, Interpolation

# Setup: Create demo database
DB_PATH = "demo.db"


def reset_database():
	"""Reset the database to initial state."""
	if os.path.exists(DB_PATH):
		os.remove(DB_PATH)

	conn = sqlite3.connect(DB_PATH)
	cursor = conn.cursor()

	cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    """)

	users = [
		("alice", "alice@example.com"),
		("bob", "bob@example.com"),
		("admin", "admin@example.com"),
	]

	cursor.executemany("INSERT INTO users (name, email) VALUES (?, ?)", users)
	conn.commit()
	conn.close()


def get_table_count(cursor):
	"""Return number of tables in database."""
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
	return len(cursor.fetchall())


# Test unsafe f-string
reset_database()
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

user_input = "admin'; DROP TABLE users; --"

# Unsafe f-string
unsafe_query = f"SELECT * FROM users WHERE name = '{user_input}'"
print(f"Query: {unsafe_query}")
cursor.executescript(unsafe_query)
print(f"Tables: {get_table_count(cursor)}\n")

# Reset and test safe version
reset_database()
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


def safe_sql(template: Template) -> str:
	"""Escape SQL values to prevent injection."""
	result = []
	for item in template:
		if isinstance(item, Interpolation):
			# Escape single quotes by doubling them
			safe_value = str(item.value).replace("'", "''")
			result.append(f"'{safe_value}'")
		else:
			result.append(item)
	return "".join(result)


# Safe t-string
query = t"SELECT * FROM users WHERE name = {user_input}"

safe_query = safe_sql(query)
print(f"Query: {safe_query}")
cursor.execute(safe_query)
print(f"Tables: {get_table_count(cursor)}")

conn.close()
