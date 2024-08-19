import sqlite3

DB_NAME = "face-auth.db"
CREATE_DB = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email VARCHAR (50) UNIQUE NOT NULL, name VARCHAR (50))"
INSERT_USER = "INSERT INTO users (email, name) VALUES (?, ?)"
SELECT_USER = "SELECT id,name FROM users WHERE email=?"
DELETE_USER = "DELETE FROM users WHERE id=?"


def create_user(email, name):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(CREATE_DB)
        conn.execute(INSERT_USER, [email, name])
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # print(e)
        return False


def get_user(email):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(CREATE_DB)
        cursor = conn.execute(SELECT_USER, [email])
        conn.commit()
        id, name = cursor.fetchone()
        conn.close()
        return id, name
    except Exception as e:
        # print(e)
        return None


def del_user(id):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(DELETE_USER, [id])
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        # print(e)
        return False
