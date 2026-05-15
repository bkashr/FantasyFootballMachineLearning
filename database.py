import sqlite3 

DB_PATH = "fantasy.db"

def players_database():
  conn = sqlite3.connect(DB_PATH)
  cur = conn.cursor()

  cur.executescript("""
    CREATE TABLE IF NOT EXISTS players;

    CREATE TABLE players (
    player_id      TEXT PRIMARY KEY,
    full_name      TEXT,
    position       TEXT, 
    
    
