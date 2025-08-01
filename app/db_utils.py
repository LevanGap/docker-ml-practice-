import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Get DB connection info from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mlapp")
DB_USER = os.getenv("DB_USER", "mluser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlpass")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def log_prediction(features, prediction):
    """
    Insert prediction info into the predictions table.
    `features` should be saved as string (e.g., JSON or repr)
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        
        cursor.execute(
            """
            INSERT INTO predictions (input_features, prediction, created_at)
            VALUES (%s, %s, %s)
            """,
            (str(features), str(prediction), datetime.now())
        )
        conn.commit()
        cursor.close()
    except Exception as e:
        print("Error logging prediction:", e)
    finally:
        if conn:
            conn.close()
