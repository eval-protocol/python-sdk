from dotenv import load_dotenv
import os
import psycopg2


def connect_database():
    # Load environment variables from .env
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    # Fetch variables
    USER = os.getenv("SUPABASE_USER")
    PASSWORD = os.getenv("SUPABASE_PASSWORD")
    HOST = os.getenv("SUPABASE_HOST")
    PORT = os.getenv("SUPABASE_PORT")
    DBNAME = os.getenv("SUPABASE_DATABASE")

    # Connect to the database
    connection = psycopg2.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME)
    print("Connection successful!")

    # Create a cursor to execute SQL queries
    cursor = connection.cursor()

    # Get schema of the database
    introspection_query = """select
  table_name,
  column_name,
  data_type,
  is_nullable
from
  information_schema.columns
where
  table_schema = 'public'
order by
  table_name,
  ordinal_position;"""
    cursor.execute(introspection_query)
    result = cursor.fetchall()
    return connection, cursor, result
