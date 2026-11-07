from sqlalchemy import text
from app.database.connection import engine


def test_postgres_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )

        database, user = result.fetchone()

        assert database == "rag_platform"
        assert user == "rag_user"

        print(f"Connected to {database} as {user}")