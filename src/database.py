from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL is not defined in .env"
    )


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT version();")
        )

        print("Connected successfully.")
        print(result.scalar())


if __name__ == "__main__":
    test_connection()