from pathlib import Path

from database import engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SQL_PATH = (
    PROJECT_ROOT
    / "sql"
    / "create_feature_view.sql"
)


def main():

    print(
        "=== CREATING POSTGRESQL FEATURE VIEW ==="
    )

    sql_text = SQL_PATH.read_text()

    connection = engine.raw_connection()

    try:
        cursor = connection.cursor()

        # Our SQL file contains simple statements
        # separated by semicolons.
        for statement in sql_text.split(";"):

            statement = statement.strip()

            if statement:
                cursor.execute(statement)

        connection.commit()

        print(
            "Feature view created successfully."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()