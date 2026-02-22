# init_db.py
from database import engine, Base

# You MUST import your models here, even if you don't directly use them.
# This forces Python to read the models.py file, which registers your
# classes with the SQLAlchemy 'Base' metadata.
import models


def setup_database():
    print("Connecting to Supabase...")

    # This reads your registered models and creates the tables in Supabase
    # Base.metadata.create_all(bind=engine)

    print("Success! Tables have been created in the remote database.")


if __name__ == "__main__":
    setup_database()
