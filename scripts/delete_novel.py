import argparse
from database import SessionLocal
from models import Novel


def delete_novel(title: str):
    """Finds a novel by title and deletes it, cascading to all segments."""
    db = SessionLocal()

    try:
        novel = db.query(Novel).filter(Novel.title == title).first()

        if not novel:
            print(f"Could not find a novel titled '{title}' in the database.")
            return

        print(f"Found '{novel.title}' (ID: {novel.id}).")
        print("Deleting novel and all associated entities...")

        db.delete(novel)
        db.commit()

        print("Deletion complete.")

    except Exception as e:
        db.rollback()
        print(f"An error occurred during deletion: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete a novel and all its segments from the database."
    )
    parser.add_argument(
        "title", type=str, help="The exact title of the novel to delete (in quotes)."
    )

    args = parser.parse_args()
    delete_novel(args.title)
