import os
from typing import Generator, TypedDict

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from fastapi.testclient import TestClient

from database import Base, get_db
from main import app
from models import (
    Novel,
    NovelCharacter,
    NovelChapter,
    NovelMood,
    NovelPlace,
    NovelSegment,
    NovelTheme,
    SegmentTheme,
)


class SeedData(TypedDict):
    novel_1: Novel
    novel_2: Novel
    seg_a: NovelSegment
    seg_b: NovelSegment
    seg_c: NovelSegment

load_dotenv()

DATABASE_TEST_URL = os.getenv("DATABASE_TEST_URL")
if not DATABASE_TEST_URL:
    raise ValueError("DATABASE_TEST_URL is not set in the .env file")

test_engine = create_engine(DATABASE_TEST_URL, pool_pre_ping=True)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def _make_embedding(index: int, dims: int = 3072) -> list[float]:
    """Create a synthetic unit-direction embedding with a 1.0 at the given index."""
    vec = [0.0] * dims
    vec[index % dims] = 1.0
    return vec


SEED_METADATA_A = {
    "primary_themes": [
        {
            "name": "isolation",
            "intensity": 0.8,
            "tone": -0.5,
            "manifestation": "The character sits alone in an empty room.",
        }
    ],
    "characters": ["Alice"],
    "place": "a dark room",
    "mood": "melancholic",
    "chapter": "Chapter 1",
}

SEED_METADATA_B = {
    "primary_themes": [
        {
            "name": "hope",
            "intensity": 0.9,
            "tone": 0.7,
            "manifestation": "Light breaks through the window at dawn.",
        }
    ],
    "characters": ["Bob"],
    "place": "a hilltop",
    "mood": "tender",
    "chapter": "Chapter 2",
}

SEED_METADATA_C = {
    "primary_themes": [
        {
            "name": "isolation",
            "intensity": 0.6,
            "tone": -0.3,
            "manifestation": "The streets are empty and silent.",
        }
    ],
    "characters": ["Clara"],
    "place": "an abandoned city",
    "mood": "contemplative",
    "chapter": "Chapter 1",
}


@pytest.fixture(scope="session", autouse=True)
def setup_database() -> Generator[None, None, None]:
    with test_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestSession(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seed_data(db_session: Session) -> SeedData:
    novel_1 = Novel(title="Test Novel", author="Test Author", publication_year=2000)
    novel_2 = Novel(title="Other Novel", author="Other Author", publication_year=2010)
    db_session.add_all([novel_1, novel_2])
    db_session.flush()

    place_a = NovelPlace(novel_id=novel_1.id, name="a dark room")
    place_b = NovelPlace(novel_id=novel_1.id, name="a hilltop")
    place_c = NovelPlace(novel_id=novel_2.id, name="an abandoned city")
    db_session.add_all([place_a, place_b, place_c])
    db_session.flush()

    mood_a = NovelMood(novel_id=novel_1.id, name="melancholic")
    mood_b = NovelMood(novel_id=novel_1.id, name="tender")
    mood_c = NovelMood(novel_id=novel_2.id, name="contemplative")
    db_session.add_all([mood_a, mood_b, mood_c])
    db_session.flush()

    chapter_n1_b0 = NovelChapter(
        novel_id=novel_1.id, block_index=0, title="Chapter 1"
    )
    chapter_n1_b1 = NovelChapter(
        novel_id=novel_1.id, block_index=1, title="Chapter 2"
    )
    chapter_n2_b0 = NovelChapter(
        novel_id=novel_2.id, block_index=0, title="Chapter 1"
    )
    db_session.add_all([chapter_n1_b0, chapter_n1_b1, chapter_n2_b0])
    db_session.flush()

    seg_a = NovelSegment(
        novel_id=novel_1.id,
        chapter_id=chapter_n1_b0.id,
        start_index=0,
        end_index=100,
        content="The room was empty and the silence pressed in from every side. She sat alone by the window.",
        token_count=20,
        metadata_col=SEED_METADATA_A,
        place_id=place_a.id,
        mood_id=mood_a.id,
        embedding=_make_embedding(0),
    )
    seg_b = NovelSegment(
        novel_id=novel_1.id,
        chapter_id=chapter_n1_b1.id,
        start_index=101,
        end_index=200,
        content="Light broke through the clouds at dawn. It was the first warmth she had felt in weeks.",
        token_count=18,
        metadata_col=SEED_METADATA_B,
        place_id=place_b.id,
        mood_id=mood_b.id,
        embedding=_make_embedding(1),
    )
    seg_c = NovelSegment(
        novel_id=novel_2.id,
        chapter_id=chapter_n2_b0.id,
        start_index=0,
        end_index=150,
        content="The streets were empty and silent. No one had walked here in years.",
        token_count=15,
        metadata_col=SEED_METADATA_C,
        place_id=place_c.id,
        mood_id=mood_c.id,
        embedding=_make_embedding(0),
    )

    db_session.add_all([seg_a, seg_b, seg_c])
    db_session.flush()

    char_alice = NovelCharacter(novel_id=novel_1.id, name="Alice")
    char_bob = NovelCharacter(novel_id=novel_1.id, name="Bob")
    char_clara = NovelCharacter(novel_id=novel_2.id, name="Clara")
    db_session.add_all([char_alice, char_bob, char_clara])
    db_session.flush()

    seg_a.characters = [char_alice]
    seg_b.characters = [char_bob]
    seg_c.characters = [char_clara]
    db_session.flush()

    theme_isolation_n1 = NovelTheme(novel_id=novel_1.id, name="isolation")
    theme_hope_n1 = NovelTheme(novel_id=novel_1.id, name="hope")
    theme_isolation_n2 = NovelTheme(novel_id=novel_2.id, name="isolation")
    db_session.add_all([theme_isolation_n1, theme_hope_n1, theme_isolation_n2])
    db_session.flush()

    db_session.add_all(
        [
            SegmentTheme(
                segment_id=seg_a.id,
                theme_id=theme_isolation_n1.id,
                intensity=0.8,
                tone=-0.5,
                manifestation="The character sits alone in an empty room.",
            ),
            SegmentTheme(
                segment_id=seg_b.id,
                theme_id=theme_hope_n1.id,
                intensity=0.9,
                tone=0.7,
                manifestation="Light breaks through the window at dawn.",
            ),
            SegmentTheme(
                segment_id=seg_c.id,
                theme_id=theme_isolation_n2.id,
                intensity=0.6,
                tone=-0.3,
                manifestation="The streets are empty and silent.",
            ),
        ]
    )
    db_session.flush()

    return {
        "novel_1": novel_1,
        "novel_2": novel_2,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "seg_c": seg_c,
    }
