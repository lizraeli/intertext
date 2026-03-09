from fastapi.testclient import TestClient
from tests.conftest import SeedData


class TestRandomSegments:
    def test_returns_requested_count(
        self, client: TestClient, seed_data: SeedData
    ) -> None:
        response = client.get("/api/segments/random?count=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_default_count(self, client: TestClient, seed_data: SeedData) -> None:
        response = client.get("/api/segments/random")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3  # only 3 segments in seed data, default is 5

    def test_response_shape(self, client: TestClient, seed_data: SeedData) -> None:
        response = client.get("/api/segments/random?count=1")
        data = response.json()
        segment = data[0]
        assert set(segment.keys()) == {"id", "opening_line", "mood"}
        assert isinstance(segment["id"], int)
        assert isinstance(segment["opening_line"], str)
        assert isinstance(segment["mood"], str)
        assert len(segment["opening_line"]) > 0


class TestGetSegment:
    def test_returns_full_response(
        self, client: TestClient, seed_data: SeedData
    ) -> None:
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}")
        assert response.status_code == 200
        assert response.json() == {
            "id": seg_id,
            "novel_id": seed_data["novel_1"].id,
            "content": "The room was empty and the silence pressed in from every side. She sat alone by the window.",
            "novel_title": "Test Novel",
            "author": "Test Author",
            "year": 2000,
            "mood": "melancholic",
            "setting": "a dark room",
            "themes": [
                {
                    "name": "isolation",
                    "intensity": 0.8,
                    "tone": -0.5,
                    "manifestation": "The character sits alone in an empty room.",
                }
            ],
        }

    def test_not_found(self, client: TestClient, seed_data: SeedData) -> None:
        response = client.get("/api/segments/999999")
        assert response.status_code == 404


class TestSimilarSegments:
    def test_returns_similar_segments(
        self, client: TestClient, seed_data: SeedData
    ) -> None:
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}/similar?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2

    def test_excludes_source_segment(
        self, client: TestClient, seed_data: SeedData
    ) -> None:
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}/similar?limit=10")
        data = response.json()
        returned_ids = [s["id"] for s in data]
        assert seg_id not in returned_ids

    def test_response_shape(self, client: TestClient, seed_data: SeedData) -> None:
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}/similar?limit=1")
        data = response.json()
        assert len(data) == 1
        segment = data[0]
        assert set(segment.keys()) == {
            "id",
            "opening_line",
            "mood",
            "novel_title",
            "author",
            "similarity_score",
        }
        assert segment["id"] == seed_data["seg_c"].id
        assert segment["opening_line"] == "The streets were empty and silent."
        assert segment["mood"] == "contemplative"
        assert segment["novel_title"] == "Other Novel"
        assert segment["author"] == "Other Author"
        assert isinstance(segment["similarity_score"], float)
        assert 0.0 <= segment["similarity_score"] <= 1.0

    def test_ordered_by_similarity(
        self, client: TestClient, seed_data: SeedData
    ) -> None:
        # seg_a and seg_c share the same embedding direction (index 0),
        # while seg_b has a different one (index 1).
        # So from seg_a, seg_c should be more similar than seg_b.
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}/similar?limit=2")
        data = response.json()
        assert len(data) == 2
        assert data[0]["similarity_score"] >= data[1]["similarity_score"]
        assert data[0]["id"] == seed_data["seg_c"].id

    def test_respects_limit(self, client: TestClient, seed_data: SeedData) -> None:
        seg_id = seed_data["seg_a"].id
        response = client.get(f"/api/segments/{seg_id}/similar?limit=1")
        data = response.json()
        assert len(data) == 1

    def test_not_found(self, client: TestClient, seed_data: SeedData) -> None:
        response = client.get("/api/segments/999999/similar")
        assert response.status_code == 404
