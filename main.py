from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Novel, NovelSegment
from schemas import SegmentResponse, TraversalRequest, TraversalResponse
from typing import List

app = FastAPI()


@app.get("/api/novels/{novel_id}/segments", response_model=List[SegmentResponse])
def get_novel_segments(novel_id: int, db: Session = Depends(get_db)):
    segments = (
        db.query(NovelSegment)
        .filter(NovelSegment.novel_id == novel_id)
        .order_by(NovelSegment.macro_block_id, NovelSegment.id)
        .all()
    )
    if not segments:
        raise HTTPException(status_code=404, detail="Novel not found")

    return [
        SegmentResponse(
            id=seg.id,
            content=seg.content,
            themes=seg.metadata_col.get("primary_themes", []),
        )
        for seg in segments
    ]


@app.post("/api/segments/similar", response_model=List[TraversalResponse])
def find_similar_segments(request: TraversalRequest, db: Session = Depends(get_db)):
    try:
        query = db.query(
            Novel.title,
            Novel.author,
            NovelSegment.content,
            NovelSegment.embedding.cosine_distance(request.current_vector).label(
                "distance"
            ),
        ).join(Novel, NovelSegment.novel_id == Novel.id)

        if request.theme_filter:
            query = query.filter(
                NovelSegment.metadata_col["primary_themes"].contains(
                    [request.theme_filter]
                )
            )

        results = (
            query.order_by(
                NovelSegment.embedding.cosine_distance(request.current_vector)
            )
            .limit(request.limit)
            .all()
        )

        return [
            TraversalResponse(
                novel_title=r.title,
                author=r.author,
                content=r.content,
                similarity_score=1.0 - r.distance,
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
