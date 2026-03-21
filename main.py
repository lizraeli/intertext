import os

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db
from schemas import (
    ChapterResponse,
    SegmentResponse,
    TraversalRequest,
    TraversalResponse,
    SegmentPreview,
    SimilarSegmentPreview,
    FullSegmentResponse,
)
from queries import (
    query_chapters_for_novel,
    query_novel_by_id,
    query_novel_segments,
    query_similar_by_vector,
    query_random_segments,
    query_segment_by_id,
    query_prev_segment_id,
    query_next_segment_id,
    query_similar_by_segment,
    query_segment_position_in_chapter,
    query_chapter_segment_count,
)
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/novels/{novel_id}/chapters", response_model=List[ChapterResponse])
def get_novel_chapters(novel_id: int, db: Session = Depends(get_db)):
    if query_novel_by_id(db=db, novel_id=novel_id) is None:
        raise HTTPException(status_code=404, detail="Novel not found")

    chapters = query_chapters_for_novel(db=db, novel_id=novel_id)
    return [ChapterResponse.from_row(chapter) for chapter in chapters]


@app.get("/api/novels/{novel_id}/segments", response_model=List[SegmentResponse])
def get_novel_segments(novel_id: int, db: Session = Depends(get_db)):
    segments = query_novel_segments(db, novel_id)
    if not segments:
        raise HTTPException(status_code=404, detail="Novel not found")

    return [SegmentResponse.from_row(segment) for segment in segments]


@app.post("/api/segments/similar", response_model=List[TraversalResponse])
def find_similar_segments(request: TraversalRequest, db: Session = Depends(get_db)):
    try:
        results = query_similar_by_vector(
            db, request.current_embedding, request.theme_filter, request.limit
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


@app.get("/api/segments/random", response_model=List[SegmentPreview])
def get_random_segments(
    count: int = Query(default=5, ge=1, le=20), db: Session = Depends(get_db)
):
    rows = query_random_segments(db, count)
    return [SegmentPreview.from_row(row) for row in rows]


@app.get("/api/segments/{segment_id}", response_model=FullSegmentResponse)
def get_segment(segment_id: int, db: Session = Depends(get_db)):
    row = query_segment_by_id(db, segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")

    prev_id = query_prev_segment_id(db, row.novel_id, row.chapter.block_index, row.id)
    next_id = query_next_segment_id(db, row.novel_id, row.chapter.block_index, row.id)
    segment_index = query_segment_position_in_chapter(db, row.chapter_id, row.id)
    chapter_segment_count = query_chapter_segment_count(db, row.chapter_id)
    return FullSegmentResponse.from_row(
        row,
        segment_index=segment_index,
        chapter_segment_count=chapter_segment_count,
        prev_segment_id=prev_id,
        next_segment_id=next_id,
    )


@app.get(
    "/api/segments/{segment_id}/similar", response_model=List[SimilarSegmentPreview]
)
def get_similar_segments(
    segment_id: int,
    limit: int = Query(default=3, ge=1, le=10),
    db: Session = Depends(get_db),
):
    source = query_segment_by_id(db, segment_id)
    if not source:
        raise HTTPException(status_code=404, detail="Segment not found")

    similar_rows = query_similar_by_segment(db, segment_id, source, limit)

    return [SimilarSegmentPreview.from_row(row) for row in similar_rows]
