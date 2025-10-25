from fastapi import APIRouter, Query
from app.services.transcript_builder import build_transcript
from app.services.summarizer import summarize_detections
from app.services.threat_score import compute_threat_score

router = APIRouter()

@router.get("/score")
async def get_threat_score(session_id: str = Query(...)):
    """Compute overall threat score from combined detections."""
    score = await compute_threat_score(session_id)
    return {"session_id": session_id, "threat_score": score}


@router.get("/transcript")
async def get_event_transcript(session_id: str = Query(...)):
    """Build chronological transcript of events for this session."""
    transcript = await build_transcript(session_id)
    return {"session_id": session_id, "transcript": transcript}


@router.get("/summary")
async def get_summary(session_id: str = Query(...)):
    """Use LLM to summarize key detections."""
    summary = await summarize_detections(session_id)
    return {"session_id": session_id, "summary": summary}

