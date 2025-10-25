from fastapi import APIRouter

router = APIRouter()

@router.get("/ready")
def readiness_probe():
    return {"status": "ready"}

@router.get("/live")
def liveness_probe():
    return {"status": "alive"}

