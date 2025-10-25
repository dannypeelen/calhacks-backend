from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_video, routes_analysis, routes_threats, routes_health
from app.workers.background_tasks import get_task_runner
from app.workers.scheduler import get_scheduler, cleanup_tmp
from app.services.baseten_client import get_baseten_client

app = FastAPI(
    title="SentriAI API",
    description="Video intelligence and real-time threat analysis backend",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(routes_video.router, prefix="/video", tags=["Video"])
app.include_router(routes_analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(routes_threats.router, prefix="/threats", tags=["Threats"])
app.include_router(routes_health.router, prefix="/health", tags=["Health"])

@app.get("/")
def root():
    return {"status": "SentriAI backend running"}


@app.on_event("startup")
async def _startup() -> None:
    # Start background task runner
    await get_task_runner().start()
    # Start periodic scheduler and register housekeeping
    sched = get_scheduler()
    await sched.start()
    # Cleanup temp dir hourly (remove files older than 24h in the task)
    sched.schedule(lambda: cleanup_tmp(24 * 3600), interval_sec=3600)


@app.on_event("shutdown")
async def _shutdown() -> None:
    # Stop scheduler
    await get_scheduler().stop()
    # Stop background tasks
    await get_task_runner().stop()
    # Close shared HTTP client(s)
    try:
        await get_baseten_client().aclose()
    except Exception:
        pass
