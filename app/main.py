from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import routes_video, routes_analysis, routes_threats, routes_health
from app.workers.background_tasks import get_task_runner
from app.workers.scheduler import get_scheduler, cleanup_tmp
from app.services.baseten_client import get_baseten_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await get_task_runner().start()
    sched = get_scheduler()
    await sched.start()
    sched.schedule(lambda: cleanup_tmp(24 * 3600), interval_sec=3600)
    try:
        yield
    finally:
        # Shutdown
        await get_scheduler().stop()
        await get_task_runner().stop()
        try:
            await get_baseten_client().aclose()
        except Exception:
            pass


app = FastAPI(
    title="SentriAI API",
    description="Video intelligence and real-time threat analysis backend",
    version="1.0.0",
    lifespan=lifespan,
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

