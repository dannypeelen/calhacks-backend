from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_video, routes_analysis, routes_threats, routes_health

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

