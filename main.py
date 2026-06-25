from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.health import router as health_router

app = FastAPI(
    title="AI Health Monitoring API",
    description="Smart Wearable Health Monitoring System",
    version="1.0.0"
)

# Allow mobile app and dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health_router, prefix="/api/health")

@app.get("/")
def root():
    return {
        "message": "AI Health Monitoring API is running",
        "status": "active"
    }