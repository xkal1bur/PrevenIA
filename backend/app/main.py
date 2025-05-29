from fastapi import FastAPI

app = FastAPI(
    title="Patient Management API",
    description="API for managing patient data",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Welcome to Patient Management API"} 