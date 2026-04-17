from fastapi import FastAPI
from .api import routes_experiments
import uvicorn

app = FastAPI(title="CPose Research Server")

app.include_router(routes_experiments.router, prefix="/experiments", tags=["Experiments"])

@app.get("/")
def read_root():
    return {"message": "Welcome to CPose Research API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
