# research/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research.api import routes_experiments

app = FastAPI(title="CPose Research API", version="0.1.0")

# Cho phép dashboard Flask (5000) gọi API research
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký routers
app.include_router(routes_experiments.router, prefix="/experiments", tags=["experiments"])


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


# Nếu muốn chạy trực tiếp: python -m research.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("research.main:app", host="0.0.0.0", port=8000, reload=True)
