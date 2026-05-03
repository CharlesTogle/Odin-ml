from fastapi import FastAPI


app = FastAPI(
    title="Odin ML Service",
    version="0.1.0",
    description="Machine learning microservice for inference and model-facing APIs.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "odin-ml", "status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Odin ML service is running."}
