"""Vercel serverless entry point for the FastAPI face anonymization backend."""

from mangum import Mangum

from app.main import create_app

app = create_app()
handler = Mangum(app)
