from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import date
from .cli import generate_report_once
from dotenv import load_dotenv
load_dotenv()   # loads .env into os.environ

app = FastAPI(title="Agentic Portfolio")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
def root():
    return {"ok": True, "message": "Agentic Portfolio API. See /report for latest HTML."}

@app.get("/report", response_class=Response, responses={200: {"content": {"text/html": {}}}})
def report():
    out = generate_report_once("today")
    html_path = Path(out["path"])
    return Response(content=html_path.read_text(encoding="utf-8"), media_type="text/html")
