"""
api.py — FastAPI wrapper for vectorize_floor pipeline.

Endpoints:
  POST /vectorize   — upload files, run pipeline, get job_id
  GET  /status/{id} — check job status
  GET  /result/{id} — download ZIP with all outputs
  GET  /            — serve frontend UI
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Import pipeline from our script
from vectorize_floor import run_pipeline, ValidationReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="Floor Plan Vectorizer",
    description="Converts raster floor plan images to SVG polygons.",
    version="1.0.0",
)

# In-memory job store (replace with Redis for production multi-worker setup)
JOBS: dict[str, dict] = {}
WORK_DIR = Path("/app/output")
WORK_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# ---------------------------------------------------------------------------
# Root → serve UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Floor Plan Vectorizer API</h1><p>See /docs</p>")


# ---------------------------------------------------------------------------
# POST /vectorize
# ---------------------------------------------------------------------------

@app.post("/vectorize", summary="Upload floor plan files and start vectorization")
async def vectorize(
    plan: UploadFile = File(..., description="Floor plan image (PNG/JPG)"),
    overlay: Optional[UploadFile] = File(None, description="Overlay image with fills (optional, defaults to plan)"),
    mapping: Optional[UploadFile] = File(None, description="Lot mapping CSV or JSON (optional)"),
    ocr_fallback: bool = Form(False, description="Enable OCR fallback for lot ID detection"),
):
    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)

    JOBS[job_id] = {"status": "processing", "error": None, "warnings": []}

    # Save uploads
    plan_path = job_dir / f"plan{Path(plan.filename).suffix}"
    async with aiofiles.open(plan_path, "wb") as f:
        await f.write(await plan.read())

    overlay_path = plan_path
    if overlay and overlay.filename:
        overlay_path = job_dir / f"overlay{Path(overlay.filename).suffix}"
        async with aiofiles.open(overlay_path, "wb") as f:
            await f.write(await overlay.read())

    mapping_path: Optional[Path] = None
    if mapping and mapping.filename:
        mapping_path = job_dir / mapping.filename
        async with aiofiles.open(mapping_path, "wb") as f:
            await f.write(await mapping.read())

    out_dir = job_dir / "output"

    # Run pipeline in background thread (OpenCV is sync)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_job,
        job_id, plan_path, overlay_path, mapping_path, out_dir, ocr_fallback,
    )

    return JSONResponse({"job_id": job_id, "status": "processing"})


def _run_job(
    job_id: str,
    plan_path: Path,
    overlay_path: Path,
    mapping_path: Optional[Path],
    out_dir: Path,
    use_ocr: bool,
):
    """Synchronous pipeline execution (called from thread executor)."""
    try:
        report: ValidationReport = run_pipeline(
            plan_path=plan_path,
            overlay_path=overlay_path,
            mapping_path=mapping_path,
            out_dir=out_dir,
            use_ocr=use_ocr,
            debug=False,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["warnings"] = report.warnings
        JOBS[job_id]["lots_found"] = report.total_regions_found
        JOBS[job_id]["lots_matched"] = report.lots_matched
        JOBS[job_id]["unmatched"] = report.unmatched_lot_ids
        JOBS[job_id]["suspicious"] = len(report.suspicious_polygons)
        logger.info("Job %s done: %d lots found.", job_id, report.total_regions_found)
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        logger.error("Job %s failed: %s", job_id, e)


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}", summary="Check job processing status")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(JOBS[job_id])


# ---------------------------------------------------------------------------
# GET /result/{job_id}
# ---------------------------------------------------------------------------

@app.get("/result/{job_id}", summary="Download result ZIP (lots.svg, lots_preview.svg, lots.json, validation_report.json)")
async def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job still processing")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Pipeline error"))

    out_dir = WORK_DIR / job_id / "output"
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Output files not found")

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.iterdir():
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=vectorized_{job_id[:8]}.zip"},
    )


# ---------------------------------------------------------------------------
# GET /result/{job_id}/{filename}  — single file download
# ---------------------------------------------------------------------------

@app.get("/result/{job_id}/{filename}", summary="Download a single result file")
async def get_result_file(job_id: str, filename: str):
    allowed = {"lots.svg", "lots_preview.svg", "lots.json", "validation_report.json"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown file: {filename}")

    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not ready")

    file_path = WORK_DIR / job_id / "output" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_types = {
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }
    mt = media_types.get(file_path.suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=mt, filename=filename)


# ---------------------------------------------------------------------------
# DELETE /job/{job_id} — cleanup
# ---------------------------------------------------------------------------

@app.delete("/job/{job_id}", summary="Delete job and its files")
async def delete_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = WORK_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    del JOBS[job_id]
    return {"deleted": job_id}
