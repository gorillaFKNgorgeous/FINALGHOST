import os
import traceback
from flask import Flask, request, abort, jsonify
from google.cloud import firestore, storage, secretmanager
from lib.audio_master import analyze_and_plan  # make sure this exists

app = Flask(__name__)

db      = firestore.Client()
gcs     = storage.Client()
secrets = secretmanager.SecretManagerServiceClient()

BUCKET = os.getenv("FILE_BUCKET")  # e.g. ghost-master-c5977-ghostmaster-bucket

def get_secret(name: str) -> str:
    proj = os.environ["GOOGLE_CLOUD_PROJECT"]
    path = f"projects/{proj}/secrets/{name}/versions/latest"
    return secrets.access_secret_version(name=path).payload.data.decode()

# Firestore transaction to mark job as started
@firestore.transactional
def mark_job_analyzing(tx, ref):
    snap = ref.get(transaction=tx)
    if not snap.exists or snap.get("status") != "pending":
        return False
    tx.update(ref, {
        "status": "analyzing",
        "progress": 5,
        "processingStartedAt": firestore.SERVER_TIMESTAMP,
    })
    return True

@app.route("/health")
def health():
    return "OK", 200

@app.route("/analyze-job", methods=["POST"])
def analyze_job():
    # Ensure invoked by Cloud Tasks (or remove this check if you call directly)
    if not request.headers.get("X-CloudTasks-TaskName"):
        abort(403)

    body   = request.get_json(silent=True) or {}
    job_id = body.get("jobId")
    if not job_id:
        abort(400, "jobId missing")

    job_ref = db.collection("jobs").document(job_id)
    tmp_in  = f"/tmp/in-{job_id}.wav"

    try:
        tx = db.transaction()
        if not mark_job_analyzing(tx, job_ref):
            return "Job not in pending state", 200

        # Download original file
        job_doc = job_ref.get().to_dict()
        gcs.bucket(BUCKET).blob(job_doc["filePath"]).download_to_filename(tmp_in)

        # Run AI analysis
        gemini_key = get_secret("GEMINI_API_KEY")
        analysis, settings, explanation = analyze_and_plan(tmp_in, gemini_key)

        # Update Firestore
        job_ref.update({
            "status": "analysis_complete",
            "progress": 100,
            "analysisResult": analysis,
            "suggestedSettings": settings,
            "aiExplanation": explanation
        })
        return "Analysis complete", 200

    except Exception as e:
        job_ref.update({"status": "error", "errorDetails": str(e)[:500]})
        traceback.print_exc()
        return "Error during analysis", 500

    finally:
        if os.path.exists(tmp_in):
            os.remove(tmp_in)
