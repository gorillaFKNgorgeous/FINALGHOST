# backend/processor/masterer_main.py
import os, traceback, datetime
from flask import Flask, request, abort
from google.cloud import firestore, storage, secretmanager
from lib.audio_master import apply_mastering_chain  # Audio processing logic

app = Flask(__name__)
db = firestore.Client()
gcs = storage.Client()
secrets = secretmanager.SecretManagerServiceClient()

BUCKET = os.environ.get("FILE_BUCKET")
SIGNED_TTL_SECONDS = 86400  # Signed URL expiration (24 hours)

def get_secret(name: str) -> str:
    proj = os.environ["GOOGLE_CLOUD_PROJECT"]
    secret_path = f"projects/{proj}/secrets/{name}/versions/latest"
    return secrets.access_secret_version(name=secret_path).payload.data.decode()

def generate_signed_url(blob_path: str) -> str:
    # Generate a signed URL for the given GCS blob
    bucket = gcs.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    return blob.generate_signed_url(expiration=datetime.timedelta(seconds=SIGNED_TTL_SECONDS))

@firestore.transactional
def start_mastering_transaction(transaction, job_ref):
    snapshot = job_ref.get(transaction=transaction)
    if not snapshot.exists or snapshot.get("status") != "analysis_complete":
        return False  # Job is not ready (either missing or already processed)
    transaction.update(job_ref, {"status": "processing", "progress": 0})
    return True

@app.route("/health")
def health_check():
    return "OK", 200

@app.route("/master-job", methods=["POST"])
def master_job_endpoint():
    if not request.headers.get("X-Cloudtasks-Taskname"):
        abort(403)
    body = request.get_json(silent=True) or {}
    job_id = body.get("jobId")
    final_settings = body.get("finalSettings")
    if not job_id or final_settings is None:
        abort(400, "jobId or finalSettings missing")

    job_ref = db.collection("jobs").document(job_id)
    local_in_path = f"/tmp/in-{job_id}"
    local_out_path = f"/tmp/out-{job_id}.wav"

    try:
        # Transition job from analysis_complete to processing
        transaction = db.transaction()
        if not start_mastering_transaction(transaction, job_ref):
            return ("Job not ready for mastering.", 200)

        # Fetch the job and download the input audio
        job_doc = job_ref.get().to_dict()
        in_blob = gcs.bucket(BUCKET).blob(job_doc["filePath"])
        in_blob.download_to_filename(local_in_path)
        job_ref.update({"progress": 10})

        # Apply the mastering chain with the final settings (user tweaks applied)
        apply_mastering_chain(local_in_path, local_out_path, final_settings)
        job_ref.update({"progress": 90})

        # Upload mastered output to GCS and generate a signed URL
        result_path = f"user-results/{job_doc['userId']}/{job_id}.wav"
        out_blob = gcs.bucket(BUCKET).blob(result_path)
        out_blob.upload_from_filename(local_out_path)
        url = generate_signed_url(result_path)

        job_ref.update({"status": "complete", "progress": 100, "resultUrl": url})
        return ("Mastering complete", 200)
    except Exception as e:
        job_ref.update({"status": "error", "errorDetails": str(e)[:500]})
        traceback.print_exc()
        return ("Error during mastering", 500)
    finally:
        if os.path.exists(local_in_path):
            os.remove(local_in_path)
        if os.path.exists(local_out_path):
            os.remove(local_out_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))