# Deployment runbook

## Local API + UI

1. Train a real model artifact with your reviewed dataset.
2. Start API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
3. In a second terminal: `streamlit run ui/app.py`.
4. Test `GET /health`, invalid input rejection, and `POST /predict` with a known sample.

The API currently serves the TF-IDF + Logistic Regression baseline. For a hidden-state probe, implement the same observer-model feature extraction in the inference path before switching the app to it. Never route hidden-state model artifacts through the text-only baseline API.

## Docker API

```bash
docker build -t your-dockerhub-name/ragguard-api:1.0 .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" your-dockerhub-name/ragguard-api:1.0
```

After verifying locally, log in to Docker Hub, push the image, and deploy it to a platform you control. Do not put credentials in the image or repository.

## Hugging Face Spaces (Streamlit UI)

1. Create a new Space and select **Streamlit** SDK.
2. Upload the repository files needed by the UI, `requirements.txt`, and a root-level `app.py` containing `from ui.app import *` (or configure the Space entrypoint to `ui/app.py` if supported).
3. Add Space Secrets/Variables: `RAGGUARD_API_URL=https://<your-api-host>`.
4. Deploy the API separately; the UI calls the API over HTTPS. Do not include large private datasets or credentials in the Space repository.
5. Test `/health` and a sample prediction after deployment; record the real public URL only after it works.

The project archive does not claim cloud deployment has already happened; deployment requires your account and target service.
