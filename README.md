# PramaanSetu

An independent hackathon prototype for a clearer certificate-request journey. It uses only mock data, has no login or backend, and never submits information to a government system.

## Run locally

```powershell
Copy-Item .env.example .env
# Add your HF_TOKEN to .env
py -m pip install -r requirements.txt
py -m uvicorn server:app --reload --port 8000
```

In a second terminal:

```powershell
npm install
npm run dev
```