# Resume Analyzer

Upload a PDF resume and ask questions about the candidate's skills, experience and fit for a role. Answers are generated with **Retrieval-Augmented Generation (RAG)**, so they come from the resume itself and not from the model's imagination.

After each answer, the resume panel highlights the exact chunks that were used, so you can check the answer against the source.

## How it works

```
PDF resume → text extraction → chunking → embeddings → FAISS vector database → retrieval → LLM → answer
```

| Step | What happens | Where in the code |
|---|---|---|
| Text extraction | `pypdf` reads the text from each page | `app/rag.py` → `extract_text` |
| Chunking | Text is split at paragraph and line boundaries into ~700 character chunks with a small overlap | `chunk_text` |
| Embeddings | Each chunk becomes a 768-number vector (Gemini `gemini-embedding-001`) | `embed` |
| Vector database | Vectors are stored in a FAISS index (cosine similarity) | `build_index` |
| Retrieval | The question is embedded and the 6 most similar chunks are found | `retrieve` |
| LLM | Gemini (`gemini-2.5-flash`) answers using only those chunks | `answer` |

## Project structure

```
resume-analyzer/
├── app/
│   ├── main.py          # FastAPI server (upload + ask endpoints)
│   └── rag.py           # The whole RAG pipeline
├── static/
│   ├── index.html       # Front end
│   ├── style.css
│   └── app.js
├── requirements.txt     # Python packages
├── render.yaml          # One-click deploy settings for Render
├── .env.example         # Template for your secret key
└── .gitignore           # Keeps secrets out of GitHub
```

---

# Beginner's guide: from zero to a live website

You need three free accounts: **Google** (for the AI key), **GitHub** (to store code) and **Render** (to host the site). Nothing here needs a credit card.

## Part 1. Get a free Gemini API key

1. Go to <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it. It is a long string of letters and numbers.
3. Keep it private. Treat it like a password. Never paste it into your code or upload it to GitHub.

## Part 2. Run it on your own computer first

1. Install **Python 3.11 or newer** from <https://www.python.org/downloads/>. On Windows, tick **"Add Python to PATH"** during install.
2. Unzip the project and open a terminal inside the `resume-analyzer` folder. On Windows, open the folder, click the address bar, type `cmd` and press Enter. On Mac, right-click the folder and choose *New Terminal at Folder*.
3. Run these commands one at a time:

   ```bash
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

   ```bash
   pip install -r requirements.txt
   ```
4. Create your secret settings file:
   - Windows: `copy .env.example .env`
   - Mac/Linux: `cp .env.example .env`

   Open `.env` in Notepad or any editor and replace `your-key-goes-here` with your real key. Save.
5. Start the app:

   ```bash
   uvicorn app.main:app --reload
   ```
6. Open <http://127.0.0.1:8000> in your browser, upload a resume PDF and ask a question. Press `Ctrl + C` in the terminal to stop.

## Part 2b. Put the code on GitHub

**Easiest way (no Git needed)**

1. Create an account at <https://github.com>.
2. Click the **+** at the top right, then **New repository**. Name it `resume-analyzer`, choose **Public**, and click **Create repository**.
3. On the next page click **uploading an existing file**.
4. Open your unzipped `resume-analyzer` folder and drag its **contents** into the browser: the `app` folder, the `static` folder, `requirements.txt`, `render.yaml`, `README.md`, `.env.example` and `.gitignore`.
5. **Do not upload `.env`** (your real key) or the `.venv` folder. If you followed Part 2, those two exist on your computer, so leave them out.
6. Click **Commit changes**.

**With Git (if you prefer the command line)**

```bash
git init
git add .
git commit -m "Resume Analyzer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/resume-analyzer.git
git push -u origin main
```
`.gitignore` already keeps `.env` and `.venv` out.

> GitHub Pages can't host this project, because it only serves static files and this app needs a Python server. That's why the next step uses Render.

## Part 3. Make it live on Render

1. Go to <https://render.com> and choose **Sign up with GitHub**.
2. Click **New +**, then **Web Service**, then connect your `resume-analyzer` repository.
3. Fill in the settings:
   - **Language:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Scroll to **Environment Variables** and add two:
   - `GEMINI_API_KEY` = your key from Part 1
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**. The first build takes about 3 to 6 minutes. When the log says the service is live, click the URL at the top, which looks like `https://resume-analyzer-xxxx.onrender.com`. That is your public link.

Whenever you change code and push to GitHub, Render redeploys automatically.

**About the free plan:** the site goes to sleep after about 15 minutes without visitors, and the next visit takes 30 to 60 seconds to wake it. Open the link a minute before a demo. Uploaded resumes are held in memory only, so they disappear when the server restarts.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Red banner: "no GEMINI_API_KEY set" | Add the key to `.env` (local) or Render's Environment Variables, then restart |
| "The AI service rejected the API key" | The key was copied wrongly or revoked. Create a new one |
| "The AI service is rate-limited" | You hit the free quota. Wait a minute and retry |
| "The configured AI model was not found" | Google renamed or retired a model. Set `GEN_MODEL` or `EMBED_MODEL` in your environment variables to a current model name from <https://ai.google.dev/gemini-api/docs/models> |
| "No readable text found" | The PDF is a scan or an image. Export a text-based PDF from Word or Google Docs |
| `pip` or `python` not recognised | Reinstall Python and tick "Add Python to PATH", or try `python3` / `pip3` |
| Render build fails on `faiss-cpu` | Check that `PYTHON_VERSION` is set to `3.11.9` |

## Privacy note

Resume text is sent to Google's Gemini API to create embeddings and answers. Free-tier API content may be used by Google to improve its products, so don't upload real personal resumes you don't have permission to share. Use sample resumes for demos.

## Settings you can change

Set these as environment variables: `GEN_MODEL`, `EMBED_MODEL`, `EMBED_DIM`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.
