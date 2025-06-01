from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import whisper
from app.llm_client import GeminiClient
from app.utils import build_prompt
import tempfile
import shutil
from app.qdrant_client import QdrantManager
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
qdrant = QdrantManager()
gemini_client = GeminiClient()
model = whisper.load_model("base")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/get-stories')
async def get_stories():
    try:
        result, _ = qdrant.get_all()
        stories = [
            {
                "id": point.id,
                "requirement": point.payload.get("requirement"),
                "user_story": point.payload.get("user_story"),
            }
            for point in result
        ]
        return stories
    except Exception as e:
        return

@app.post("/generate-story")
async def generate_story(requirement: str = Form(...)):
    embedding = gemini_client.get_embedding(requirement)
    similar_payloads = qdrant.query_similar_stories(embedding)
    related_stories = [p["user_story"] for p in similar_payloads]
    prompt = build_prompt(requirement, related_stories)
    user_story = gemini_client.generate_user_story(prompt)
    qdrant.store_user_story(embedding, requirement, user_story)
    return {"user_story": user_story}

@app.post("/audio-to-story")
async def audio_to_story(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Unsupported audio type")

    try:
        # Save uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        file.file.close()

        # Transcribe with whisper
        result = model.transcribe(tmp_path)
        transcript = result["text"]

        if not transcript:
            raise HTTPException(status_code=400, detail="Audio transcription failed or empty")

        # Build prompt & generate story with Gemini
        prompt = build_prompt(transcript, "")
        embedding = gemini_client.get_embedding(transcript)
        user_story = gemini_client.generate_user_story(prompt)
        qdrant.store_user_story(embedding, transcript, user_story)

        return {
            "transcript": transcript,
            "user_story": user_story
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")