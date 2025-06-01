import os
import google.genai as genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

class GeminiClient:
    def generate_user_story(self, prompt: str) -> str:
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return response.text
        except Exception as e:
            print(f"[Gemini Error] Story generation failed: {e}")
            return ""
    def get_embedding(self, text: str) -> list[float]:
        try:
            response = client.models.embed_content(
                model="models/embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )

            print("🔍 RAW response from Gemini:", response)

            # ✅ Proper extraction
            embedding = response.embeddings[0].values
            print("🧬 Extracted embedding:", embedding[:5])  # first 5 values

            if not embedding or len(embedding) != 768:
                raise ValueError("❌ Gemini embedding failed: Invalid or empty vector")

            return embedding

        except Exception as e:
            print(f"[Gemini Error] Embedding failed: {e}")
        return []



