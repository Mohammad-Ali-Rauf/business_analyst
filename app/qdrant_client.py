from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.models import Distance
import uuid

COLLECTION_NAME = "user_stories"

class QdrantManager:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.ensure_collection()

    def ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]  # ✅ Extract names

        if COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print("🆕 Collection created:", COLLECTION_NAME)
        else:
            print("✅ Collection already exists:", COLLECTION_NAME)

    def store_user_story(self, embedding: list[float], requirement: str, user_story: str):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "requirement": requirement,
                "user_story": user_story
            }
        )
        self.client.upsert(collection_name=COLLECTION_NAME, points=[point])
        print("📥 Saved to Qdrant!")

    def query_similar_stories(self, embedding: list[float], top_k=3) -> list[dict]:
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k
        )

        return [hit.payload for hit in results if hit.score > 0.75]