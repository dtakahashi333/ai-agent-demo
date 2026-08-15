#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import json
from tqdm import tqdm

with open("squad_embeddings.json", "r") as f:
    data = json.load(f)

points = [
    PointStruct(
        id=item["id"],
        vector=item["vector"],
        payload=item["payload"],
    )
    for item in data
]

client = QdrantClient("localhost", port=6333)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


for batch in tqdm(chunks(points, 200)):
    client.upsert(
        collection_name="squad",
        points=batch,
    )
