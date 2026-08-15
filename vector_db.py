#!/usr/bin/env python3
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import json

dataset = load_dataset("squad")
train = dataset["train"]

print(len(train))

contexts = {}

for row in train:
    context = row["context"]

    if context not in contexts:
        contexts[context] = {"title": row["title"], "answers": []}

    contexts[context]["answers"].extend(row["answers"]["text"])

print(len(contexts))

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = QdrantClient("localhost", port=6333)

# client.recreate_collection(
#     collection_name="squad",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )
if not client.collection_exists("squad"):
    client.create_collection(
        collection_name="squad",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

# points = []

# for idx, (context, title) in enumerate(tqdm(contexts.items())):
#     embedding = model.encode(context).tolist()

#     points.append(
#         PointStruct(
#             id=idx,
#             vector=embedding,
#             payload={
#                 "title": title,
#                 "text": context,
#             },
#         )
#     )

texts = list(contexts.keys())
titles = list(contexts.values())

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
points = []

for idx, (text, title, embedding) in enumerate(tqdm(zip(texts, titles, embeddings))):
    points.append(
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "title": title,
                "text": text,
            },
        )
    )


data = []

for p in points:
    data.append(
        {
            "id": p.id,
            "vector": p.vector,
            "payload": p.payload,
        }
    )

with open("squad_embeddings.json", "w") as f:
    json.dump(data, f)

# def chunks(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]


# for batch in chunks(points, 200):
#     client.upsert(
#         collection_name="squad",
#         points=batch,
#     )

# query = "Who created Python?"

# query_vector = model.encode(query).tolist()

# results = client.query_points(
#     collection_name="squad",
#     query=query_vector,
#     limit=5,
# ).points

# for result in results:
#     print(result.score)
#     print(result.payload["title"])
#     print(result.payload["text"][:200])
#     print("-" * 80)
