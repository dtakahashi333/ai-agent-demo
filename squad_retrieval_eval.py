#!/usr/bin/env python3
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import random
from tqdm import tqdm
import re
import argparse
import string

parser = argparse.ArgumentParser(description="Example script")

parser.add_argument("--ranking-limit", default=5, type=int, help="Ranking limit")

args = parser.parse_args()

print(f"Ranking limit: {args.ranking_limit}")

dataset = load_dataset("squad")
train = dataset["train"]

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = QdrantClient("localhost", port=6333)

queries, results = [], []

indices = random.sample(range(len(train)), 200)

for index in tqdm(indices):
    row = train[index]

    queries.append(
        {
            "question": row["question"],
            "title": row["title"],
            "answers": row["answers"]["text"],
        }
    )
    # print("Question:", row["question"])
    # print("Title:", row["title"])

    # query = "Where is Imperial College London located?"
    # query = "What is Unicode?"
    # query = "Who was Apollo?"
    query = (
        "Represent this sentence for searching relevant passages: "
        + queries[-1]["question"]
    )

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results.append(
        client.query_points(
            collection_name="squad",
            query=query_vector,
            limit=args.ranking_limit,
        ).points
    )


def normalize_answer(text):
    text = text.lower()

    # remove punctuation
    text = "".join(c for c in text if c not in string.punctuation)

    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # normalize whitespace
    text = " ".join(text.split())

    return text


def contains_answer(text, answers):
    text = normalize_answer(text)

    for answer in answers:
        answer = normalize_answer(answer)

        if answer in text:
            return True


hits = 0

for query, result_list in zip(queries, results):
    answer_hit = False

    for result in result_list:
        if contains_answer(result.payload["text"], query["answers"]):
            answer_hit = True
            break

    if answer_hit:
        hits += 1

    print(query["question"])
    print("Answer found:", answer_hit)

    for result in result_list:
        print(result.score, result.payload["title"])
        print(result.payload["text"][:200])
        print("-" * 80)

print(f"Recall@{args.ranking_limit}: {hits / len(queries)}")
