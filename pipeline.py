#!/usr/bin/env python3
import os
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from tqdm import tqdm

load_dotenv()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

vecdb_client = QdrantClient("localhost", port=6333)

llm_client = OpenAI(
    # API keys vary by region. To get an API key, visit: https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # The following base_url is for the Singapore region. If you use a model in the US East 1 (Virginia) region, change the base_url to https://dashscope-us.aliyuncs.com/compatible-mode/v1.
    # If you use a model in the China (Beijing) region, change the base_url to https://dashscope.aliyuncs.com/compatible-mode/v1.
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    base_url="https://ws-a95hgp91msvbk42j.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)

queries = [
    "Where is Imperial College London located?",
    "Who was Apollo?",
    "What is Unicode?",
    "When was the Eiffel Tower completed?",
    "Who founded the University of Oxford?",
]

# Retriever
for i in tqdm(range(len(queries))):
    query = "Represent this sentence for searching relevant passages: " + queries[i]

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = vecdb_client.query_points(
        collection_name="squad",
        query=query_vector,
        limit=20,
    ).points

    contexts = []

    # Rerank the contexts
    pairs = [(queries[i], result.payload["text"]) for result in results]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)

    retrieved_context = "\n\n".join(context for context, _ in ranked[:5])

    # Prompt template
    prompt = f"""
Answer the question using only the provided context.

Context:
{retrieved_context}

Question:
{query}

Answer:
"""

    # # print(prompt)

    # completion = llm_client.chat.completions.create(
    #     # This example uses qwen-plus. You can replace it with another model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    #     model="qwen-plus",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     # extra_body={"enable_thinking": False},
    # )

    # print(completion.model_dump_json())

    # # Save full response as JSON
    # file_num = i + 1
    # with open(f"responses/response-{file_num}.json", "w", encoding="utf-8") as f:
    #     f.write(completion.model_dump_json(indent=2))

    # print(f"Response saved to response-{file_num}.json")

    # completion = llm_client.chat.completions.create(
    #     # This example uses qwen-plus. You can replace it with another model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    #     model="qwen-plus",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Who are you?"},
    #     ],
    #     # extra_body={"enable_thinking": False},
    # )
    # print(completion.model_dump_json())
