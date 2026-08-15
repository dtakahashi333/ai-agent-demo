#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Config:
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    collection_name: str = "squad"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    batch_size: int = 64
