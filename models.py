#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Document:
    id: str
    text: str
    title: str | None = None


@dataclass
class SearchResult:
    id: str
    score: float
    document: Document
