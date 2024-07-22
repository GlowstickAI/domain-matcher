from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ColumnName(str, Enum):
    text = "text"
    labels = "labels"
    split = "split"  # type: ignore
    topic_name = "topic_name"
    topic_confidence = "topic_confidence"
    topic_of_interest = "topic_of_interest"
    in_domain = "in_domain"
    topic_id = "topic_id"
    adjacent_topics = "adjacent_topics"
    idx = "idx"
    embedding = "embedding"
    embedding_index = "embedding_index"


class TopicOfInterest(BaseModel):
    title: str
    keywords: list[str] = []
    description: Optional[str] = None
