import os
import pickle
from collections import defaultdict
from pprint import pprint
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import structlog
from bertopic import BERTopic
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from domain_matcher.components.hf_manager import HFPipelineManager
from domain_matcher.components.pipeline import GSComponent
from domain_matcher.config import DMConfig
from domain_matcher.types import ColumnName, TopicOfInterest

log = structlog.get_logger()


class TopicModelingComponent(GSComponent):
    def __init__(self, config: DMConfig, topic_of_interest):
        super().__init__(config)
        self.topic_of_interest = topic_of_interest

    def _get_topic_model_path(self):
        return os.path.join(self.config.artifact_path, "most_recent_topic_model.berttopic")

    def _get_topic_mapping_path(self):
        return os.path.join(
            self.config.artifact_path,
            "most_recent_topic_model_mapping.pkl",
        )

    def __call__(self, dataset: Dataset):
        dataset, self.topic_model = get_topics(
            dataset=dataset,
            min_topic_size=1 / 200,
            model_name=self.config.embedding_model,
        )
        self.toi_mapping = get_toi(
            self.topic_of_interest, self.topic_model, self.config, discarded={-1, 0}
        )
        log.info("In domain topics:")
        pprint(
            {
                f"Domain {d.title}": [
                    get_title(self.topic_model, k)
                    for k, tois in self.toi_mapping.items()
                    if d in tois
                ]
                for d in self.topic_of_interest
            }
        )

        self.topic_model.save(
            self._get_topic_model_path(),
            save_embedding_model=False,
        )
        pickle.dump(
            self.toi_mapping,
            open(
                self._get_topic_mapping_path(),
                "wb",
            ),
        )

        dataset = dataset.map(
            lambda u: {
                ColumnName.topic_of_interest: list(
                    map(
                        lambda u: u.dict(),  # type: ignore
                        self.toi_mapping[u[ColumnName.topic_id]],
                    )
                )
            },
            desc="Assigning TOIs",
        )
        return dataset


class TopicModelingPredictionComponent(TopicModelingComponent):
    def __init__(self, config: DMConfig):
        super().__init__(config=config, topic_of_interest=None)
        # Setup, takes a while
        self.topic_model: BERTopic = BERTopic.load(self._get_topic_model_path())
        self.toi_mapping = pickle.load(open(self._get_topic_mapping_path(), "rb"))
        topic_mapping = dict(
            self.topic_model.get_topic_info()[["Topic", "Name"]].itertuples(index=False, name=None)
        )
        self.topic_mapping = {
            topic_id: get_title(self.topic_model, topic_id) for topic_id in topic_mapping.keys()
        }

    def __call__(self, dataset):
        dataset = topic_model_predict(dataset, self.topic_model, self.topic_mapping)
        dataset = dataset.map(
            lambda u: {
                ColumnName.topic_of_interest: list(
                    map(lambda u: u.dict(), self.toi_mapping[u[ColumnName.topic_id]])
                ),
                ColumnName.in_domain: len(self.toi_mapping[u[ColumnName.topic_id]]) > 0,
            },
            desc="Assigning TOIs",
        )
        return dataset


def get_topics(
    dataset: Dataset,
    n_try=5,
    min_topic_size: Union[float, int] = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[Dataset, BERTopic]:
    """
    From a dataset, get the topics and add them as columns.

    Args:
        dataset: Dataset with column `text`
        n_try: Number of tries to train the topic model.
        min_topic_size: minimal topic size
        model_name: SBERT model to use.

    Returns:
        (1) New dataset with columns `topic_name` and `topic_id`.
        (2) BERTopic model (useful for viz)
    """
    docs = dataset[ColumnName.text]

    # TODO the embedding can be in the dataset already.
    sentence_model = SentenceTransformer(model_name)
    embeddings = sentence_model.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    if min_topic_size < 1:
        min_topic_size = int(min_topic_size * len(docs))

    # Train BERTopic
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
    for _ in range(n_try):
        topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=min_topic_size)
        topics, probs = topic_model.fit_transform(docs, embeddings)
        topic_mapping = dict(
            topic_model.get_topic_info()[["Topic", "Name"]].itertuples(index=False, name=None)
        )
        if len(topic_mapping) > 3:
            break
    else:
        print("Unable to get more than 3 topics, expect bad results.")

    dataset = dataset.map(
        lambda u, i: {
            ColumnName.topic_name: topic_mapping[topics[i]],
            ColumnName.topic_id: topics[i],
        },
        with_indices=True,
    )
    return dataset, topic_model


def topic_model_predict(dataset: Dataset, topic_model: BERTopic, topic_mapping):
    # We enable `calculate_probabilities` for this one.
    topic_model.calculate_probabilities = True
    topics, probs = topic_model.transform(
        dataset[ColumnName.text], np.stack(dataset[ColumnName.embedding])
    )
    topic_model.calculate_probabilities = False
    dataset = dataset.map(
        lambda u, i: {
            ColumnName.topic_name: topic_mapping[topics[i]],
            ColumnName.topic_confidence: probs[i].max(),
            ColumnName.topic_id: topics[i],
            ColumnName.adjacent_topics: (
                []
                if topics[i] == -1
                else [
                    topic_mapping[t_id]
                    for t_id in np.where(np.isclose(probs[i], probs[i].max(), rtol=0.02))[0]
                ]
            ),
        },
        with_indices=True,
    )
    return dataset


def get_title(topic_model, topic_id):
    important_words = [(w, p) for w, p in topic_model.get_topic(topic_id) if p > 0.02]
    if not important_words:
        # Return first word
        return topic_model.get_topic(topic_id)[0][0].capitalize()
    words, probs = zip(*important_words)

    if wi := [w for w in words if len(w.split()) > 1]:
        # Has two words should make sense.
        return wi[0].capitalize()

    if len(words) == 1:
        return words[0].capitalize()

    first = words[0]
    new_words = [wi for wi in words[1:] if not (wi in first or first in wi)]
    return " ".join(set([first] + new_words)).capitalize()


def get_toi(
    topics_of_interest: List[TopicOfInterest],
    topic_model: BERTopic,
    config: DMConfig,
    discarded: Optional[Iterable[int]] = None,
) -> Dict[int, List[TopicOfInterest]]:
    """
    Assign topics to domains.
    Args:
        topics_of_interest: List of domain to match.
        topic_model: Topic Model fitted on all meetings.
        config: Application configuration
        discarded: List of topics to discard (Default to -1,0)

    Returns:

    """
    discarded = discarded or [-1, 0]
    embedder = HFPipelineManager().get_pipeline(
        "sentence-embedding", model_name=config.embedding_model, device="cpu"
    )
    results = defaultdict(list)
    for toi in topics_of_interest:
        if not toi.keywords:
            # No keywords
            continue
        # Get embedding from product descriptions
        embeddings = embedder.encode(toi.keywords, normalize_embeddings=True)
        # Assign topics to each description
        topics, confidence = topic_model.transform(toi.keywords, embeddings=embeddings)
        # Group products per topics (A topic can belong to multiple domain).
        for t, c in set(zip(topics, confidence)):
            if t not in discarded:
                results[t].append(toi)
    return results
