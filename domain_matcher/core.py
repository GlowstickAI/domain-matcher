import os
from functools import partial
from typing import List, Optional, Union

import structlog
from datasets import ClassLabel, Dataset
from keybert import KeyBERT

from domain_matcher.components.pipeline import (
    GSPipeline,
    PreprocessingComponent,
    SentenceEmbeddingExtractionComponent,
)
from domain_matcher.components.topic_modeling import (
    TopicModelingComponent,
    TopicModelingPredictionComponent,
)
from domain_matcher.config import DMConfig
from domain_matcher.types import ColumnName, TopicOfInterest

log = structlog.get_logger()


class DomainMatcher:
    def __init__(self, config: DMConfig, domains: Optional[List[TopicOfInterest]] = None):
        self.config = config
        self.domains = domains

    def fit(self, train_dataset: Dataset):
        os.makedirs(self.config.artifact_path, exist_ok=True)
        self.domains = self.domains or create_domains(dataset=train_dataset, config=self.config)
        train_pipeline = GSPipeline(
            [
                PreprocessingComponent,
                SentenceEmbeddingExtractionComponent,
                partial(TopicModelingComponent, topic_of_interest=self.domains),
            ],
            self.config,
        )
        return train_pipeline(train_dataset)

    def transform(self, dataset: Union[str, List[str], Dataset]):
        if self.domains is None:
            raise ValueError("Please fit the model first.")
        if isinstance(dataset, str):
            dataset = [dataset]
        if isinstance(dataset, List):
            dataset = Dataset.from_dict({self.config.text_column: dataset})
        test_pipeline = GSPipeline(
            [
                PreprocessingComponent,
                SentenceEmbeddingExtractionComponent,
                TopicModelingPredictionComponent,
            ],
            self.config,
        )
        return test_pipeline(dataset)

    def filter_ood(self, dataset):
        return self.transform(dataset).filter(
            lambda in_domain: in_domain, input_columns=ColumnName.in_domain
        )


def get_keywords(dataset: Dataset, config: DMConfig) -> List[str]:
    label_feature: ClassLabel = dataset.features[config.label_column]
    keywords: List[tuple[str, float]] = []
    for class_id in range(label_feature.num_classes):
        if label_feature.int2str(class_id) == config.oos_class:
            continue
        kw_model = KeyBERT()
        _ds = dataset.filter(lambda u: u[config.label_column] == class_id)
        kws = kw_model.extract_keywords(
            " ".join(_ds["text"]),
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=5,
        )
        keywords += kws
    # Get top 100 keywords
    return [k for k, _ in sorted(keywords, key=lambda i: i[1], reverse=True)][:100]


def create_domains(
    dataset: Dataset, config: DMConfig, domain_name="Domain"
) -> List[TopicOfInterest]:
    """
    To automatically create domain, we merge class names and keywords extracted from the documents.

    Args:
        dataset: Train dataset.
        config: Project config
        domain_name: Name of the domain for documentation

    Returns:

    """
    log.info("Automatically detecting domain")
    label_feature: ClassLabel = dataset.features[config.label_column]
    keywords = get_keywords(dataset, config)
    return [
        TopicOfInterest(
            title=domain_name,
            keywords=[c.replace("_", " ") for c in label_feature.names if c != config.oos_class],
        ),
        TopicOfInterest(title=f"{domain_name}_KW", keywords=keywords),
    ]
