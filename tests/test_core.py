import os

import pytest
from datasets import load_dataset

from domain_matcher.config import DMConfig
from domain_matcher.core import DomainMatcher
from domain_matcher.types import ColumnName


@pytest.fixture
def a_dataset():
    return (
        load_dataset("GlowstickAI/banking-clinc-oos", "small")["test"].shuffle().select(range(1000))
    )


def test_happy_path(a_dataset, tmp_path):
    config = DMConfig(
        text_column="text", label_column="intent", oos_class="oos", artifact_path=str(tmp_path)
    )
    dmatcher = DomainMatcher(config)
    new_ds = dmatcher.fit(a_dataset)

    assert len(dmatcher.domains) == 2
    assert {
        ColumnName.topic_of_interest,
        ColumnName.topic_name,
        ColumnName.topic_id,
        ColumnName.embedding,
    }.issubset(new_ds.column_names)
    assert {"most_recent_topic_model.berttopic", "most_recent_topic_model_mapping.pkl"}.issubset(
        os.listdir(tmp_path)
    )
