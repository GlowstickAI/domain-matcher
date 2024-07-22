import re
from copy import copy
from typing import List

import pandas as pd
from datasets import Dataset

from domain_matcher.types import ColumnName

FILTERED_TOKENS = {"okay", "like", "know", "yeah", "think", "thing"}


def split_on_sentences(s) -> List[str]:
    return [si.strip() for si in re.split(r"[?\.…!]", s) if len(si.strip()) > 0]


def explode_dataset_in_sentences(dataset: Dataset, min_sentence_length=0) -> Dataset:
    """
    Explode documents inside the Dataset into many items for
     each sentence, others fields are copied.

    Args:
        dataset: Dataset with column `text`
        min_sentence_length: Filter sentences with less word than this value.

    Returns:
        New extended dataset.
    """

    def get_items(dataset):
        for item in dataset:
            sentences = split_on_sentences(item[ColumnName.text])
            for si in sentences:
                if len(si.split()) > min_sentence_length:
                    item.update({ColumnName.text: si})
                    yield copy(item)

    return Dataset.from_pandas(pd.DataFrame.from_records(list(get_items(dataset))))


def remove_repetition(sentence: str) -> str:
    """Remove repetition common in transcript.

    Examples:
        "hello everyone this is... this is friday and we will..."
        > "hello everyone this is friday and we will..."

    Args:
        sentence: one string to remove repetition from.

    Returns:
        sentence without repetition
    """

    def _remove_duplicate(match_obj):
        if match_obj.group(1) is not None:
            return match_obj.group(1)

    return re.sub(r"\b([a-zA-Z\s]+)[\s….!\W]+\1\b", _remove_duplicate, sentence)


def remove_uhm(sentence: str) -> str:
    """Remove Um, Uh, uH from the sentence."""
    pattern = r",*\s*(U|u)(m|h)+\s*,\s*"
    return re.sub(pattern, " ", sentence)


def decontracted(sentence):
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"he\'s", "he is", sentence)
    sentence = re.sub(r"He\'s", "He is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence
