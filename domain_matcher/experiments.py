import os
import pickle
from functools import partial
from pprint import pprint
from typing import Any, Dict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score, precision_score

from domain_matcher.components.pipeline import (
    GSPipeline,
    PreprocessingComponent,
    SentenceEmbeddingExtractionComponent,
)
from domain_matcher.components.topic_modeling import (
    TopicModelingComponent,
    TopicModelingPredictionComponent,
)
from domain_matcher.components.training import (
    PredictModelComponent,
    TrainModelComponent,
)
from domain_matcher.config import DMConfig, ExperimentConfig
from domain_matcher.core import DomainMatcher, create_domains
from domain_matcher.types import ColumnName

pjoin = os.path.join


def analyze_results(dataset: Dataset, config: DMConfig) -> Dict[str, Any]:
    """
    Compare prediction of a model with and without domain matching.


    WHAT we have:
      - ColumnName.topic_of_interest
      - 'prediction'

    Args:
        dataset:
        config:

    Returns:
        Dict[str, float], a list of metrics.

    """

    def postprocess(u):
        preds = u["prediction"]
        is_in_domain = len(u[ColumnName.topic_of_interest]) > 0
        if is_in_domain and not config.allow_oos_pred:
            # Select the most likely class that is not oos
            return [p for p in preds if p["label"] != config.oos_class][0]["label"]
        elif is_in_domain and config.allow_oos_pred:
            # Select the most likely class
            return preds[0]["label"]
        # Not in domain, skip.
        return config.oos_class

    lbl_feature = dataset.features[config.label_column]
    oos_id = lbl_feature.str2int(config.oos_class)
    y_true = np.array([i for i in dataset[config.label_column]])
    y_pred = np.array([lbl_feature.str2int(u[0]["label"]) for u in dataset["prediction"]])
    y_pred_post = np.array([lbl_feature.str2int(postprocess(u)) for u in dataset])

    in_domain_true = [lbl != oos_id for lbl in dataset[config.label_column]]
    in_domain_pred = [len(lbl) > 0 for lbl in dataset[ColumnName.topic_of_interest]]
    return {
        "correct": ((y_true == y_pred) & (y_pred != oos_id)).sum() / (y_true != oos_id).sum(),
        "correct_post": ((y_true == y_pred_post) & (y_pred_post != oos_id)).sum()
        / (y_true != oos_id).sum(),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "f1_post": f1_score(y_true, y_pred_post, average="macro"),
        "precision_post": precision_score(y_true, y_pred_post, average="macro"),
        "domain_matching_f1_score": f1_score(in_domain_true, in_domain_pred),
        "domain_matching_precision": precision_score(
            in_domain_true, in_domain_pred, average="macro"
        ),
    }


def run_experiments(
    dataset_dict: DatasetDict,
    config: DMConfig,
    train_on_oos: bool,
    domain="Domain",
) -> Dict[str, Any]:
    """
    Run Domain Matching Experiment!

    Args:
        dataset_dict: Dataset to process
        config: Domain Matching config
        domain: Name of domain for documentation.

    Returns:
        Useful metrics hopefully
    """
    os.makedirs(config.artifact_path, exist_ok=True)
    tois = create_domains(dataset=dataset_dict["train"], config=config, domain_name=domain)
    domain_matcher = DomainMatcher(config=config, domains=tois)
    pickle.dump(domain_matcher, open(pjoin(config.artifact_path, "dm.pkl"), "wb"))
    # Train
    train_pipeline = GSPipeline(
        [
            PreprocessingComponent,
            SentenceEmbeddingExtractionComponent,
            partial(TopicModelingComponent, topic_of_interest=tois),
            partial(
                TrainModelComponent,
                hparams=ExperimentConfig(
                    dataset_path="",
                    pretrained_pipeline="distilbert-base-uncased",
                    text_column=config.text_column,
                    label_column=config.label_column,
                    train_on_oos=train_on_oos,
                    num_train_epochs=5,
                    weight_decay=0,
                    lr_scheduler_type="linear",
                    freeze_backbone=True,
                    learning_rate=1e-5,
                ),
            ),
        ],
        config,
    )

    dataset_dict["train"] = train_pipeline(dataset_dict["train"])

    # Test
    test_pipeline = GSPipeline(
        [
            PreprocessingComponent,
            SentenceEmbeddingExtractionComponent,
            TopicModelingPredictionComponent,
            partial(
                PredictModelComponent,
                hparams=ExperimentConfig(
                    dataset_path="",
                    pretrained_pipeline="distilbert-base-uncased",
                    text_column=config.text_column,
                    label_column=config.label_column,
                    train_on_oos=train_on_oos,
                    num_train_epochs=1,
                    weight_decay=0,
                    lr_scheduler_type="linear",
                    freeze_backbone=True,
                    learning_rate=1e-5,
                ),
            ),
        ],
        config,
    )
    dataset_dict["test"] = test_pipeline(dataset_dict["test"])

    return analyze_results(dataset_dict["test"], config)


def experiment_script(
    ds_path: str,
    ds_name: str,
    text_column: str,
    label_column: str,
    oos_class: str,
    train_on_oos: bool,
    allow_oos_pred: bool,
):
    ds = load_dataset(ds_path, ds_name, token=True)
    cfg = DMConfig(
        artifact_path=".cache",
        text_column=text_column,
        label_column=label_column,
        oos_class=oos_class,
        allow_oos_pred=allow_oos_pred,
    )
    result = run_experiments(
        ds,
        config=cfg,
        train_on_oos=train_on_oos,
    )
    pprint(result)
    return result
