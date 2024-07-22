from functools import partial
from typing import List, Optional, Tuple, Type, Union

import structlog
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from domain_matcher.components.hf_manager import HFPipelineManager
from domain_matcher.config import DMConfig
from domain_matcher.preprocessing import decontracted, remove_repetition, remove_uhm
from domain_matcher.types import ColumnName
from domain_matcher.utils import md5_hash

log = structlog.get_logger("domain_matcher")


class GSComponent:
    def __init__(self, config: DMConfig):
        self.config = config

    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    def _get_caching_key(self, *args, model_args: Optional[List[str]] = None) -> Tuple[str, str]:
        # Get caching key for this component and its args (columns name, time of day, etc)
        return (
            f"{'-'.join(map(md5_hash, args))}",
            f"{'-'.join([self.__class__.__name__, *map(md5_hash, model_args or [])])}",
        )


class GSPipeline:
    def __init__(
        self,
        callables: List[Union[Type[GSComponent], partial[GSComponent]]],
        config: DMConfig,
    ):
        self.callables = [Component(config=config) for Component in callables]
        self.config = config

    def __call__(self, dataset: Union[Dataset, List[Dataset]]) -> Union[Dataset, List[Dataset]]:
        for clb in self.callables:
            if isinstance(dataset, List) and isinstance(clb, MergeDatasetsComponent):
                dataset = clb(dataset)
            elif isinstance(dataset, Dataset) and isinstance(clb, MergeDatasetsComponent):
                # Already Merged
                pass
            elif isinstance(dataset, List) and not isinstance(clb, MergeDatasetsComponent):
                dataset = [clb(ds) for ds in tqdm(dataset, desc=f"Processing {clb}") if len(ds) > 0]
            elif isinstance(dataset, Dataset) and not isinstance(clb, MergeDatasetsComponent):
                dataset = clb(dataset)
            else:
                raise ValueError(f"Can't run pipeline on f{dataset} and f{clb}")

        return dataset


class PreprocessingComponent(GSComponent):
    def __call__(self, dataset: Dataset):
        """Default preprocessing will extract and clean sentences.

        Args:
            dataset: Initial dataset

        Returns:
            Cleaned up dataset
        """
        return dataset.map(
            lambda u: {"text": decontracted(remove_repetition(remove_uhm(u[ColumnName.text])))},
            desc="Preprocessing",
        )


class MergeDatasetsComponent:
    def __init__(self, config: DMConfig):
        self.config = config

    def __call__(self, datasets: List[Dataset]) -> Dataset:
        dataset: Dataset = concatenate_datasets(datasets, axis=0)
        return dataset


class SentenceEmbeddingExtractionComponent(GSComponent):
    def __call__(self, dataset: Dataset):
        """Gather features used for later processing: embedding.

        Args:
            dataset: preprocessed dataset.

        Returns:
            Augmented dataset with embedding.
        """
        sentence_model = HFPipelineManager.get_pipeline(
            "sentence-embedding", self.config.embedding_model
        )
        if ColumnName.embedding in dataset.column_names:
            log.warn("Removing duplicate columns", cols=ColumnName.embedding)
            dataset = dataset.remove_columns([ColumnName.embedding])
        dataset = dataset.add_column(
            ColumnName.embedding,
            sentence_model.encode(
                dataset[ColumnName.text],
                show_progress_bar=True,
                batch_size=256,
                normalize_embeddings=True,
            ).tolist(),  # type: ignore
        )
        return dataset
