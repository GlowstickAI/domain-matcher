from multiprocessing import Lock
from typing import Any, Dict, Literal, Optional, Union, overload

import structlog
import torch.cuda
from sentence_transformers import SentenceTransformer
from transformers import Pipeline, pipeline

log = structlog.get_logger(__name__)


class HFPipelineManager:
    """
    This class enables us to have a single pipeline loaded at the same time.
    We only have 1 gpu per server, this effectively act as a LRU cache.
    """

    _curr_pipeline: Optional[Union[Pipeline, SentenceTransformer]] = None
    _curr_config: Dict[str, Any] = {}

    @overload
    @classmethod
    def get_pipeline(
        cls, task: Literal["sentence-embedding"], model_name: str, **kwargs
    ) -> SentenceTransformer:
        pass

    @overload
    @classmethod
    def get_pipeline(cls, task: str, model_name: str, **kwargs) -> Pipeline:
        pass

    @classmethod
    def get_pipeline(cls, task: str, model_name: str, **kwargs) -> Pipeline:
        with Lock():
            cfg = {
                "task": task,
                "model_name": model_name,
                **kwargs,
            }
            if cfg != cls._curr_config:
                log.info("Loading new config", config=cfg)
                cls._curr_config = cfg
                del cls._curr_pipeline  # Free mem manually
                torch.cuda.empty_cache()
                if task == "sentence-embedding":
                    cls._curr_pipeline = SentenceTransformer(model_name, **kwargs)
                else:
                    cls._curr_pipeline = pipeline(
                        task=task,
                        model=model_name,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        **kwargs,
                    )
            return cls._curr_pipeline
