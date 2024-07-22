from datetime import datetime
from typing import List

from pydantic import BaseModel, Extra

from domain_matcher.utils import md5_hash

TEXT_COL = "text"
LABEL_COL = "label"


class ExperimentConfig(BaseModel, extra=Extra.ignore):
    # General stuff
    dataset_path: str
    dataset_args: List[str] = []
    pretrained_pipeline: str
    text_column: str = TEXT_COL
    label_column: str = LABEL_COL
    seed: int = 1337
    # Experiments
    train_on_oos: bool = True
    # HP Search
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 50
    freeze_backbone: bool = False
    loss: str = "crossentropy"

    @property
    def ckpt_path(self):
        return f"./logs/ckpt-{datetime.now().strftime('%Y-%m-%d')}-{md5_hash(self.dict())[:5]}"


class DMConfig(BaseModel):
    seed: int = 1337
    artifact_path: str = "/tmp"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Dataset info
    text_column: str
    label_column: str
    oos_class: str

    # Let the model predict oos even if it's in Domain.
    allow_oos_pred: bool = True

    def __hash__(self):
        return int(md5_hash(self.dict()), base=16)

    def to_hash(self):
        return md5_hash(
            self.dict(
                by_alias=True,
            )
        )
