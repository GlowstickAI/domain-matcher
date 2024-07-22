import os
from typing import Dict

from datasets import ClassLabel, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from domain_matcher.components.pipeline import GSComponent
from domain_matcher.config import DMConfig, ExperimentConfig
from domain_matcher.preprocessing import remove_repetition, remove_uhm

pjoin = os.path.join

TEXT_COL = "text"
LABEL_COL = "label"


def get_tokenized_ds_from_args(ds_dict: Dataset, hparams: ExperimentConfig):
    # Tokenize the dataset and some light preprocessing
    tokenizer = AutoTokenizer.from_pretrained(hparams.pretrained_pipeline)
    if (
        hparams.label_column not in ds_dict.column_names
        or hparams.text_column not in ds_dict.column_names
    ):
        raise ValueError(
            f"Expecting {hparams.label_column} and {hparams.text_column} in dataset"
            f" found {ds_dict.column_names}"
        )

    # Preprocessing
    if hparams.text_column != TEXT_COL:
        ds_dict = ds_dict.rename_column(hparams.text_column, TEXT_COL)
    if hparams.label_column != LABEL_COL:
        ds_dict = ds_dict.rename_column(hparams.label_column, LABEL_COL)

    ds_dict = ds_dict.filter(lambda u: u[TEXT_COL] is not None and len(u[TEXT_COL]) > 0)
    tokenizer_kwargs: Dict[str, str] = {}

    ds_dict = ds_dict.map(lambda u: {TEXT_COL: remove_repetition(remove_uhm(u[TEXT_COL]))})

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess_function(examples):
        return tokenizer(
            examples[TEXT_COL],
            truncation=True,
            padding=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        )

    tokenized_ds_dict = ds_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=list(set(ds_dict.column_names) - {LABEL_COL}),
    )
    return tokenized_ds_dict, tokenizer, ds_dict


def get_trainer_from_args(hparams: ExperimentConfig, tokenized_ds, tokenizer):
    # Create our model and trainer
    class_label: ClassLabel = tokenized_ds.features[LABEL_COL]
    num_classes = class_label.num_classes
    # Setup training objects
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        hparams.pretrained_pipeline,
        num_labels=num_classes,
        id2label={i: class_label.int2str(i) for i in range(num_classes)},
        label2id={label: class_label.str2int(label) for label in class_label.names},
        ignore_mismatched_sizes=True,
    )
    if hparams.freeze_backbone:
        for p in model.base_model.parameters():
            p.require_grad = False

    training_args = TrainingArguments(
        output_dir=pjoin(hparams.ckpt_path, "models"),
        optim=hparams.optimizer,
        lr_scheduler_type=hparams.lr_scheduler_type,
        warmup_ratio=hparams.warmup_ratio,
        overwrite_output_dir=True,
        dataloader_num_workers=2,
        num_train_epochs=hparams.num_train_epochs,
        max_steps=-1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        # Batch size stuff
        per_device_eval_batch_size=32,
        per_device_train_batch_size=32,
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[],
    )
    return trainer


class TrainModelComponent(GSComponent):
    def __init__(
        self,
        config: DMConfig,
        hparams: ExperimentConfig,
    ):
        super().__init__(config)
        self.hparams = hparams

    def __call__(self, dataset):
        if not self.hparams.train_on_oos:
            label_feat = dataset.features[self.config.label_column]
            train_ds = dataset.filter(
                lambda u: label_feat.int2str(u[self.config.label_column]) != self.config.oos_class
            )
        else:
            train_ds = dataset
        tokenized_ds, tokenizer, _ = get_tokenized_ds_from_args(train_ds, self.hparams)
        trainer = get_trainer_from_args(self.hparams, tokenized_ds, tokenizer)

        trainer.train()
        pipe = pipeline(
            "text-classification",
            model=trainer.model,
            tokenizer=tokenizer,
            top_k=None,
            device="cuda",
        )
        pipe.save_pretrained(".cache/pipeline")
        return dataset.map(lambda u: {"prediction": pipe(u["text"])})


class PredictModelComponent(GSComponent):
    def __init__(self, config: DMConfig, hparams: ExperimentConfig):
        super().__init__(config)
        self.hparams = hparams

    def __call__(self, dataset):
        pipe = pipeline("text-classification", model=".cache/pipeline", top_k=None, device="cuda")
        return dataset.map(lambda u: {"prediction": pipe(u["text"])[0]})
