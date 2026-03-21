from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import (
    ARTIFACT_DIR,
    MAX_LENGTH,
    MODEL_NAME,
    RANDOM_STATE,
    SMOKE_TEST,
    SMOKE_TEST_ROWS,
    TECH_NOTES_PATH,
)
from data_loader import load_technician_notes


def make_splits(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vc = df[target_col].value_counts()
    keep_classes = vc[vc >= 3].index
    df = df[df[target_col].isin(keep_classes)].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df[target_col],
    )

    return train_df, val_df, test_df


def encode_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
):
    le = LabelEncoder()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["label"] = le.fit_transform(train_df[target_col])
    val_df["label"] = le.transform(val_df[target_col])
    test_df["label"] = le.transform(test_df[target_col])

    return train_df, val_df, test_df, le


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["note_text", "label"]].reset_index(drop=True))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_one_classifier(df: pd.DataFrame, target_col: str, artifact_prefix: str):
    train_df, val_df, test_df = make_splits(df, target_col)
    train_df, val_df, test_df, label_encoder = encode_labels(
        train_df, val_df, test_df, target_col
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["note_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = to_hf_dataset(train_df).map(tokenize_fn, batched=True)
    val_ds = to_hf_dataset(val_df).map(tokenize_fn, batched=True)
    test_ds = to_hf_dataset(test_df).map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining {artifact_prefix} on device: {device}")

    training_args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / f"{artifact_prefix}_runs"),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(test_ds)

    preds_output = trainer.predict(test_ds)
    pred_ids = np.argmax(preds_output.predictions, axis=-1)
    true_ids = preds_output.label_ids

    report = classification_report(
        true_ids,
        pred_ids,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    save_dir = ARTIFACT_DIR / artifact_prefix
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / "label_map.json", "w") as f:
        json.dump(
            {int(i): cls for i, cls in enumerate(label_encoder.classes_)},
            f,
            indent=2,
        )

    payload = {
        "target_col": target_col,
        "model_name": MODEL_NAME,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    set_seed(RANDOM_STATE)

    df = load_technician_notes(TECH_NOTES_PATH)

    if SMOKE_TEST:
        df = df.sample(n=min(SMOKE_TEST_ROWS, len(df)), random_state=RANDOM_STATE).copy()

    print(f"Loaded rows: {len(df)}")

    subsystem_results = train_one_classifier(
        df=df,
        target_col="extracted_subsystem",
        artifact_prefix="subsystem_transformer",
    )

    failure_mode_results = train_one_classifier(
        df=df,
        target_col="extracted_failure_mode",
        artifact_prefix="failure_mode_transformer",
    )

    summary = {
        "subsystem_transformer": subsystem_results,
        "failure_mode_transformer": failure_mode_results,
    }

    with open(ARTIFACT_DIR / "slm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Training complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
