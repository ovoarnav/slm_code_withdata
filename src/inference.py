from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ARTIFACT_DIR, MAX_LENGTH


def load_classifier(artifact_prefix: str):
    model_dir = ARTIFACT_DIR / artifact_prefix

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    with open(model_dir / "label_map.json", "r") as f:
        label_map = json.load(f)

    label_map = {int(k): v for k, v in label_map.items()}
    return tokenizer, model, label_map


def predict_single(text: str, tokenizer, model, label_map):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": label_map[pred_idx],
        "confidence": float(np.max(probs)),
    }


def predict_failure(note_text: str):
    sub_tok, sub_model, sub_map = load_classifier("subsystem_transformer")
    fail_tok, fail_model, fail_map = load_classifier("failure_mode_transformer")

    subsystem = predict_single(note_text, sub_tok, sub_model, sub_map)
    failure_mode = predict_single(note_text, fail_tok, fail_model, fail_map)

    return {
        "subsystem": subsystem["label"],
        "subsystem_confidence": subsystem["confidence"],
        "failure_mode": failure_mode["label"],
        "failure_mode_confidence": failure_mode["confidence"],
    }


if __name__ == "__main__":
    sample_note = (
        "Engine overheated after repeated cooling alarm. "
        "Raw water flow weak and impeller pieces found in housing."
    )
    print(json.dumps(predict_failure(sample_note), indent=2))
