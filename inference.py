from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer

from config import ARTIFACT_DIR, MAX_LENGTH


def load_classifier(artifact_prefix: str):
    model_dir = Path(ARTIFACT_DIR) / artifact_prefix

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    tokenizer = BertTokenizer.from_pretrained(str(model_dir), do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    label_map_path = model_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label map: {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    label_map = {int(k): v for k, v in label_map.items()}
    return tokenizer, model, label_map


def predict_single(
    text: str,
    tokenizer,
    model,
    label_map: dict[int, str],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

    if pred_idx not in label_map:
        raise KeyError(f"Predicted class index {pred_idx} not found in label_map")

    return {
        "label": label_map[pred_idx],
        "confidence": float(np.max(probs)),
    }


def predict_failure(note_text: str) -> dict[str, Any]:
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
    sample_note = input("Enter technician note: ").strip()
    if not sample_note:
        raise ValueError("Technician note cannot be empty.")

    result = predict_failure(sample_note)
    print(json.dumps(result, indent=2))