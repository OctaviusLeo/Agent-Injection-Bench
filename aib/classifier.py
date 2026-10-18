"""Prompt-injection classifiers for the `filter` defense (task 4.1; Req 4.4, 4.5).

`InjectionClassifier` wraps protectai/deberta-v3-base-prompt-injection-v2 from the local snapshot in
~/aib-models/classifier (never the network). A text is split into windows of at most `window_tokens`
tokens (special tokens included) with `overlap` tokens shared between neighbours; its score is the
highest INJECTION probability over the windows. Scores are cached by the text's SHA-256 (the same
text always gets the same score) and the model runs under a lock, so several workers can share it.

`KeywordClassifier` is the test and CI stand-in: it flags text that contains any of its needles.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_KEYWORDS = ("TODO:", "###(system_message)", "IMPORTANT!!!", "<INFORMATION>", "assistant_instruction")
SPECIAL_TOKENS = 2  # [CLS] and [SEP]


def window_spans(n_tokens: int, window_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) spans over `n_tokens` content tokens: each at most `window_tokens - 2` long (room for
    the special tokens), consecutive spans sharing exactly `overlap` tokens, together covering everything."""
    size = window_tokens - SPECIAL_TOKENS
    if size <= overlap:
        raise ValueError("window must be longer than the overlap plus the special tokens")
    if n_tokens <= size:
        return [(0, n_tokens)]
    spans, start, stride = [], 0, size - overlap
    while True:
        end = min(start + size, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            return spans
        start += stride


class InjectionClassifier:
    def __init__(self, model_dir: Path | str | None = None, window_tokens: int = 512, overlap: int = 64,
                 injection_label: str = "INJECTION", tokenize: Callable[[str], list[int]] | None = None,
                 score_windows: Callable[[list[list[int]]], list[float]] | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path(os.environ.get("AIB_MODELS", Path.home() / "aib-models")) / "classifier"
        self.window_tokens, self.overlap, self.injection_label = window_tokens, overlap, injection_label
        self._tokenize, self._score_windows = tokenize, score_windows
        self._cache: dict[str, float] = {}
        self._lock = threading.Lock()
        self.calls = 0  # model invocations (cache misses), for timing reports

    def _load(self) -> None:
        """Load tokenizer and model from the local snapshot only."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"classifier snapshot not found in {self.model_dir} (run scripts/fetch_models.py)")
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
        model.eval()
        label_to_id = {v: int(k) for k, v in model.config.id2label.items()}
        if self.injection_label not in label_to_id:
            raise ValueError(f"label {self.injection_label!r} not in {model.config.id2label}")
        index = label_to_id[self.injection_label]
        cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

        def tokenize(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        def score_windows(windows: list[list[int]]) -> list[float]:
            seqs = [[cls, *w, sep] for w in windows]
            width = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs])
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
            with torch.no_grad():
                probs = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)
            return [float(p) for p in probs[:, index]]

        self._tokenize, self._score_windows = tokenize, score_windows

    def score(self, text: str) -> float:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self._tokenize is None or self._score_windows is None:
                self._load()
            ids = self._tokenize(text)
            windows = [ids[a:b] for a, b in window_spans(len(ids), self.window_tokens, self.overlap)]
            value = max(self._score_windows(windows))
            self.calls += 1
            self._cache[key] = value
            return value


class KeywordClassifier:
    """Flags (score 1.0) text containing any needle, case-insensitively; otherwise 0.0."""

    def __init__(self, needles: Sequence[str] = DEFAULT_KEYWORDS) -> None:
        self.needles = tuple(n.lower() for n in needles)

    def score(self, text: str) -> float:
        lowered = text.lower()
        return 1.0 if any(n in lowered for n in self.needles) else 0.0
