"""
Lightweight heuristics-based AI vs Human text detector.

The goal is not to be perfect but to demonstrate how stylometric signals such
as sentence length, lexical diversity, burstiness, and repetitiveness can be
combined to approximate an AI/Human likelihood. The detector exposes a simple
API for the Streamlit UI.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "to",
    "of",
    "in",
    "is",
    "it",
    "that",
    "as",
    "for",
    "with",
    "was",
    "were",
    "on",
    "by",
    "be",
    "are",
    "this",
    "at",
    "from",
    "or",
    "which",
    "but",
    "have",
    "has",
    "had",
    "not",
    "we",
    "they",
    "you",
    "i",
    "their",
    "its",
    "our",
    "will",
    "can",
    "about",
    "also",
    "into",
    "more",
    "than",
}


WORD_RE = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[.!?？！。；;]+")


@dataclass
class DetectionResult:
    label: str
    ai_score: float
    human_score: float
    features: Dict[str, float]

    def as_dict(self) -> Dict[str, float | str]:
        data: Dict[str, float | str] = {
            "label": self.label,
            "ai_score": self.ai_score,
            "human_score": self.human_score,
        }
        data.update(self.features)
        return data


class HeuristicAIHumanDetector:
    """Simple detector that mixes stylometric heuristics."""

    def __init__(self) -> None:
        # Weight signs indicate how much a feature pushes toward "AI".
        self.weights = {
            "complexity": 1.2,
            "burstiness": -1.4,
            "repetition": 1.1,
            "diversity": -1.3,
            "stopword_ratio": 0.8,
            "punctuation_density": -0.4,
            "entropy": -0.7,
        }
        self.bias = 0.15

    def predict(self, text: str) -> DetectionResult:
        tokens = WORD_RE.findall(text.lower())
        token_lengths = [len(t) for t in tokens if t.strip()]
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
        features = self._extract_features(text, tokens, token_lengths, sentences)
        score = self.bias
        for name, value in features.items():
            if name in self.weights:
                score += self.weights[name] * value
        ai_score = self._sigmoid(score)
        human_score = 1 - ai_score
        label = "AI-written" if ai_score >= 0.5 else "Human-written"
        return DetectionResult(label=label, ai_score=ai_score, human_score=human_score, features=features)

    def batch_predict(self, texts: Sequence[str]) -> List[DetectionResult]:
        return [self.predict(text) for text in texts]

    @staticmethod
    def _extract_features(
        text: str,
        tokens: Sequence[str],
        token_lengths: Sequence[int],
        sentences: Sequence[str],
    ) -> Dict[str, float]:
        total_chars = len(text) if text else 1
        total_tokens = len(tokens) if tokens else 1
        avg_sentence_len = _safe_mean(len(s.split()) for s in sentences) if sentences else len(tokens)
        avg_word_len = _safe_mean(token_lengths) if token_lengths else 0.0
        stopword_ratio = sum(1 for t in tokens if t in STOPWORDS) / total_tokens
        punctuation_density = _clamp(sum(1 for ch in text if ch in ".,;:!?()[]\"'") / total_chars)
        uppercase_ratio = _clamp(sum(1 for ch in text if ch.isupper()) / total_chars)
        digit_ratio = _clamp(sum(1 for ch in text if ch.isdigit()) / total_chars)
        diversity = len(set(tokens)) / total_tokens
        repetition = _get_repetition(tokens)
        burstiness = _get_burstiness(sentences)
        entropy = _get_entropy(tokens)

        complexity = _scale(avg_sentence_len, 10, 40) * 0.7 + _scale(avg_word_len, 4, 8) * 0.3

        return {
            "complexity": complexity,
            "burstiness": burstiness,
            "repetition": repetition,
            "diversity": diversity,
            "stopword_ratio": stopword_ratio,
            "punctuation_density": punctuation_density,
            "uppercase_ratio": uppercase_ratio,
            "digit_ratio": digit_ratio,
            "entropy": entropy,
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _scale(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return _clamp((value - min_value) / (max_value - min_value))


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(value, max_value))


def _get_repetition(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    max_freq = max(freq.values())
    return max_freq / len(tokens)


def _get_burstiness(sentences: Sequence[str]) -> float:
    lengths = [len(s.split()) for s in sentences if s.split()]
    if not lengths:
        return 0.0
    if len(lengths) == 1:
        return 0.2
    mean_len = statistics.mean(lengths)
    std_len = statistics.pstdev(lengths)
    if mean_len == 0:
        return 0.0
    return _clamp(std_len / mean_len)


def _get_entropy(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    total = len(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    max_entropy = math.log(len(freq), 2) if len(freq) > 1 else 1.0
    return _clamp(entropy / max_entropy if max_entropy > 0 else entropy)


def detect_text(text: str) -> DetectionResult:
    detector = HeuristicAIHumanDetector()
    return detector.predict(text)


def detect_batch(texts: Sequence[str]) -> List[DetectionResult]:
    detector = HeuristicAIHumanDetector()
    return detector.batch_predict(texts)

