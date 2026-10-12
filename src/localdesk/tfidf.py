"""Portable hashed word + character TF-IDF linear model.

Inference needs only the Python standard library. Like the baseline, the model stores
hash buckets rather than a vocabulary of training text. With 2**18 buckets, anyone
holding the artifact can still test whether a candidate n-gram occurred in training
data, so treat trained artifacts with the same care as the data they were fitted on.
"""

import array
import base64
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from importlib.resources import files

from localdesk.baseline import softmax
from localdesk.config import TaskSpec
from localdesk.core import validate_text

FORMAT = "localdesk-tfidf-v1"
N_BUCKETS = 1 << 18
CHAR_NGRAMS = (3, 5)
DEFAULT_MAX_FEATURES = 40000
# Wider than the legacy baseline grid: TF-IDF logits are often under-confident.
TEMPERATURE_GRID = tuple(0.05 * 1.05**step for step in range(124))


def _bucket(prefix: bytes, gram: str) -> int:
    digest = hashlib.blake2b(prefix + gram.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % N_BUCKETS


def term_counts(text: str) -> tuple[Counter, Counter]:
    """Word 1-2 grams and within-word 3-5 character grams, like sklearn's char_wb."""
    text = unicodedata.normalize("NFKC", validate_text(text))
    text = re.sub(r"\[[^\]]*\]", " ", text).casefold()
    words = re.findall(r"\b\w\w+\b", text)
    word_grams = words + [f"{left} {right}" for left, right in zip(words, words[1:], strict=False)]
    char_grams = []
    low, high = CHAR_NGRAMS
    for word in text.split():
        padded = f" {word} "
        for size in range(low, high + 1):
            if len(padded) <= size:  # A short word counts once, as in sklearn.
                char_grams.append(padded)
                break
            char_grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return (
        Counter(_bucket(b"w", gram) for gram in word_grams),
        Counter(_bucket(b"c", gram) for gram in char_grams),
    )


def vectorize(counts: tuple[Counter, Counter], columns: dict, idf: list[float]) -> dict:
    """Sublinear TF x IDF, L2-normalized per block; unseen buckets are ignored."""
    vector = {}
    for block, block_counts in enumerate(counts):
        weights = {}
        for bucket, count in block_counts.items():
            column = columns.get(block * N_BUCKETS + bucket)
            if column is not None:
                weights[column] = (1 + math.log(count)) * idf[column]
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm:
            vector.update((column, value / norm) for column, value in weights.items())
    return dict(sorted(vector.items()))


def pack(values) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode("ascii")


def unpack(payload: str) -> array.array:
    values = array.array("f")
    values.frombytes(base64.b64decode(payload.encode("ascii"), validate=True))
    return values


class TfidfBackend:
    name = "tfidf"
    probability_kind = "softmax_temperature_fitted_on_calibration"

    def __init__(self, *, bundle: dict):
        if bundle.get("format") != FORMAT:
            raise ValueError("unsupported TF-IDF model format")
        self.task = TaskSpec.from_dict(bundle["task"])
        if bundle.get("task_sha256") != self.task.digest:
            raise ValueError("TF-IDF task hash mismatch")
        classes = bundle["classes"]
        if set(classes) != set(self.task.labels) or len(classes) != len(self.task.labels):
            raise ValueError("TF-IDF class keys mismatch")
        if bundle["n_buckets"] != N_BUCKETS or bundle["char_ngrams"] != list(CHAR_NGRAMS):
            raise ValueError("TF-IDF feature configuration mismatch")
        buckets, idf = bundle["columns"], bundle["idf"]
        if len(buckets) != len(idf) or len(set(buckets)) != len(buckets):
            raise ValueError("TF-IDF column table is inconsistent")
        if any(type(b) is not int or not 0 <= b < 2 * N_BUCKETS for b in buckets):
            raise ValueError("TF-IDF column bucket out of range")
        coef = unpack(bundle["coef"])
        if len(coef) != len(buckets) * len(classes) or len(bundle["intercept"]) != len(classes):
            raise ValueError("TF-IDF coefficient shape mismatch")
        values = [*idf, *coef, *bundle["intercept"]]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("non-finite TF-IDF model value")
        temperature = bundle["temperature"]
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid calibration temperature")
        self.bundle, self.classes, self.coef, self.idf = bundle, classes, coef, idf
        self.columns = {bucket: column for column, bucket in enumerate(buckets)}

    def logits(self, text: str) -> list[float]:
        width = len(self.classes)
        logits = list(self.bundle["intercept"])
        for column, value in vectorize(term_counts(text), self.columns, self.idf).items():
            row = self.coef[column * width : (column + 1) * width]
            for index in range(width):
                logits[index] += row[index] * value
        return logits

    def predict(self, text: str) -> dict[str, dict[str, float]]:
        probabilities = softmax(self.logits(text), self.bundle["temperature"])
        return {"category": dict(zip(self.classes, probabilities, strict=True))}


def demo_backend() -> TfidfBackend:
    """Packaged demo: the shipped IT Support Tickets experiment, built by scripts/train_demo.py."""
    payload = files("localdesk").joinpath("assets/demo.json.gz").read_bytes()
    return TfidfBackend(bundle=json.loads(gzip.decompress(payload)))


def fit_tfidf(experiment, prepared):
    """Fit on train, export, verify parity, then fit temperature on calibration."""
    import importlib.metadata
    import warnings

    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    train = prepared.rows["train"]
    counts = [term_counts(row["state"]) for row in train]
    max_features = experiment.training.get("max_features", DEFAULT_MAX_FEATURES)
    buckets, idf = [], []
    for block in range(2):
        frequency = Counter(bucket for row in counts for bucket in row[block])
        # Keep the most frequent buckets per block, ties broken by bucket for determinism.
        kept = sorted(frequency, key=lambda bucket: (-frequency[bucket], bucket))[:max_features]
        for bucket in sorted(kept):
            buckets.append(block * N_BUCKETS + bucket)
            idf.append(math.log((1 + len(train)) / (1 + frequency[bucket])) + 1)
    columns = {bucket: column for column, bucket in enumerate(buckets)}

    def matrix(row_counts):
        indices, positions, values = [], [], []
        for index, row in enumerate(row_counts):
            for column, value in vectorize(row, columns, idf).items():
                indices.append(index)
                positions.append(column)
                values.append(value)
        return csr_matrix((values, (indices, positions)), shape=(len(row_counts), len(buckets)))

    classifier = LogisticRegression(
        C=experiment.training.get("C", 4.0),
        class_weight="balanced",
        max_iter=experiment.training.get("max_iter", 2000),
        solver="lbfgs",
        random_state=experiment.seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(matrix(counts), [row["gold"]["category"] for row in train])
    coef, intercept = classifier.coef_, classifier.intercept_
    if len(classifier.classes_) == 2:
        # sklearn stores one binary logit; [0, z] gives identical sigmoid probabilities.
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    labels = classifier.classes_.tolist()
    bundle = {
        "format": FORMAT,
        "task": experiment.task.to_dict(),
        "task_sha256": experiment.task.digest,
        "classes": labels,
        "n_buckets": N_BUCKETS,
        "char_ngrams": list(CHAR_NGRAMS),
        "columns": buckets,
        "idf": idf,
        # Column-major float32: one contiguous run of class weights per feature.
        "coef": pack(coef.T.astype(np.float32).ravel().tolist()),
        "intercept": intercept.tolist(),
        "temperature": 1.0,
        "dataset_sha256": prepared.manifest["split_sha256"],
        "parameters": classifier.get_params(),
        "max_features": max_features,
        "versions": {
            name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "scipy")
        },
    }
    calibration = prepared.rows["calibration"]
    backend = TfidfBackend(bundle=bundle)
    exported = np.array([backend.logits(row["state"]) for row in calibration])
    fitted = classifier.decision_function(matrix([term_counts(r["state"]) for r in calibration]))
    if fitted.ndim == 1:
        fitted = np.column_stack([np.zeros(len(fitted)), fitted])
    # float32 export: compare to sklearn within storage precision.
    if not np.allclose(exported, fitted, rtol=1e-4, atol=1e-4):
        raise RuntimeError("portable TF-IDF export differs from fitted logits")
    label_ids = {label: index for index, label in enumerate(labels)}
    gold = [label_ids[row["gold"]["category"]] for row in calibration]

    def loss(temperature):
        return -sum(
            math.log(max(softmax(list(z), temperature)[target], 1e-15))
            for z, target in zip(exported, gold, strict=True)
        ) / len(gold)

    temperature = min(TEMPERATURE_GRID, key=loss)
    if temperature in (TEMPERATURE_GRID[0], TEMPERATURE_GRID[-1]):
        warnings.warn("calibration temperature reached the search bound", stacklevel=2)
    bundle.update(temperature=temperature, calibration_nll=loss(temperature))
    return TfidfBackend(bundle=bundle), bundle
