import argparse
import copy
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CPU_WORKER_LIMIT = max(1, min(8, (os.cpu_count() or 2) - 1))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(CPU_WORKER_LIMIT))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_WORKER_LIMIT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_WORKER_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_WORKER_LIMIT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_WORKER_LIMIT))

import joblib

try:
    from joblib.externals.loky.backend import context as loky_context
    loky_context.physical_cores_cache = CPU_WORKER_LIMIT
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except Exception:
    HAS_POSTHOCS = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

@dataclass
class Profile:
    name: str
    seeds: List[int]
    v1_per_class: int
    v3_total: int
    ssl_epochs: int
    cls_epochs: int
    patience: int
    bootstrap_iterations: int
    mc_passes: int
    unknown_class_limit: int
    drift_window: int
    baseline_estimators: int
    batch_gpu: int
    batch_cpu: int
    num_workers: int

@dataclass
class PathsConfig:
    project_root: Path
    data_root: Path
    output_root: Path
    ton_v1: Path
    bot_v1: Path
    ton_v3: Path
    bot_v3: Optional[Path]

@dataclass
class RunConfig:
    profile: Profile
    paths: PathsConfig
    device: str
    batch_size: int
    alpha_conformal: float
    alpha_open: float
    mask_ratio: float
    latent_dim: int
    dropout: float
    ssl_lr: float
    cls_lr: float
    weight_decay: float
    adaptation_budgets: List[int]
    max_plot_points: int
    run_id: str

class MemoryMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.peak = self.process.memory_info().rss
        self.thread = None

    def _run(self):
        while self.running:
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except Exception:
            pass

    @property
    def peak_mb(self):
        return self.peak / (1024 ** 2)

class NetworkPreprocessor:
    def __init__(self, clip_value: float = 10.0):
        self.clip_value = clip_value
        self.columns = []
        self.medians = None
        self.log_mask = None
        self.scaler = RobustScaler(quantile_range=(5.0, 95.0))

    def fit(self, frame: pd.DataFrame):
        self.columns = list(frame.columns)
        numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.medians = numeric.median(axis=0).fillna(0.0)
        numeric = numeric.fillna(self.medians)
        mins = numeric.min(axis=0)
        skew = numeric.skew(axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self.log_mask = ((mins >= 0) & (skew.abs() > 2.0)).values
        values = numeric.to_numpy(dtype=np.float64)
        if np.any(self.log_mask):
            values[:, self.log_mask] = np.log1p(np.maximum(values[:, self.log_mask], 0.0))
        self.scaler.fit(values)
        return self

    def transform(self, frame: pd.DataFrame):
        aligned = frame.reindex(columns=self.columns)
        numeric = aligned.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.medians)
        values = numeric.to_numpy(dtype=np.float64)
        if np.any(self.log_mask):
            values[:, self.log_mask] = np.log1p(np.maximum(values[:, self.log_mask], 0.0))
        values = self.scaler.transform(values)
        values = np.nan_to_num(values, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        values = np.clip(values, -self.clip_value, self.clip_value)
        return values.astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame):
        return self.fit(frame).transform(frame)

class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        hidden1 = max(128, min(512, input_dim * 8))
        hidden2 = max(96, min(256, input_dim * 4))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class MaskedVICRegAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = TabularEncoder(input_dim, latent_dim, dropout)
        hidden = max(96, min(256, input_dim * 4))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class DeepClassifier(nn.Module):
    def __init__(self, encoder: TabularEncoder, latent_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / torch.exp(self.log_temperature).clamp(0.05, 20.0)

    @property
    def temperature(self):
        return float(torch.exp(self.log_temperature.detach()).cpu().item())

class PageHinkley:
    def __init__(self, delta: float = 0.002, threshold: float = 20.0, alpha: float = 0.995):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 0

    def update(self, value: float):
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative = self.alpha * self.cumulative + value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        detected = (self.cumulative - self.minimum) > self.threshold
        if detected:
            self.reset()
        return detected

def get_profile(name: str):
    profiles = {
        "quick": Profile("quick", [42], 12000, 60000, 5, 8, 3, 100, 5, 1, 2500, 120, 1024, 256, 0),
        "standard": Profile("standard", [17, 42, 71], 60000, 350000, 20, 25, 6, 300, 10, 3, 5000, 300, 2048, 512, 0),
        "publication": Profile("publication", [17, 29, 42, 71, 101], 150000, 800000, 45, 40, 10, 1000, 20, 99, 10000, 500, 4096, 1024, 2)
    }
    return profiles[name]

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_logging(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(output_root / "execution.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)

def sha1_file(path: Path, block_size: int = 1024 * 1024):
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def manifest_sha1(path: Path):
    manifest = path.parent.parent / "manifest-sha1.txt"
    if manifest.exists():
        target_names = {path.name, f"data/{path.name}", f"data\\{path.name}"}
        for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[1].replace("*", "").strip() in target_names:
                return parts[0].strip()
    return "manifest_not_found"

def discover_paths(project_root: Path):
    data_root = project_root / "data"
    expected_ton_v1 = data_root / "7ca78ae35fa4961a_MOHANAD_A4706" / "data" / "NF-ToN-IoT.csv"
    expected_bot_v1 = data_root / "de2c6f75dd50d933_MOHANAD_A4706" / "data" / "NF-BoT-IoT.csv"
    expected_ton_v3 = data_root / "02934b58528a226b_NFV3DATA-A11964_A11964" / "data" / "NF-ToN-IoT-v3.csv"
    ton_v1 = expected_ton_v1 if expected_ton_v1.exists() else next(iter(data_root.rglob("NF-ToN-IoT.csv")), expected_ton_v1)
    bot_v1 = expected_bot_v1 if expected_bot_v1.exists() else next(iter(data_root.rglob("NF-BoT-IoT.csv")), expected_bot_v1)
    ton_v3 = expected_ton_v3 if expected_ton_v3.exists() else next(iter(data_root.rglob("NF-ToN-IoT-v3.csv")), expected_ton_v3)
    bot_v3_candidates = list(data_root.rglob("NF-BoT-IoT-v3.csv"))
    bot_v3 = bot_v3_candidates[0] if bot_v3_candidates else None
    output_root = project_root / "security_article_output"
    return PathsConfig(project_root, data_root, output_root, ton_v1, bot_v1, ton_v3, bot_v3)

def validate_paths(paths: PathsConfig):
    missing = [str(path) for path in [paths.ton_v1, paths.bot_v1, paths.ton_v3] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required datasets:\n" + "\n".join(missing))

def read_feature_schema(dataset_path: Path):
    candidates = list(dataset_path.parent.glob("NetFlow_*_Features.csv"))
    if not candidates:
        return []
    frame = pd.read_csv(candidates[0])
    if frame.empty:
        return []
    return frame.iloc[:, 0].astype(str).str.strip().tolist()

def detect_columns(columns: Sequence[str]):
    normalized = {str(c).strip().lower(): c for c in columns}
    binary_candidates = ["label", "binary_label", "class", "target"]
    attack_candidates = ["attack", "attack_label", "category", "attack_category"]
    binary = next((normalized[c] for c in binary_candidates if c in normalized), None)
    attack = next((normalized[c] for c in attack_candidates if c in normalized), None)
    if binary is None:
        for c in columns:
            low = str(c).lower()
            if "label" in low and "attack" not in low:
                binary = c
                break
    if attack is None:
        for c in columns:
            low = str(c).lower()
            if "attack" in low or "category" in low:
                attack = c
                break
    timestamp_candidates = []
    for c in columns:
        low = str(c).lower()
        if any(token in low for token in ["timestamp", "flow_start", "start_time", "start_milli", "start_sec"]):
            timestamp_candidates.append(c)
    return binary, attack, timestamp_candidates

def normalize_attack_name(value: Any):
    text = str(value).strip()
    low = "".join(ch for ch in text.lower() if ch.isalnum())
    if low in {"benign", "normal", "0", "none", "background"}:
        return "Benign"
    if "ddos" in low:
        return "DDoS"
    if low == "dos" or ("dos" in low and "ddos" not in low):
        return "DoS"
    if "recon" in low or "scan" in low or "probe" in low:
        return "Reconnaissance"
    if "brute" in low:
        return "Brute Force"
    if "password" in low:
        return "Password"
    if "xss" in low or "crosssite" in low:
        return "XSS"
    if "inject" in low:
        return "Injection"
    if "mitm" in low or "maninthemiddle" in low:
        return "MITM"
    if "ransom" in low:
        return "Ransomware"
    if "theft" in low:
        return "Theft"
    if "backdoor" in low:
        return "Backdoor"
    if "shell" in low:
        return "Shellcode"
    if "worm" in low:
        return "Worms"
    if "fuzz" in low:
        return "Fuzzers"
    if "exploit" in low:
        return "Exploits"
    if "infil" in low:
        return "Infiltration"
    if "bot" in low or "mirai" in low:
        return "Botnet"
    if "generic" in low:
        return "Generic"
    if "analysis" in low:
        return "Analysis"
    return text if text else "UnknownLabel"

def infer_binary_labels(frame: pd.DataFrame, binary_col: Optional[str], attack_col: str):
    if binary_col is not None and binary_col in frame.columns:
        raw = frame[binary_col]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().mean() > 0.9:
            return (numeric.fillna(0).astype(float) > 0).astype(int)
        mapped = raw.astype(str).str.lower().str.strip().map(lambda x: 0 if x in {"0", "benign", "normal", "false"} else 1)
        return mapped.astype(int)
    return (frame[attack_col].map(normalize_attack_name) != "Benign").astype(int)

def safe_read_header(path: Path):
    return pd.read_csv(path, nrows=0).columns.tolist()

def sampling_columns(path: Path, columns: Sequence[str], binary_col: Optional[str], attack_col: Optional[str], timestamps: Sequence[str]):
    schema = read_feature_schema(path)
    base = [c for c in schema if c in columns] if schema else list(columns)
    selected = []
    for col in base:
        low = str(col).lower()
        if any(token in low for token in ["ipv4_src", "ipv4_dst", "src_ip", "dst_ip", "ip_src", "ip_dst", "mac"]):
            continue
        selected.append(col)
    for col in [binary_col, attack_col, *timestamps]:
        if col is not None and col in columns and col not in selected:
            selected.append(col)
    return selected

def stratified_chunk_sample(path: Path, max_per_class: int, seed: int, chunksize: int = 200000):
    columns = safe_read_header(path)
    binary_col, attack_col, timestamps = detect_columns(columns)
    if attack_col is None:
        raise ValueError(f"Attack label column not found in {path}")
    usecols = sampling_columns(path, columns, binary_col, attack_col, timestamps)
    reservoirs: Dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        chunk[attack_col] = chunk[attack_col].map(normalize_attack_name)
        chunk["__sample_key__"] = rng.random(len(chunk))
        for label, group in chunk.groupby(attack_col, sort=False):
            previous = reservoirs.get(label)
            combined = group if previous is None else pd.concat([previous, group], ignore_index=True)
            if len(combined) > max_per_class:
                combined = combined.nsmallest(max_per_class, "__sample_key__")
            reservoirs[label] = combined
        if chunk_index % 10 == 0:
            logging.info("Sampling %s chunk %s", path.name, chunk_index)
    sampled = pd.concat(reservoirs.values(), ignore_index=True).drop(columns=["__sample_key__"])
    sampled[attack_col] = sampled[attack_col].map(normalize_attack_name)
    sampled["__binary_label__"] = infer_binary_labels(sampled, binary_col, attack_col)
    sampled["__attack_label__"] = sampled[attack_col]
    return sampled, binary_col, attack_col, timestamps

def uniform_reservoir_sample(path: Path, target_rows: int, seed: int, chunksize: int = 200000):
    columns = safe_read_header(path)
    binary_col, attack_col, timestamps = detect_columns(columns)
    if attack_col is None:
        raise ValueError(f"Attack label column not found in {path}")
    usecols = sampling_columns(path, columns, binary_col, attack_col, timestamps)
    reservoir = None
    rng = np.random.default_rng(seed)
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        chunk[attack_col] = chunk[attack_col].map(normalize_attack_name)
        chunk["__sample_key__"] = rng.random(len(chunk))
        combined = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        reservoir = combined.nsmallest(target_rows, "__sample_key__") if len(combined) > target_rows else combined
        if chunk_index % 10 == 0:
            logging.info("Reservoir sampling %s chunk %s", path.name, chunk_index)
    sampled = reservoir.drop(columns=["__sample_key__"]).reset_index(drop=True)
    sampled["__binary_label__"] = infer_binary_labels(sampled, binary_col, attack_col)
    sampled["__attack_label__"] = sampled[attack_col]
    return sampled, binary_col, attack_col, timestamps

def choose_feature_columns(frame: pd.DataFrame, schema: Sequence[str], binary_col: Optional[str], attack_col: Optional[str], timestamps: Sequence[str], allow_temporal_aggregates: bool = True):
    candidates = [c for c in schema if c in frame.columns] if schema else list(frame.columns)
    excluded_exact = {binary_col, attack_col, "__binary_label__", "__attack_label__", None}
    selected = []
    for col in candidates:
        low = str(col).lower()
        if col in excluded_exact:
            continue
        if any(token in low for token in ["ipv4_src", "ipv4_dst", "src_ip", "dst_ip", "ip_src", "ip_dst", "mac", "dataset"]):
            continue
        if col in timestamps or any(token in low for token in ["timestamp", "flow_start", "flow_end", "start_time", "end_time"]):
            continue
        if not allow_temporal_aggregates and any(token in low for token in ["iat", "ipat", "interarrival"]):
            continue
        selected.append(col)
    usable = []
    for col in selected:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().mean() >= 0.5 and numeric.nunique(dropna=True) > 1:
            usable.append(col)
    return usable

def matched_feature_columns(source: pd.DataFrame, target: pd.DataFrame, source_features: Sequence[str], target_features: Sequence[str]):
    target_map = {str(c).strip().lower(): c for c in target_features}
    source_map = {str(c).strip().lower(): c for c in source_features}
    common_keys = sorted(set(source_map).intersection(target_map))
    pairs = [(source_map[key], target_map[key]) for key in common_keys]
    source_cols = [a for a, b in pairs]
    target_cols = [b for a, b in pairs]
    valid_source = []
    valid_target = []
    for s, t in zip(source_cols, target_cols):
        if pd.to_numeric(source[s], errors="coerce").notna().mean() >= 0.5 and pd.to_numeric(target[t], errors="coerce").notna().mean() >= 0.5:
            valid_source.append(s)
            valid_target.append(t)
    return valid_source, valid_target

def four_way_split(frame: pd.DataFrame, label_col: str, seed: int):
    train, remaining = train_test_split(frame, test_size=0.4, stratify=frame[label_col], random_state=seed)
    validation, remaining = train_test_split(remaining, test_size=0.5, stratify=remaining[label_col], random_state=seed + 1)
    calibration, test = train_test_split(remaining, test_size=0.5, stratify=remaining[label_col], random_state=seed + 2)
    return train.reset_index(drop=True), validation.reset_index(drop=True), calibration.reset_index(drop=True), test.reset_index(drop=True)

def class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / np.maximum(counts * n_classes, 1.0)
    weights = np.clip(weights, 0.25, 20.0)
    return weights.astype(np.float32)

def make_loader(x: np.ndarray, y: Optional[np.ndarray], batch_size: int, shuffle: bool, workers: int):
    xt = torch.from_numpy(x.astype(np.float32))
    dataset = TensorDataset(xt) if y is None else TensorDataset(xt, torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False)

def covariance_loss(z):
    n, d = z.shape
    z = z - z.mean(dim=0)
    covariance = (z.T @ z) / max(n - 1, 1)
    off_diagonal = covariance.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return (off_diagonal ** 2).sum() / d

def variance_loss(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1.0 - std))

def augment_tabular(x, mask_probability: float = 0.15, noise_std: float = 0.03):
    keep = (torch.rand_like(x) > mask_probability).float()
    noise = torch.randn_like(x) * noise_std
    return x * keep + noise

def amp_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def pretrain_ssl(x_train: np.ndarray, x_val: np.ndarray, config: RunConfig, seed: int, model_path: Optional[Path] = None):
    set_global_seed(seed)
    device = torch.device(config.device)
    model = MaskedVICRegAutoencoder(x_train.shape[1], config.latent_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ssl_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.profile.ssl_epochs, 1))
    train_loader = make_loader(x_train, None, config.batch_size, True, config.profile.num_workers)
    val_loader = make_loader(x_val, None, config.batch_size, False, config.profile.num_workers)
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    best_state = None
    best_loss = float("inf")
    patience_count = 0
    history = []
    for epoch in range(config.profile.ssl_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            feature_mask = torch.rand_like(x) < config.mask_ratio
            masked = x.clone()
            masked[feature_mask] = 0.0
            view1 = augment_tabular(x)
            view2 = augment_tabular(x)
            with amp_context(device, amp_enabled):
                reconstruction, _ = model(masked)
                z1 = model.projector(model.encoder(view1))
                z2 = model.projector(model.encoder(view2))
                reconstruction_error = ((reconstruction - x) ** 2 * feature_mask.float()).sum() / feature_mask.float().sum().clamp_min(1.0)
                invariance = F.mse_loss(z1, z2)
                variance = variance_loss(z1) + variance_loss(z2)
                covariance = covariance_loss(z1) + covariance_loss(z2)
                loss = reconstruction_error + 5.0 * invariance + 2.0 * variance + 0.02 * covariance
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                feature_mask = torch.rand_like(x) < config.mask_ratio
                masked = x.clone()
                masked[feature_mask] = 0.0
                reconstruction, _ = model(masked)
                loss = ((reconstruction - x) ** 2 * feature_mask.float()).sum() / feature_mask.float().sum().clamp_min(1.0)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        scheduler.step()
        if patience_count >= config.profile.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if model_path is not None:
        torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1], "latent_dim": config.latent_dim, "dropout": config.dropout}, model_path)
    return model, pd.DataFrame(history)

def train_deep_classifier(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, pretrained: Optional[MaskedVICRegAutoencoder], n_classes: int, config: RunConfig, seed: int, model_path: Optional[Path] = None):
    set_global_seed(seed)
    device = torch.device(config.device)
    if pretrained is None:
        encoder = TabularEncoder(x_train.shape[1], config.latent_dim, config.dropout)
    else:
        encoder = copy.deepcopy(pretrained.encoder).cpu()
    model = DeepClassifier(encoder, config.latent_dim, n_classes, config.dropout).to(device)
    weights = torch.from_numpy(class_weights(y_train, n_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.01)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.cls_lr * 0.35},
        {"params": model.head.parameters(), "lr": config.cls_lr}
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.profile.cls_epochs, 1))
    train_loader = make_loader(x_train, y_train, config.batch_size, True, config.profile.num_workers)
    val_loader = make_loader(x_val, y_val, config.batch_size, False, config.profile.num_workers)
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    best_state = None
    best_f1 = -1.0
    patience_count = 0
    history = []
    for epoch in range(config.profile.cls_epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        logits, _, _ = deep_logits_embeddings(model, x_val, config, mc_passes=1)
        predictions = np.argmax(logits, axis=1)
        val_f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_macro_f1": float(val_f1)})
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        scheduler.step()
        if patience_count >= config.profile.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if model_path is not None:
        torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1], "latent_dim": config.latent_dim, "n_classes": n_classes, "dropout": config.dropout}, model_path)
    return model, pd.DataFrame(history)

def enable_mc_dropout(model: nn.Module):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def deep_logits_embeddings(model: DeepClassifier, x: np.ndarray, config: RunConfig, mc_passes: int = 1):
    device = torch.device(config.device)
    loader = make_loader(x, None, config.batch_size, False, config.profile.num_workers)
    probability_passes = []
    logits_first = []
    embeddings_first = []
    start = time.perf_counter()
    for pass_index in range(mc_passes):
        enable_mc_dropout(model) if mc_passes > 1 else model.eval()
        pass_probabilities = []
        pass_logits = []
        pass_embeddings = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(device, non_blocking=True)
                logits, embeddings = model(xb)
                probs = torch.softmax(logits, dim=1)
                pass_probabilities.append(probs.cpu().numpy())
                if pass_index == 0:
                    pass_logits.append(logits.cpu().numpy())
                    pass_embeddings.append(embeddings.cpu().numpy())
        probability_passes.append(np.concatenate(pass_probabilities, axis=0))
        if pass_index == 0:
            logits_first = np.concatenate(pass_logits, axis=0)
            embeddings_first = np.concatenate(pass_embeddings, axis=0)
    elapsed = time.perf_counter() - start
    stacked = np.stack(probability_passes, axis=0)
    mean_probabilities = stacked.mean(axis=0)
    return logits_first, embeddings_first, {"probabilities": mean_probabilities, "passes": stacked, "inference_seconds": elapsed, "milliseconds_per_sample": elapsed * 1000 / max(len(x), 1)}

def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: str):
    scaler = TemperatureScaler().to(device)
    logits_tensor = torch.from_numpy(logits.astype(np.float32)).to(device)
    labels_tensor = torch.from_numpy(labels.astype(np.int64)).to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=100, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_tensor), labels_tensor)
        loss.backward()
        return loss
    optimizer.step(closure)
    return scaler.temperature

def softmax_temperature(logits: np.ndarray, temperature: float):
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)

def temperature_rescale_probabilities(probabilities: np.ndarray, temperature: float):
    powered = np.power(np.clip(probabilities, 1e-12, 1.0), 1.0 / max(temperature, 1e-6))
    return powered / powered.sum(axis=1, keepdims=True)

def predictive_uncertainty(passes: np.ndarray):
    mean_probs = passes.mean(axis=0)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)
    expected_entropy = -np.mean(np.sum(passes * np.log(passes + 1e-12), axis=2), axis=0)
    mutual_information = predictive_entropy - expected_entropy
    return predictive_entropy, mutual_information

def fit_prototypes(embeddings: np.ndarray, labels: np.ndarray, n_classes: int):
    means = []
    variances = []
    for class_index in range(n_classes):
        class_embeddings = embeddings[labels == class_index]
        means.append(class_embeddings.mean(axis=0))
        variances.append(class_embeddings.var(axis=0) + 1e-3)
    return np.stack(means), np.stack(variances)

def mahalanobis_min(embeddings: np.ndarray, means: np.ndarray, variances: np.ndarray):
    distances = []
    for mean, variance in zip(means, variances):
        distances.append(np.mean(((embeddings - mean) ** 2) / variance, axis=1))
    return np.min(np.stack(distances, axis=1), axis=1)

def robust_standardize(values: np.ndarray, reference: np.ndarray):
    center = np.median(reference)
    scale = np.percentile(reference, 75) - np.percentile(reference, 25)
    if scale <= 1e-8:
        scale = np.std(reference) + 1e-8
    return (values - center) / scale

def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidences >= edges[index]) & (confidences <= edges[index + 1])
        else:
            mask = (confidences >= edges[index]) & (confidences < edges[index + 1])
        if np.any(mask):
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)

def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray):
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, task: str):
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "ece": expected_calibration_error(y_true, probabilities),
        "brier_score": multiclass_brier(y_true, probabilities)
    }
    if task == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        metrics.update({
            "recall_detection_rate": float(tp / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "pr_auc": float(average_precision_score(y_true, probabilities[:, 1])),
            "auroc": float(roc_auc_score(y_true, probabilities[:, 1])) if len(np.unique(y_true)) > 1 else np.nan
        })
    else:
        labels = np.arange(probabilities.shape[1])
        y_binary = label_binarize(y_true, classes=labels)
        recalls = []
        fprs = []
        fnrs = []
        for class_index in labels:
            truth = (y_true == class_index).astype(int)
            predicted = (predictions == class_index).astype(int)
            tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
            recalls.append(tp / max(tp + fn, 1))
            fprs.append(fp / max(fp + tn, 1))
            fnrs.append(fn / max(fn + tp, 1))
        metrics.update({
            "recall_detection_rate": float(np.mean(recalls)),
            "false_positive_rate": float(np.mean(fprs)),
            "false_negative_rate": float(np.mean(fnrs)),
            "pr_auc": float(average_precision_score(y_binary, probabilities, average="macro")),
            "auroc": float(roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")) if len(np.unique(y_true)) == probabilities.shape[1] else np.nan
        })
    return metrics

def bootstrap_confidence_intervals(y_true: np.ndarray, probabilities: np.ndarray, task: str, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    values = {"macro_f1": [], "balanced_accuracy": [], "pr_auc": []}
    n = len(y_true)
    if n > 50000:
        selected = []
        for class_value in np.unique(y_true):
            class_indices = np.where(y_true == class_value)[0]
            target = max(50, int(round(50000 * len(class_indices) / n)))
            selected.extend(rng.choice(class_indices, min(target, len(class_indices)), replace=False).tolist())
        selected = np.array(selected, dtype=int)
        y_true = y_true[selected]
        probabilities = probabilities[selected]
        n = len(y_true)
    for _ in range(iterations):
        indices = rng.integers(0, n, n)
        yt = y_true[indices]
        if len(np.unique(yt)) < 2:
            continue
        probs = probabilities[indices]
        metrics = classification_metrics(yt, probs, task)
        for key in values:
            if np.isfinite(metrics[key]):
                values[key].append(metrics[key])
    rows = []
    for metric, samples in values.items():
        if samples:
            rows.append({"metric": metric, "mean": float(np.mean(samples)), "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5))})
    return rows

def conformal_quantile(scores: np.ndarray, alpha: float):
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))

def conformal_evaluation(calibration_probabilities: np.ndarray, calibration_labels: np.ndarray, test_probabilities: np.ndarray, test_labels: np.ndarray, alpha: float):
    nonconformity = 1.0 - calibration_probabilities[np.arange(len(calibration_labels)), calibration_labels]
    qhat = conformal_quantile(nonconformity, alpha)
    prediction_sets = test_probabilities >= (1.0 - qhat)
    sizes = prediction_sets.sum(axis=1)
    covered = prediction_sets[np.arange(len(test_labels)), test_labels]
    singleton = sizes == 1
    singleton_predictions = test_probabilities.argmax(axis=1)
    singleton_risk = float(np.mean(singleton_predictions[singleton] != test_labels[singleton])) if np.any(singleton) else np.nan
    return {
        "alpha": alpha,
        "qhat": qhat,
        "coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(sizes)),
        "singleton_coverage": float(np.mean(singleton)),
        "risk": singleton_risk
    }, sizes, prediction_sets

def risk_coverage_curve(y_true: np.ndarray, probabilities: np.ndarray, uncertainty: Optional[np.ndarray] = None):
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != y_true).astype(float)
    if uncertainty is None:
        uncertainty = 1.0 - probabilities.max(axis=1)
    order = np.argsort(uncertainty)
    ordered_errors = errors[order]
    cumulative_errors = np.cumsum(ordered_errors)
    counts = np.arange(1, len(y_true) + 1)
    risk = cumulative_errors / counts
    coverage = counts / len(y_true)
    return coverage, risk

def fpr_at_tpr(y_true_unknown: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95):
    fpr, tpr, thresholds = roc_curve(y_true_unknown, scores)
    candidates = np.where(tpr >= target_tpr)[0]
    return float(fpr[candidates[0]]) if len(candidates) else 1.0

def open_set_metrics(y_unknown: np.ndarray, scores: np.ndarray, threshold: float):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_unknown, predictions, labels=[0, 1]).ravel()
    return {
        "unknown_auroc": float(roc_auc_score(y_unknown, scores)),
        "unknown_pr_auc": float(average_precision_score(y_unknown, scores)),
        "fpr_at_95_tpr": fpr_at_tpr(y_unknown, scores),
        "unknown_recall": float(tp / max(tp + fn, 1)),
        "unknown_false_positive_rate": float(fp / max(fp + tn, 1)),
        "unknown_mcc": float(matthews_corrcoef(y_unknown, predictions)),
        "threshold": float(threshold)
    }

def fit_open_reference(calibration_logits: np.ndarray, calibration_embeddings: np.ndarray, calibration_probabilities: np.ndarray, calibration_labels: np.ndarray, means: np.ndarray, variances: np.ndarray, alpha_open: float):
    energy = -np.log(np.exp(calibration_logits - calibration_logits.max(axis=1, keepdims=True)).sum(axis=1) + 1e-12) - calibration_logits.max(axis=1)
    distance = mahalanobis_min(calibration_embeddings, means, variances)
    entropy = -np.sum(calibration_probabilities * np.log(calibration_probabilities + 1e-12), axis=1)
    msp = 1.0 - calibration_probabilities.max(axis=1)
    reference = {"energy": energy, "distance": distance, "entropy": entropy, "msp": msp}
    composite = np.mean(np.column_stack([robust_standardize(reference[key], reference[key]) for key in ["energy", "distance", "entropy", "msp"]]), axis=1)
    reference["composite"] = composite
    thresholds = {key: float(np.quantile(value, 1.0 - alpha_open)) for key, value in reference.items()}
    return reference, thresholds

def calculate_open_scores(logits: np.ndarray, embeddings: np.ndarray, probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray, reference: Dict[str, np.ndarray]):
    energy = -np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1) + 1e-12) - logits.max(axis=1)
    distance = mahalanobis_min(embeddings, means, variances)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    msp = 1.0 - probabilities.max(axis=1)
    raw = {"energy": energy, "mahalanobis": distance, "entropy": entropy, "msp": msp}
    composite = np.mean(np.column_stack([
        robust_standardize(energy, reference["energy"]),
        robust_standardize(distance, reference["distance"]),
        robust_standardize(entropy, reference["entropy"]),
        robust_standardize(msp, reference["msp"])
    ]), axis=1)
    raw["composite"] = composite
    return raw

def build_baselines(task: str, n_classes: int, profile: Profile, seed: int):
    worker_count = CPU_WORKER_LIMIT
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=worker_count, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=profile.baseline_estimators, class_weight="balanced_subsample", max_features="sqrt", min_samples_leaf=2, n_jobs=worker_count, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=profile.baseline_estimators, class_weight="balanced", max_features="sqrt", min_samples_leaf=2, n_jobs=worker_count, random_state=seed),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(max_iter=profile.baseline_estimators, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.1, random_state=seed)
    }
    if HAS_XGBOOST:
        objective = "binary:logistic" if task == "binary" else "multi:softprob"
        eval_metric = "aucpr" if task == "binary" else "mlogloss"
        params = {
            "n_estimators": profile.baseline_estimators,
            "max_depth": 8,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": objective,
            "eval_metric": eval_metric,
            "tree_method": "hist",
            "n_jobs": worker_count,
            "random_state": seed
        }
        if task != "binary":
            params["num_class"] = n_classes
        models["XGBoost"] = XGBClassifier(**params)
    return models

def model_probabilities(model, x: np.ndarray, n_classes: int):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
    else:
        decision = model.decision_function(x)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        decision = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(decision)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    if probabilities.shape[1] != n_classes:
        expanded = np.zeros((len(x), n_classes), dtype=float)
        classes = getattr(model, "classes_", np.arange(probabilities.shape[1])).astype(int)
        expanded[:, classes] = probabilities
        probabilities = expanded
    return probabilities

def train_evaluate_baseline(model_name: str, model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, task: str, n_classes: int):
    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    with MemoryMonitor() as monitor:
        start_train = time.perf_counter()
        if isinstance(model, HistGradientBoostingClassifier) or model_name == "XGBoost":
            weights = class_weights(y_train, n_classes)[y_train]
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start_train
        start_inference = time.perf_counter()
        probabilities = model_probabilities(model, x_test, n_classes)
        inference_seconds = time.perf_counter() - start_inference
    metrics = classification_metrics(y_test, probabilities, task)
    metrics.update({
        "model": model_name,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": inference_seconds * 1000 / max(len(x_test), 1),
        "peak_memory_mb": monitor.peak_mb,
        "memory_delta_mb": max(0.0, monitor.peak_mb - start_memory)
    })
    return model, probabilities, metrics

def per_class_rows(y_true: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str], context: Dict[str, Any]):
    predictions = probabilities.argmax(axis=1)
    report = classification_report(y_true, predictions, labels=np.arange(len(class_names)), target_names=list(class_names), output_dict=True, zero_division=0)
    rows = []
    for class_name in class_names:
        values = report[class_name]
        row = dict(context)
        row.update({"class": class_name, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": values["support"]})
        rows.append(row)
    return rows

def create_output_tree(root: Path):
    subdirs = {}
    for name in ["figures", "models", "logs", "cache", "predictions", "excel"]:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs

def save_json(data: Any, path: Path):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    path.write_text(json.dumps(convert(data), ensure_ascii=False, indent=2), encoding="utf-8")

def apply_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "path.simplify": True,
        "path.simplify_threshold": 0.8,
        "agg.path.chunksize": 10000
    })

def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300, pil_kwargs={"compress_level": 3})
    plt.close(fig)

def plot_class_distribution(frames: Dict[str, pd.DataFrame], figures_dir: Path):
    labels = sorted(set().union(*[set(frame["__attack_label__"].unique()) for frame in frames.values()]))
    x = np.arange(len(labels))
    width = 0.8 / max(len(frames), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 6))
    for index, (name, frame) in enumerate(frames.items()):
        counts = frame["__attack_label__"].value_counts().reindex(labels, fill_value=0)
        ax.bar(x + (index - (len(frames) - 1) / 2) * width, counts.values, width, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Sample count on logarithmic scale")
    ax.set_title("Attack-class distribution across datasets")
    ax.legend()
    save_figure(fig, figures_dir / "01_dataset_class_distribution.png")

def plot_model_comparison(results: pd.DataFrame, figures_dir: Path):
    if results.empty:
        return
    metrics = ["macro_f1", "pr_auc", "balanced_accuracy", "mcc"]
    available = [m for m in metrics if m in results.columns]
    grouped = results.groupby("model")[available].mean().sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grouped.index))
    width = 0.8 / max(len(available), 1)
    for index, metric in enumerate(available):
        ax.bar(x + (index - (len(available) - 1) / 2) * width, grouped[metric], width, label=metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=30, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title("Aggregate model performance across evaluation scenarios")
    ax.legend(ncol=2)
    save_figure(fig, figures_dir / "02_model_comparison.png")

def plot_confusion(y_true: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str], title: str, path: Path):
    matrix = confusion_matrix(y_true, probabilities.argmax(axis=1), labels=np.arange(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.6)))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, path)

def plot_binary_curves(y_true: np.ndarray, probabilities: np.ndarray, title_prefix: str, figures_dir: Path, stem: str):
    fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, probabilities[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUROC = {roc_auc_score(y_true, probabilities[:, 1]):.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title_prefix} ROC curve")
    ax.legend()
    save_figure(fig, figures_dir / f"{stem}_roc.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {average_precision_score(y_true, probabilities[:, 1]):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} precision-recall curve")
    ax.legend()
    save_figure(fig, figures_dir / f"{stem}_pr.png")

def plot_calibration(y_true: np.ndarray, probabilities_before: np.ndarray, probabilities_after: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for probabilities, name in [(probabilities_before, "Before temperature scaling"), (probabilities_after, "After temperature scaling")]:
        if probabilities.shape[1] == 2:
            fraction, mean = calibration_curve(y_true, probabilities[:, 1], n_bins=15, strategy="quantile")
            ax.plot(mean, fraction, marker="o", label=name)
        else:
            confidence = probabilities.max(axis=1)
            correct = (probabilities.argmax(axis=1) == y_true).astype(int)
            bins = pd.qcut(confidence, q=min(15, len(np.unique(confidence))), duplicates="drop")
            grouped = pd.DataFrame({"confidence": confidence, "correct": correct, "bin": bins}).groupby("bin", observed=True).mean()
            ax.plot(grouped["confidence"], grouped["correct"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    save_figure(fig, path)

def plot_risk_coverage(y_true: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray, path: Path):
    coverage, risk = risk_coverage_curve(y_true, probabilities, uncertainty)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(coverage, risk, linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk-coverage curve for abstention")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, min(1.0, np.nanmax(risk) * 1.1)))
    save_figure(fig, path)

def plot_open_set_scores(y_unknown: np.ndarray, scores: Dict[str, np.ndarray], figures_dir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = list(scores.keys())
    positions = []
    data = []
    labels = []
    pos = 1
    for method in methods:
        known = scores[method][y_unknown == 0]
        unknown = scores[method][y_unknown == 1]
        data.extend([known, unknown])
        positions.extend([pos, pos + 1])
        labels.extend([f"{method}\nKnown", f"{method}\nUnknown"])
        pos += 3
    ax.boxplot(data, positions=positions, showfliers=False, widths=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Unknownness score")
    ax.set_title("Open-set score separation")
    save_figure(fig, figures_dir / f"{stem}_score_distributions.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, values in scores.items():
        fpr, tpr, _ = roc_curve(y_unknown, values)
        auc = roc_auc_score(y_unknown, values)
        ax.plot(fpr, tpr, linewidth=2, label=f"{method}: {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Unknown detection rate")
    ax.set_title("Open-set ROC comparison")
    ax.legend()
    save_figure(fig, figures_dir / f"{stem}_unknown_roc.png")

def plot_training_history(ssl_history: pd.DataFrame, cls_history: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if not ssl_history.empty:
        ax.plot(ssl_history["epoch"], ssl_history["validation_loss"], label="SSL validation reconstruction loss")
    if not cls_history.empty:
        ax2 = ax.twinx()
        ax2.plot(cls_history["epoch"], cls_history["validation_macro_f1"], label="Classifier validation macro F1", linestyle="--")
        ax2.set_ylabel("Validation macro F1")
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSL loss")
    ax.set_title("Representation pretraining and supervised fine-tuning")
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    save_figure(fig, path)

def plot_embedding(embeddings: np.ndarray, labels: np.ndarray, class_names: Sequence[str], path: Path, max_points: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_points:
        indices = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
        reduced = reducer.fit_transform(embeddings)
        title = "UMAP projection of learned traffic representations"
    else:
        centered = embeddings - embeddings.mean(axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        reduced = centered @ vt[:2].T
        title = "PCA projection of learned traffic representations"
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=8, alpha=0.55, label=class_name)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, path)

def plot_drift_timeline(drift_frame: pd.DataFrame, path: Path):
    if drift_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drift_frame["window"], drift_frame["macro_f1"], marker="o", label="Macro F1")
    ax.plot(drift_frame["window"], drift_frame["ece"], marker="s", label="ECE")
    ax.plot(drift_frame["window"], drift_frame["mean_entropy"], marker="^", label="Predictive entropy")
    detections = drift_frame[drift_frame["drift_detected"] == 1]
    for value in detections["window"]:
        ax.axvline(value, linestyle="--", alpha=0.7)
    ax.set_xlabel("Stream window")
    ax.set_ylabel("Metric value")
    ax.set_title("Temporal performance, uncertainty and detected concept drift")
    ax.legend(ncol=3)
    save_figure(fig, path)

def plot_psi_heatmap(psi_frame: pd.DataFrame, path: Path):
    if psi_frame.empty:
        return
    pivot = psi_frame.pivot(index="feature", columns="window", values="psi").fillna(0.0)
    top = pivot.mean(axis=1).sort_values(ascending=False).head(20).index
    pivot = pivot.loc[top]
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_xlabel("Stream window")
    ax.set_ylabel("Feature")
    ax.set_title("Population Stability Index across temporal windows")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, path)

def plot_adaptation(adaptation_frame: pd.DataFrame, path: Path):
    if adaptation_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = adaptation_frame.groupby("budget")["post_adaptation_macro_f1"].mean().sort_index()
    ax.plot(grouped.index, grouped.values, marker="o", linewidth=2)
    if "target_macro_f1" in adaptation_frame.columns:
        ax.axhline(adaptation_frame["target_macro_f1"].mean(), linestyle="--", label="Recovery target")
        ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Labelled adaptation samples")
    ax.set_ylabel("Post-adaptation macro F1")
    ax.set_title("Sample efficiency of drift adaptation")
    save_figure(fig, path)

def plot_ablation(ablation_frame: pd.DataFrame, path: Path):
    if ablation_frame.empty:
        return
    columns = [c for c in ["macro_f1", "unknown_auroc", "ece", "coverage"] if c in ablation_frame.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ablation_frame))
    width = 0.8 / max(len(columns), 1)
    for index, column in enumerate(columns):
        values = ablation_frame[column].fillna(0.0)
        if column == "ece":
            values = 1.0 - values.clip(0, 1)
        ax.bar(x + (index - (len(columns) - 1) / 2) * width, values, width, label="1-ECE" if column == "ece" else column.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_frame["variant"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation analysis of the proposed framework")
    ax.legend()
    save_figure(fig, path)

def plot_average_ranks(results: pd.DataFrame, path: Path):
    if results.empty or "macro_f1" not in results.columns:
        return
    pivot = results.pivot_table(index=["seed", "scenario", "dataset", "task"], columns="model", values="macro_f1", aggfunc="mean").dropna(axis=0, how="any")
    if len(pivot) < 2 or pivot.shape[1] < 2:
        return
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean(axis=0).sort_values()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(ranks.values, np.zeros(len(ranks)), s=80)
    for index, (model, rank) in enumerate(ranks.items()):
        ax.text(rank, 0.02 + 0.04 * (index % 2), model, rotation=25, ha="center")
    ax.set_xlim(0.5, max(ranks.max() + 0.5, 2))
    ax.set_yticks([])
    ax.set_xlabel("Average rank, lower is better")
    ax.set_title("Average model ranks across matched experiments")
    save_figure(fig, path)


def panel_label(ax, label: str):
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

def draw_confusion_panel(ax, y_true: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str], title: str):
    matrix = confusion_matrix(y_true, probabilities.argmax(axis=1), labels=np.arange(len(class_names)), normalize="true")
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = 0.50
    font_size = 7 if len(class_names) > 6 else 8
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > threshold else "black", fontsize=font_size)
    return image

def draw_roc_panel(ax, y_true: np.ndarray, probabilities: np.ndarray, title: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
        auc = roc_auc_score(y_true, probabilities[:, 1])
        if len(fpr) > 2000:
            indices = np.linspace(0, len(fpr) - 1, 2000, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
        ax.plot(fpr, tpr, linewidth=2.2, label=f"AUROC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Chance")
        ax.legend(loc="lower right")
    except Exception:
        ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)

def draw_pr_panel(ax, y_true: np.ndarray, probabilities: np.ndarray, title: str):
    try:
        precision, recall, _ = precision_recall_curve(y_true, probabilities[:, 1])
        score = average_precision_score(y_true, probabilities[:, 1])
        prevalence = float(np.mean(y_true == 1))
        if len(recall) > 2000:
            indices = np.linspace(0, len(recall) - 1, 2000, dtype=int)
            recall = recall[indices]
            precision = precision[indices]
        ax.plot(recall, precision, linewidth=2.2, label=f"PR-AUC = {score:.4f}")
        ax.axhline(prevalence, linestyle="--", linewidth=1.0, label=f"Prevalence = {prevalence:.3f}")
        ax.legend(loc="lower left")
    except Exception:
        ax.text(0.5, 0.5, "PR curve unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)

def draw_calibration_panel(ax, y_true: np.ndarray, probabilities_before: np.ndarray, probabilities_after: np.ndarray, title: str):
    for probabilities, name, marker in [(probabilities_before, "Before scaling", "o"), (probabilities_after, "After scaling", "s")]:
        try:
            if probabilities.shape[1] == 2:
                fraction, mean = calibration_curve(y_true, probabilities[:, 1], n_bins=12, strategy="quantile")
                ax.plot(mean, fraction, marker=marker, linewidth=1.8, markersize=4, label=name)
            else:
                confidence = probabilities.max(axis=1)
                correct = (probabilities.argmax(axis=1) == y_true).astype(int)
                bins = pd.qcut(confidence, q=min(12, max(2, len(np.unique(confidence)))), duplicates="drop")
                grouped = pd.DataFrame({"confidence": confidence, "correct": correct, "bin": bins}).groupby("bin", observed=True).mean()
                ax.plot(grouped["confidence"], grouped["correct"], marker=marker, linewidth=1.8, markersize=4, label=name)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")

def draw_risk_coverage_panel(ax, y_true: np.ndarray, probabilities: np.ndarray, uncertainty: np.ndarray, title: str):
    coverage, risk = risk_coverage_curve(y_true, probabilities, uncertainty)
    if len(coverage) > 2000:
        indices = np.linspace(0, len(coverage) - 1, 2000, dtype=int)
        coverage = coverage[indices]
        risk = risk[indices]
    ax.plot(coverage, risk, linewidth=2.2)
    ax.set_xlim(0, 1)
    finite = risk[np.isfinite(risk)]
    upper = max(0.05, min(1.0, float(np.max(finite)) * 1.12)) if len(finite) else 1.0
    ax.set_ylim(0, upper)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title(title)

def compact_figure_payload(y_true: np.ndarray, probabilities: np.ndarray, probabilities_before: Optional[np.ndarray], uncertainty: Optional[np.ndarray], class_names: Sequence[str], dataset: str, task: str, scenario: str, seed: int, max_points: int = 80000):
    rng = np.random.default_rng(seed)
    if len(y_true) > max_points:
        indices = rng.choice(len(y_true), max_points, replace=False)
    else:
        indices = np.arange(len(y_true))
    return {
        "dataset": dataset,
        "task": task,
        "scenario": scenario,
        "class_names": list(class_names),
        "y_true": np.asarray(y_true)[indices].copy(),
        "probabilities": np.asarray(probabilities)[indices].copy(),
        "probabilities_before": None if probabilities_before is None else np.asarray(probabilities_before)[indices].copy(),
        "uncertainty": None if uncertainty is None else np.asarray(uncertainty)[indices].copy()
    }

def plot_within_dataset_comparison(payloads: Sequence[Dict[str, Any]], task: str, path: Path):
    selected = [item for item in payloads if item["task"] == task and item["scenario"] == "within_dataset"]
    selected = sorted(selected, key=lambda item: item["dataset"])
    if not selected:
        return
    if task == "binary":
        fig, axes = plt.subplots(len(selected), 4, figsize=(18, 4.8 * len(selected)), squeeze=False)
        for row_index, item in enumerate(selected):
            y_true = item["y_true"]
            probabilities = item["probabilities"]
            before = item["probabilities_before"] if item["probabilities_before"] is not None else probabilities
            uncertainty = item["uncertainty"] if item["uncertainty"] is not None else -probabilities.max(axis=1)
            image = draw_confusion_panel(axes[row_index, 0], y_true, probabilities, item["class_names"], f"{item['dataset']} confusion matrix")
            fig.colorbar(image, ax=axes[row_index, 0], fraction=0.046, pad=0.04)
            draw_roc_panel(axes[row_index, 1], y_true, probabilities, f"{item['dataset']} ROC")
            draw_pr_panel(axes[row_index, 2], y_true, probabilities, f"{item['dataset']} precision-recall")
            draw_calibration_panel(axes[row_index, 3], y_true, before, probabilities, f"{item['dataset']} calibration")
            for col_index in range(4):
                panel_label(axes[row_index, col_index], chr(65 + row_index * 4 + col_index))
        fig.suptitle("Within-dataset binary detection: discrimination and calibration", fontsize=16, fontweight="bold", y=1.01)
    else:
        fig, axes = plt.subplots(len(selected), 3, figsize=(16, 5.0 * len(selected)), squeeze=False)
        for row_index, item in enumerate(selected):
            y_true = item["y_true"]
            probabilities = item["probabilities"]
            before = item["probabilities_before"] if item["probabilities_before"] is not None else probabilities
            uncertainty = item["uncertainty"] if item["uncertainty"] is not None else -probabilities.max(axis=1)
            image = draw_confusion_panel(axes[row_index, 0], y_true, probabilities, item["class_names"], f"{item['dataset']} confusion matrix")
            fig.colorbar(image, ax=axes[row_index, 0], fraction=0.046, pad=0.04)
            draw_calibration_panel(axes[row_index, 1], y_true, before, probabilities, f"{item['dataset']} calibration")
            draw_risk_coverage_panel(axes[row_index, 2], y_true, probabilities, uncertainty, f"{item['dataset']} abstention")
            for col_index in range(3):
                panel_label(axes[row_index, col_index], chr(65 + row_index * 3 + col_index))
        fig.suptitle("Within-dataset multiclass detection: classification, calibration and abstention", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, path)

def plot_cross_dataset_comparison(payloads: Sequence[Dict[str, Any]], task: str, path: Path):
    selected = [item for item in payloads if item["task"] == task and item["scenario"] == "cross_dataset"]
    selected = sorted(selected, key=lambda item: item["dataset"])
    if not selected:
        return
    if task == "binary":
        fig, axes = plt.subplots(len(selected), 3, figsize=(16, 5.0 * len(selected)), squeeze=False)
        for row_index, item in enumerate(selected):
            y_true = item["y_true"]
            probabilities = item["probabilities"]
            image = draw_confusion_panel(axes[row_index, 0], y_true, probabilities, item["class_names"], f"{item['dataset']} confusion matrix")
            fig.colorbar(image, ax=axes[row_index, 0], fraction=0.046, pad=0.04)
            draw_roc_panel(axes[row_index, 1], y_true, probabilities, f"{item['dataset']} ROC")
            draw_pr_panel(axes[row_index, 2], y_true, probabilities, f"{item['dataset']} precision-recall")
            for col_index in range(3):
                panel_label(axes[row_index, col_index], chr(65 + row_index * 3 + col_index))
        fig.suptitle("Cross-dataset binary generalization in both transfer directions", fontsize=16, fontweight="bold", y=1.01)
    else:
        fig, axes = plt.subplots(len(selected), 2, figsize=(14, 5.2 * len(selected)), squeeze=False)
        for row_index, item in enumerate(selected):
            y_true = item["y_true"]
            probabilities = item["probabilities"]
            class_names = item["class_names"]
            image = draw_confusion_panel(axes[row_index, 0], y_true, probabilities, class_names, f"{item['dataset']} confusion matrix")
            fig.colorbar(image, ax=axes[row_index, 0], fraction=0.046, pad=0.04)
            matrix = confusion_matrix(y_true, probabilities.argmax(axis=1), labels=np.arange(len(class_names)), normalize="true")
            recall_values = np.diag(matrix)
            order = np.argsort(recall_values)
            axes[row_index, 1].barh(np.asarray(class_names)[order], recall_values[order])
            axes[row_index, 1].set_xlim(0, 1)
            axes[row_index, 1].set_xlabel("Per-class recall")
            axes[row_index, 1].set_title(f"{item['dataset']} class-level transfer performance")
            panel_label(axes[row_index, 0], chr(65 + row_index * 2))
            panel_label(axes[row_index, 1], chr(66 + row_index * 2))
        fig.suptitle("Cross-dataset multiclass generalization in both transfer directions", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, path)

def plot_open_set_dashboard(y_unknown: np.ndarray, scores: Dict[str, np.ndarray], path: Path, holdout: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    methods = list(scores.keys())
    positions = []
    data = []
    labels = []
    position = 1
    for method in methods:
        known = scores[method][y_unknown == 0]
        unknown = scores[method][y_unknown == 1]
        data.extend([known, unknown])
        positions.extend([position, position + 1])
        labels.extend([f"{method}\nKnown", f"{method}\nUnknown"])
        position += 3
    axes[0].boxplot(data, positions=positions, showfliers=False, widths=0.70, patch_artist=True)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("Unknownness score")
    axes[0].set_title("Known-versus-unknown score separation")
    for method, values in scores.items():
        try:
            fpr, tpr, _ = roc_curve(y_unknown, values)
            auc = roc_auc_score(y_unknown, values)
            if len(fpr) > 2000:
                indices = np.linspace(0, len(fpr) - 1, 2000, dtype=int)
                fpr = fpr[indices]
                tpr = tpr[indices]
            axes[1].plot(fpr, tpr, linewidth=2, label=f"{method}: {auc:.4f}")
        except Exception:
            pass
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("Unknown detection rate")
    axes[1].set_title("Open-set ROC comparison")
    axes[1].legend(loc="lower right")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.suptitle(f"Open-set detection with held-out attack family: {holdout}", fontsize=15, fontweight="bold", y=1.01)
    save_figure(fig, path)

def plot_drift_dashboard(drift_frame: pd.DataFrame, psi_frame: pd.DataFrame, adaptation_frame: pd.DataFrame, path: Path):
    if drift_frame.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for stream, group in drift_frame.groupby("stream"):
        grouped = group.groupby("window", as_index=False)[["macro_f1", "ece", "mean_entropy"]].mean()
        axes[0, 0].plot(grouped["window"], grouped["macro_f1"], marker="o", linewidth=2, label=f"{stream.title()} Macro F1")
        axes[0, 1].plot(grouped["window"], grouped["ece"], marker="s", linewidth=1.8, label=f"{stream.title()} ECE")
        axes[0, 1].plot(grouped["window"], grouped["mean_entropy"], marker="^", linewidth=1.8, linestyle="--", label=f"{stream.title()} entropy")
    detections = drift_frame[drift_frame["drift_detected"] == 1]
    for value in sorted(detections["window"].unique()):
        axes[0, 0].axvline(value, linestyle="--", alpha=0.45)
        axes[0, 1].axvline(value, linestyle="--", alpha=0.45)
    axes[0, 0].set_xlabel("Stream window")
    axes[0, 0].set_ylabel("Macro F1")
    axes[0, 0].set_title("Performance under temporal and controlled drift")
    axes[0, 0].legend()
    axes[0, 1].set_xlabel("Stream window")
    axes[0, 1].set_ylabel("Metric value")
    axes[0, 1].set_title("Calibration error and predictive uncertainty")
    axes[0, 1].legend(ncol=2)
    if not psi_frame.empty:
        pivot = psi_frame.pivot_table(index="feature", columns="window", values="psi", aggfunc="mean").fillna(0.0)
        top = pivot.mean(axis=1).sort_values(ascending=False).head(15).index
        pivot = pivot.loc[top]
        image = axes[1, 0].imshow(pivot.values, aspect="auto", cmap="magma")
        axes[1, 0].set_yticks(np.arange(len(pivot.index)))
        axes[1, 0].set_yticklabels(pivot.index, fontsize=8)
        axes[1, 0].set_xticks(np.arange(len(pivot.columns)))
        axes[1, 0].set_xticklabels(pivot.columns, rotation=45, ha="right")
        axes[1, 0].set_xlabel("Stream window")
        axes[1, 0].set_ylabel("Feature")
        axes[1, 0].set_title("Top feature distribution shifts by PSI")
        fig.colorbar(image, ax=axes[1, 0], fraction=0.030, pad=0.02)
    else:
        axes[1, 0].text(0.5, 0.5, "PSI unavailable", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Feature distribution shift")
    if not adaptation_frame.empty:
        grouped = adaptation_frame.groupby("budget")["post_adaptation_macro_f1"].mean().sort_index()
        axes[1, 1].plot(grouped.index, grouped.values, marker="o", linewidth=2.2, label="Post-adaptation Macro F1")
        if "pre_adaptation_macro_f1" in adaptation_frame.columns:
            pre = adaptation_frame.groupby("budget")["pre_adaptation_macro_f1"].mean().sort_index()
            axes[1, 1].plot(pre.index, pre.values, marker="s", linestyle="--", linewidth=1.6, label="Before adaptation")
        if "target_macro_f1" in adaptation_frame.columns:
            axes[1, 1].axhline(adaptation_frame["target_macro_f1"].mean(), linestyle=":", linewidth=2, label="Recovery target")
        axes[1, 1].set_xscale("log", base=2)
        axes[1, 1].set_xlabel("Labelled adaptation samples")
        axes[1, 1].set_ylabel("Macro F1")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "Adaptation results unavailable", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Adaptation sample efficiency")
    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(65 + index))
    fig.suptitle("Concept drift detection, uncertainty monitoring and adaptation", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, path)

def plot_summary_dashboard(results: pd.DataFrame, ablation_frame: pd.DataFrame, robustness_frame: pd.DataFrame, path: Path):
    if results.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    available = [column for column in ["macro_f1", "pr_auc", "balanced_accuracy", "mcc"] if column in results.columns]
    grouped = results.groupby("model")[available].mean()
    if "macro_f1" in grouped.columns:
        grouped = grouped.sort_values("macro_f1", ascending=False)
    x = np.arange(len(grouped))
    width = 0.82 / max(1, len(available))
    for index, metric in enumerate(available):
        axes[0, 0].bar(x + (index - (len(available) - 1) / 2) * width, grouped[metric], width, label=metric.replace("_", " ").title())
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(grouped.index, rotation=25, ha="right")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Mean score")
    axes[0, 0].set_title("Aggregate performance across all scenarios")
    axes[0, 0].legend(ncol=2)
    proposed = results[results["model"] == "Proposed SSL-OpenSet"].copy()
    if not proposed.empty:
        scenario = proposed.groupby(["task", "scenario"])["macro_f1"].mean().unstack("scenario")
        scenario.plot(kind="bar", ax=axes[0, 1], width=0.72)
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].set_xlabel("Task")
        axes[0, 1].set_ylabel("Mean Macro F1")
        axes[0, 1].set_title("Within-dataset versus cross-dataset generalization")
        axes[0, 1].tick_params(axis="x", rotation=0)
        axes[0, 1].legend(title="Scenario")
    else:
        axes[0, 1].text(0.5, 0.5, "Proposed-model results unavailable", ha="center", va="center", transform=axes[0, 1].transAxes)
    if not ablation_frame.empty:
        ablation = ablation_frame.groupby("variant", as_index=False).agg(macro_f1=("macro_f1", "mean"), ece=("ece", "mean"), unknown_auroc=("unknown_auroc", "mean"))
        ablation["calibration_quality"] = 1.0 - ablation["ece"].clip(0, 1)
        ablation = ablation.sort_values("macro_f1", ascending=True).tail(12)
        y = np.arange(len(ablation))
        axes[1, 0].barh(y - 0.18, ablation["macro_f1"], height=0.34, label="Macro F1")
        axes[1, 0].barh(y + 0.18, ablation["calibration_quality"], height=0.34, label="1 - ECE")
        axes[1, 0].set_yticks(y)
        axes[1, 0].set_yticklabels(ablation["variant"], fontsize=8)
        axes[1, 0].set_xlim(0, 1.05)
        axes[1, 0].set_xlabel("Score")
        axes[1, 0].set_title("Ablation of representation learning and calibration")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "Ablation results unavailable", ha="center", va="center", transform=axes[1, 0].transAxes)
    if not robustness_frame.empty:
        for perturbation, group in robustness_frame.groupby("perturbation"):
            group_mean = group.groupby("level")["macro_f1"].mean().sort_index()
            axes[1, 1].plot(group_mean.index, group_mean.values, marker="o", linewidth=2, label=perturbation.replace("_", " ").title())
        axes[1, 1].set_xlabel("Perturbation level")
        axes[1, 1].set_ylabel("Macro F1")
        axes[1, 1].set_title("Robustness under bounded feature perturbations")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "Robustness results unavailable", ha="center", va="center", transform=axes[1, 1].transAxes)
    for index, ax in enumerate(axes.flat):
        panel_label(ax, chr(65 + index))
    fig.suptitle("Comparative summary of model performance, generalization and robustness", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, path)

def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10):
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)
    ref_ratio = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_ratio = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))

def adapt_classifier(model: DeepClassifier, x: np.ndarray, y: np.ndarray, config: RunConfig, epochs: int = 3):
    adapted = copy.deepcopy(model).to(config.device)
    adapted.train()
    optimizer = torch.optim.AdamW(adapted.parameters(), lr=config.cls_lr * 0.1, weight_decay=config.weight_decay)
    weights = torch.from_numpy(class_weights(y, len(np.unique(y)) if len(np.unique(y)) > 1 else 2)).to(config.device)
    if len(weights) != adapted.head[-1].out_features:
        weights = torch.ones(adapted.head[-1].out_features, device=config.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = make_loader(x, y, min(config.batch_size, max(32, len(x))), True, 0)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = adapted(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            optimizer.step()
    return adapted

def generate_synthetic_drift_stream(x: np.ndarray, y: np.ndarray, attack_labels: np.ndarray, window_size: int, seed: int):
    rng = np.random.default_rng(seed)
    phases = [
        {"name": "benign_dominant", "attack_ratio": 0.10, "family": None},
        {"name": "balanced", "attack_ratio": 0.50, "family": None},
        {"name": "attack_dominant", "attack_ratio": 0.90, "family": None},
        {"name": "family_shift", "attack_ratio": 0.75, "family": None}
    ]
    attack_families = [name for name in np.unique(attack_labels) if name != "Benign"]
    if attack_families:
        phases[-1]["family"] = attack_families[-1]
    benign_indices = np.where(y == 0)[0]
    attack_indices = np.where(y == 1)[0]
    stream_indices = []
    boundaries = []
    phase_names = []
    for phase_index, phase in enumerate(phases):
        attack_count = int(window_size * phase["attack_ratio"] * 3)
        benign_count = window_size * 3 - attack_count
        if phase["family"] is not None:
            family_indices = np.where(attack_labels == phase["family"])[0]
            selected_attack_pool = family_indices if len(family_indices) else attack_indices
        else:
            selected_attack_pool = attack_indices
        selected_benign = rng.choice(benign_indices, benign_count, replace=len(benign_indices) < benign_count)
        selected_attack = rng.choice(selected_attack_pool, attack_count, replace=len(selected_attack_pool) < attack_count)
        selected = np.concatenate([selected_benign, selected_attack])
        rng.shuffle(selected)
        stream_indices.extend(selected.tolist())
        phase_names.extend([phase["name"]] * len(selected))
        if phase_index > 0:
            boundaries.append(len(stream_indices) - len(selected))
    return x[stream_indices], y[stream_indices], attack_labels[stream_indices], np.array(phase_names), boundaries

def run_drift_analysis(frame: pd.DataFrame, features: Sequence[str], timestamp_columns: Sequence[str], config: RunConfig, output_dirs: Dict[str, Path], result_store: Dict[str, List[Dict[str, Any]]], seed: int):
    working = frame.copy()
    timestamp_col = timestamp_columns[0] if timestamp_columns else None
    if timestamp_col is not None and timestamp_col in working.columns:
        numeric_time = pd.to_numeric(working[timestamp_col], errors="coerce")
        if numeric_time.notna().mean() > 0.8:
            working["__time_order__"] = numeric_time
        else:
            parsed = pd.to_datetime(working[timestamp_col], errors="coerce", utc=True)
            working["__time_order__"] = parsed.astype("int64")
        working = working.sort_values("__time_order__").reset_index(drop=True)
        temporal_mode = "natural_timestamp"
    else:
        working = working.reset_index(drop=True)
        temporal_mode = "row_order"
    n = len(working)
    train_end = int(n * 0.5)
    calibration_end = int(n * 0.6)
    train_frame = working.iloc[:train_end].copy()
    calibration_frame = working.iloc[train_end:calibration_end].copy()
    stream_frame = working.iloc[calibration_end:].copy()
    preprocessor = NetworkPreprocessor()
    x_train = preprocessor.fit_transform(train_frame[list(features)])
    x_calibration = preprocessor.transform(calibration_frame[list(features)])
    x_stream_natural = preprocessor.transform(stream_frame[list(features)])
    y_train = train_frame["__binary_label__"].to_numpy(dtype=int)
    y_calibration = calibration_frame["__binary_label__"].to_numpy(dtype=int)
    y_stream_natural = stream_frame["__binary_label__"].to_numpy(dtype=int)
    attack_stream_natural = stream_frame["__attack_label__"].to_numpy()
    train_internal, val_internal = train_test_split(np.arange(len(x_train)), test_size=0.15, stratify=y_train, random_state=seed)
    ssl_model, ssl_history = pretrain_ssl(x_train[train_internal], x_train[val_internal], config, seed)
    model, cls_history = train_deep_classifier(x_train[train_internal], y_train[train_internal], x_train[val_internal], y_train[val_internal], ssl_model, 2, config, seed)
    calibration_logits, _, calibration_output = deep_logits_embeddings(model, x_calibration, config, mc_passes=1)
    temperature = fit_temperature(calibration_logits, y_calibration, config.device)
    natural_logits, _, natural_output = deep_logits_embeddings(model, x_stream_natural, config, mc_passes=1)
    natural_probabilities = softmax_temperature(natural_logits, temperature)
    natural_entropy = -np.sum(natural_probabilities * np.log(natural_probabilities + 1e-12), axis=1)
    window_size = min(config.profile.drift_window, max(500, len(x_stream_natural) // 10))
    ph = PageHinkley(delta=0.002, threshold=15.0, alpha=0.995)
    drift_rows = []
    psi_rows = []
    reference_values = x_train[-min(len(x_train), window_size * 2):]
    for window_index, start in enumerate(range(0, len(x_stream_natural), window_size)):
        end = min(start + window_size, len(x_stream_natural))
        if end - start < 100:
            continue
        yt = y_stream_natural[start:end]
        probs = natural_probabilities[start:end]
        metrics = classification_metrics(yt, probs, "binary")
        errors = (probs.argmax(axis=1) != yt).astype(float)
        detected = any(ph.update(float(value)) for value in errors)
        row = {"seed": seed, "mode": temporal_mode, "stream": "natural", "window": window_index, "start_index": start, "end_index": end, "drift_detected": int(detected), "mean_entropy": float(natural_entropy[start:end].mean()), "dominant_attack": pd.Series(attack_stream_natural[start:end]).value_counts().index[0]}
        row.update(metrics)
        drift_rows.append(row)
        current = x_stream_natural[start:end]
        for feature_index, feature in enumerate(features):
            psi_rows.append({"seed": seed, "stream": "natural", "window": window_index, "feature": feature, "psi": population_stability_index(reference_values[:, feature_index], current[:, feature_index])})
    x_synthetic, y_synthetic, attack_synthetic, phase_names, boundaries = generate_synthetic_drift_stream(x_stream_natural, y_stream_natural, attack_stream_natural, window_size, seed)
    synthetic_logits, _, _ = deep_logits_embeddings(model, x_synthetic, config, mc_passes=1)
    synthetic_probabilities = softmax_temperature(synthetic_logits, temperature)
    ph_synthetic = PageHinkley(delta=0.001, threshold=10.0, alpha=0.99)
    first_detection = None
    adaptation_rows = []
    baseline_reference = classification_metrics(y_calibration, softmax_temperature(calibration_logits, temperature), "binary")["macro_f1"]
    recovery_target = baseline_reference * 0.95
    for window_index, start in enumerate(range(0, len(x_synthetic), window_size)):
        end = min(start + window_size, len(x_synthetic))
        if end - start < 100:
            continue
        yt = y_synthetic[start:end]
        probs = synthetic_probabilities[start:end]
        metrics = classification_metrics(yt, probs, "binary")
        errors = (probs.argmax(axis=1) != yt).astype(float)
        detected = any(ph_synthetic.update(float(value)) for value in errors)
        if detected and first_detection is None:
            first_detection = start
        row = {"seed": seed, "mode": "controlled_distribution_shift", "stream": "synthetic", "window": window_index, "start_index": start, "end_index": end, "drift_detected": int(detected), "mean_entropy": float((-np.sum(probs * np.log(probs + 1e-12), axis=1)).mean()), "dominant_attack": pd.Series(attack_synthetic[start:end]).value_counts().index[0], "phase": pd.Series(phase_names[start:end]).mode().iloc[0]}
        row.update(metrics)
        drift_rows.append(row)
    boundary = boundaries[0] if boundaries else window_size * 3
    adaptation_start = max(0, boundary)
    evaluation_start = min(len(x_synthetic) - window_size, adaptation_start + window_size)
    evaluation_end = min(len(x_synthetic), evaluation_start + window_size)
    for budget in config.adaptation_budgets:
        available = min(budget, max(0, evaluation_start - adaptation_start))
        if available < 16:
            continue
        x_adapt = x_synthetic[evaluation_start - available:evaluation_start]
        y_adapt = y_synthetic[evaluation_start - available:evaluation_start]
        adapted_model = adapt_classifier(model, x_adapt, y_adapt, config, epochs=3)
        adapted_logits, _, _ = deep_logits_embeddings(adapted_model, x_synthetic[evaluation_start:evaluation_end], config, mc_passes=1)
        adapted_probs = softmax_temperature(adapted_logits, temperature)
        pre_probs = synthetic_probabilities[evaluation_start:evaluation_end]
        pre_metric = classification_metrics(y_synthetic[evaluation_start:evaluation_end], pre_probs, "binary")["macro_f1"]
        post_metric = classification_metrics(y_synthetic[evaluation_start:evaluation_end], adapted_probs, "binary")["macro_f1"]
        adaptation_rows.append({"seed": seed, "budget": available, "pre_adaptation_macro_f1": pre_metric, "post_adaptation_macro_f1": post_metric, "target_macro_f1": recovery_target, "recovered": int(post_metric >= recovery_target)})
    minimum_samples = min([row["budget"] for row in adaptation_rows if row["recovered"] == 1], default=np.nan)
    for row in adaptation_rows:
        row["minimum_samples_for_recovery"] = minimum_samples
    result_store["drift"].extend(drift_rows)
    result_store["psi"].extend(psi_rows)
    result_store["adaptation"].extend(adaptation_rows)
    result_store["drift_summary"].append({"seed": seed, "temporal_mode": temporal_mode, "synthetic_first_detection_index": first_detection, "first_true_boundary": boundary, "detection_delay_samples": (first_detection - boundary) if first_detection is not None else np.nan, "minimum_adaptation_samples": minimum_samples, "natural_windows": sum(row["stream"] == "natural" for row in drift_rows), "synthetic_windows": sum(row["stream"] == "synthetic" for row in drift_rows)})
    if seed == config.profile.seeds[0]:
        plot_drift_dashboard(pd.DataFrame(drift_rows), pd.DataFrame(psi_rows), pd.DataFrame(adaptation_rows), output_dirs["figures"] / "07_concept_drift_and_adaptation_dashboard.png")
    return model, preprocessor

def deep_feature_importance(model: DeepClassifier, x_test: np.ndarray, y_test: np.ndarray, temperature: float, feature_names: Sequence[str], config: RunConfig, seed: int):
    rng = np.random.default_rng(seed)
    sample_count = min(12000, len(x_test))
    indices = rng.choice(len(x_test), sample_count, replace=False)
    x_sample = x_test[indices]
    y_sample = y_test[indices]
    logits, _, _ = deep_logits_embeddings(model, x_sample, config, mc_passes=1)
    baseline = classification_metrics(y_sample, softmax_temperature(logits, temperature), "binary")["macro_f1"]
    rows = []
    repeats = 2 if config.profile.name in {"quick", "smoke"} else 3
    for feature_index, feature_name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            perturbed = x_sample.copy()
            perturbed[:, feature_index] = rng.permutation(perturbed[:, feature_index])
            perturbed_logits, _, _ = deep_logits_embeddings(model, perturbed, config, mc_passes=1)
            metric = classification_metrics(y_sample, softmax_temperature(perturbed_logits, temperature), "binary")["macro_f1"]
            drops.append(baseline - metric)
        rows.append({"method": "Proposed permutation importance", "feature": feature_name, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops)), "baseline_macro_f1": baseline})
    return rows

def plot_feature_importance(frame: pd.DataFrame, path: Path):
    if frame.empty:
        return
    grouped = frame.groupby(["method", "feature"], as_index=False)["importance_mean"].mean()
    proposed = grouped[grouped["method"] == "Proposed permutation importance"].sort_values("importance_mean", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(proposed["feature"], proposed["importance_mean"])
    ax.set_xlabel("Decrease in macro F1 after feature permutation")
    ax.set_title("Global feature importance of the proposed model")
    save_figure(fig, path)

def constraint_robustness(model: DeepClassifier, x_test: np.ndarray, y_test: np.ndarray, temperature: float, config: RunConfig, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    clean_logits, _, _ = deep_logits_embeddings(model, x_test, config, mc_passes=1)
    clean_probs = softmax_temperature(clean_logits, temperature)
    clean_metric = classification_metrics(y_test, clean_probs, "binary")
    rows.append({"perturbation": "clean", "level": 0.0, **clean_metric})
    for level in [0.01, 0.03, 0.05, 0.10]:
        noisy = np.clip(x_test + rng.normal(0.0, level, x_test.shape).astype(np.float32), -10, 10)
        logits, _, _ = deep_logits_embeddings(model, noisy, config, mc_passes=1)
        probs = softmax_temperature(logits, temperature)
        rows.append({"perturbation": "bounded_gaussian", "level": level, **classification_metrics(y_test, probs, "binary")})
    for level in [0.05, 0.10, 0.20]:
        masked = x_test.copy()
        mask = rng.random(masked.shape) < level
        masked[mask] = 0.0
        logits, _, _ = deep_logits_embeddings(model, masked, config, mc_passes=1)
        probs = softmax_temperature(logits, temperature)
        rows.append({"perturbation": "feature_missingness", "level": level, **classification_metrics(y_test, probs, "binary")})
    device = torch.device(config.device)
    subset_count = min(20000, len(x_test))
    indices = rng.choice(len(x_test), subset_count, replace=False)
    for epsilon in [0.005, 0.01, 0.03]:
        xb = torch.from_numpy(x_test[indices].astype(np.float32, copy=True)).to(device).requires_grad_(True)
        yb = torch.from_numpy(y_test[indices].astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
        model.eval()
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        perturbed = torch.clamp(xb + epsilon * xb.grad.sign(), -10, 10).detach().cpu().numpy()
        adv_logits, _, _ = deep_logits_embeddings(model, perturbed, config, mc_passes=1)
        adv_probs = softmax_temperature(adv_logits, temperature)
        rows.append({"perturbation": "fgsm_bounded", "level": epsilon, **classification_metrics(y_test[indices], adv_probs, "binary")})
    return rows

def statistical_tests(results: pd.DataFrame):
    rows = []
    nemenyi = pd.DataFrame()
    if results.empty:
        return rows, nemenyi
    index_columns = ["seed", "scenario", "dataset", "task"]
    pivot = results.pivot_table(index=index_columns, columns="model", values="macro_f1", aggfunc="mean").dropna(axis=0, how="any")
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
        arrays = [pivot[column].values for column in pivot.columns]
        statistic, p_value = friedmanchisquare(*arrays)
        rows.append({"test": "Friedman", "comparison": "All models", "statistic": statistic, "p_value": p_value, "n_blocks": len(pivot)})
        if HAS_POSTHOCS:
            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.index = pivot.columns
            nemenyi.columns = pivot.columns
    if "Proposed SSL-OpenSet" in pivot.columns:
        proposed = pivot["Proposed SSL-OpenSet"]
        for model in pivot.columns:
            if model == "Proposed SSL-OpenSet":
                continue
            try:
                statistic, p_value = wilcoxon(proposed, pivot[model], alternative="greater", zero_method="wilcox")
            except Exception:
                statistic, p_value = np.nan, np.nan
            rows.append({"test": "Wilcoxon signed-rank", "comparison": f"Proposed SSL-OpenSet > {model}", "statistic": statistic, "p_value": p_value, "n_blocks": len(pivot)})
    return rows, nemenyi

def write_excel(result_store: Dict[str, List[Dict[str, Any]]], metadata: Dict[str, Any], output_path: Path, nemenyi: pd.DataFrame):
    sheets = {
        "Configuration": pd.DataFrame([metadata["configuration"]]),
        "Environment": pd.DataFrame([metadata["environment"]]),
        "Dataset_Summary": pd.DataFrame(result_store["dataset_summary"]),
        "Feature_Set": pd.DataFrame(result_store["features"]),
        "Main_Results": pd.DataFrame(result_store["main"]),
        "Confidence_Intervals": pd.DataFrame(result_store["confidence"]),
        "Per_Class": pd.DataFrame(result_store["per_class"]),
        "Calibration": pd.DataFrame(result_store["calibration"]),
        "Conformal": pd.DataFrame(result_store["conformal"]),
        "Open_Set": pd.DataFrame(result_store["open_set"]),
        "Drift_Windows": pd.DataFrame(result_store["drift"]),
        "Drift_Summary": pd.DataFrame(result_store["drift_summary"]),
        "Feature_Drift_PSI": pd.DataFrame(result_store["psi"]),
        "Adaptation": pd.DataFrame(result_store["adaptation"]),
        "Robustness": pd.DataFrame(result_store["robustness"]),
        "Ablation": pd.DataFrame(result_store["ablation"]),
        "Statistical_Tests": pd.DataFrame(result_store["statistics"]),
        "Nemenyi_PValues": nemenyi.reset_index().rename(columns={"index": "model"}) if not nemenyi.empty else pd.DataFrame(),
        "Training_History": pd.DataFrame(result_store["training_history"]),
        "Runtime_Memory": pd.DataFrame(result_store["runtime"]),
        "Feature_Importance": pd.DataFrame(result_store["feature_importance"])
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
        workbook = writer.book
        header_format = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D", "align": "center", "valign": "vcenter"})
        title_format = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#17365D"})
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        decimal_format = workbook.add_format({"num_format": "0.000000"})
        integer_format = workbook.add_format({"num_format": "0"})
        for sheet_name, source_frame in sheets.items():
            frame = source_frame.copy()
            if frame.empty:
                frame = pd.DataFrame({"status": ["No records generated for this execution profile"]})
            frame = frame.replace([np.inf, -np.inf], np.nan)
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False, startrow=2)
            worksheet = writer.sheets[safe_name]
            worksheet.write(0, 0, safe_name.replace("_", " "), title_format)
            worksheet.set_row(2, 24)
            worksheet.freeze_panes(3, 0)
            worksheet.autofilter(2, 0, 2 + len(frame), max(0, len(frame.columns) - 1))
            sample = frame.head(1000)
            for column_index, column in enumerate(frame.columns):
                worksheet.write(2, column_index, column, header_format)
                values = sample[column].astype(str) if column in sample.columns else pd.Series(dtype=str)
                observed = int(values.str.len().max()) if len(values) else 0
                width = min(42, max(11, len(str(column)) + 2, observed + 2))
                if pd.api.types.is_integer_dtype(frame[column]):
                    worksheet.set_column(column_index, column_index, width, integer_format)
                elif pd.api.types.is_float_dtype(frame[column]):
                    worksheet.set_column(column_index, column_index, width, decimal_format)
                else:
                    worksheet.set_column(column_index, column_index, width, wrap_format)
            if len(frame) > 1:
                metric_columns = [index for index, column in enumerate(frame.columns) if any(token in str(column).lower() for token in ["f1", "accuracy", "recall", "rate", "auc", "coverage", "risk", "ece", "mcc", "brier"])]
                for column_index in metric_columns:
                    column_name = str(frame.columns[column_index]).lower()
                    lower_is_better = any(token in column_name for token in ["false_positive", "false_negative", "risk", "ece", "brier", "time", "memory", "delay"])
                    colors = {"min_color": "#63BE7B", "mid_color": "#FFEB84", "max_color": "#F8696B"} if lower_is_better else {"min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B"}
                    worksheet.conditional_format(3, column_index, 2 + len(frame), column_index, {"type": "3_color_scale", **colors})
        summary = workbook.add_worksheet("Executive_Summary")
        summary.set_column("A:A", 30)
        summary.set_column("B:B", 95, wrap_format)
        summary.write("A1", "Article", header_format)
        summary.write("B1", "An Uncertainty-Aware Deep Learning Framework for Cross-Dataset Detection of Unknown Attacks and Concept Drift in IoT Networks", wrap_format)
        summary.write("A3", "Primary numerical table", header_format)
        summary.write("B3", "Main_Results contains the principal within-dataset and cross-dataset metrics. Confidence_Intervals reports bootstrap uncertainty across matched runs.", wrap_format)
        summary.write("A5", "Open-set and uncertainty", header_format)
        summary.write("B5", "Open_Set, Calibration and Conformal contain unknown-attack AUROC, ECE, Brier score, prediction-set coverage and selective-risk results.", wrap_format)
        summary.write("A7", "Drift and adaptation", header_format)
        summary.write("B7", "Drift_Windows, Drift_Summary, Feature_Drift_PSI and Adaptation contain temporal degradation, Page-Hinkley detection, PSI and adaptation sample-complexity results.", wrap_format)
        summary.write("A9", "Statistical reporting", header_format)
        summary.write("B9", "Use means and 95% confidence intervals across seeds and report Friedman, Nemenyi and matched one-sided Wilcoxon results from the statistical sheets.", wrap_format)
        summary.freeze_panes(1, 0)

def cleanup_runtime():
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass
    logging.shutdown()

def system_environment():
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "scipy": scipy.__version__,
        "xgboost_available": HAS_XGBOOST,
        "umap_available": HAS_UMAP,
        "scikit_posthocs_available": HAS_POSTHOCS
    }

def add_dataset_summary(result_store: Dict[str, List[Dict[str, Any]]], name: str, frame: pd.DataFrame, path: Path, features: Sequence[str]):
    counts = frame["__attack_label__"].value_counts()
    result_store["dataset_summary"].append({
        "dataset": name,
        "path": str(path),
        "sampled_rows": len(frame),
        "benign_rows": int((frame["__binary_label__"] == 0).sum()),
        "attack_rows": int((frame["__binary_label__"] == 1).sum()),
        "attack_classes": int(frame["__attack_label__"].nunique()),
        "feature_count": len(features),
        "largest_class": counts.index[0],
        "largest_class_count": int(counts.iloc[0]),
        "file_size_gb": round(path.stat().st_size / (1024 ** 3), 4),
        "sha1": manifest_sha1(path)
    })
    for index, feature in enumerate(features):
        result_store["features"].append({"dataset": name, "feature_index": index + 1, "feature": feature})

def run_task(source_name: str, source_frame: pd.DataFrame, source_features: Sequence[str], target_name: Optional[str], target_frame: Optional[pd.DataFrame], target_features: Optional[Sequence[str]], task: str, config: RunConfig, output_dirs: Dict[str, Path], result_store: Dict[str, List[Dict[str, Any]]], seed: int):
    label_col = "__binary_label__" if task == "binary" else "__attack_label__"
    if task == "multiclass" and target_frame is not None:
        common = sorted(set(source_frame[label_col].unique()).intersection(set(target_frame[label_col].unique())))
        source_frame = source_frame[source_frame[label_col].isin(common)].copy()
        target_frame = target_frame[target_frame[label_col].isin(common)].copy()
    train_frame, validation_frame, calibration_frame, test_frame = four_way_split(source_frame, label_col, seed)
    if task == "binary":
        encoder = None
        class_names = ["Benign", "Attack"]
        y_train = train_frame[label_col].to_numpy(dtype=int)
        y_validation = validation_frame[label_col].to_numpy(dtype=int)
        y_calibration = calibration_frame[label_col].to_numpy(dtype=int)
        y_test = test_frame[label_col].to_numpy(dtype=int)
    else:
        encoder = LabelEncoder().fit(source_frame[label_col])
        class_names = encoder.classes_.tolist()
        y_train = encoder.transform(train_frame[label_col])
        y_validation = encoder.transform(validation_frame[label_col])
        y_calibration = encoder.transform(calibration_frame[label_col])
        y_test = encoder.transform(test_frame[label_col])
    preprocessor = NetworkPreprocessor()
    x_train = preprocessor.fit_transform(train_frame[list(source_features)])
    x_validation = preprocessor.transform(validation_frame[list(source_features)])
    x_calibration = preprocessor.transform(calibration_frame[list(source_features)])
    x_test = preprocessor.transform(test_frame[list(source_features)])
    n_classes = len(class_names)
    baseline_models = build_baselines(task, n_classes, config.profile, seed)
    fitted_models = {}
    for model_name, model in baseline_models.items():
        logging.info("Training %s | %s | %s | seed %s", model_name, source_name, task, seed)
        fitted, probabilities, metrics = train_evaluate_baseline(model_name, model, x_train, y_train, x_test, y_test, task, n_classes)
        fitted_models[model_name] = fitted
        context = {"seed": seed, "scenario": "within_dataset", "dataset": source_name, "source": source_name, "target": source_name, "task": task}
        row = dict(context)
        row.update(metrics)
        result_store["main"].append(row)
        result_store["runtime"].append({key: row.get(key) for key in ["seed", "scenario", "dataset", "task", "model", "training_seconds", "inference_seconds", "milliseconds_per_sample", "peak_memory_mb", "memory_delta_mb"]})
        result_store["per_class"].extend(per_class_rows(y_test, probabilities, class_names, {**context, "model": model_name}))
        for ci in bootstrap_confidence_intervals(y_test, probabilities, task, config.profile.bootstrap_iterations, seed):
            result_store["confidence"].append({**context, "model": model_name, **ci})
    ssl_path = output_dirs["models"] / f"ssl_{source_name}_{task}_seed{seed}.pt"
    deep_path = output_dirs["models"] / f"proposed_{source_name}_{task}_seed{seed}.pt"
    with MemoryMonitor() as monitor:
        ssl_start = time.perf_counter()
        ssl_model, ssl_history = pretrain_ssl(x_train, x_validation, config, seed, ssl_path)
        ssl_seconds = time.perf_counter() - ssl_start
        cls_start = time.perf_counter()
        proposed, cls_history = train_deep_classifier(x_train, y_train, x_validation, y_validation, ssl_model, n_classes, config, seed, deep_path)
        cls_seconds = time.perf_counter() - cls_start
        validation_logits, validation_embeddings, validation_output = deep_logits_embeddings(proposed, x_validation, config, mc_passes=1)
        temperature = fit_temperature(validation_logits, y_validation, config.device)
        calibration_logits, calibration_embeddings, calibration_output = deep_logits_embeddings(proposed, x_calibration, config, mc_passes=1)
        test_logits, test_embeddings, test_output = deep_logits_embeddings(proposed, x_test, config, mc_passes=max(1, config.profile.mc_passes))
    probabilities_before = test_output["probabilities"]
    probabilities = temperature_rescale_probabilities(probabilities_before, temperature)
    calibration_probabilities = softmax_temperature(calibration_logits, temperature)
    predictive_entropy, mutual_information = predictive_uncertainty(test_output["passes"])
    metrics = classification_metrics(y_test, probabilities, task)
    metrics.update({
        "model": "Proposed SSL-OpenSet",
        "training_seconds": ssl_seconds + cls_seconds,
        "inference_seconds": test_output["inference_seconds"],
        "milliseconds_per_sample": test_output["milliseconds_per_sample"],
        "peak_memory_mb": monitor.peak_mb,
        "memory_delta_mb": np.nan,
        "temperature": temperature,
        "mean_predictive_entropy": float(np.mean(predictive_entropy)),
        "mean_mutual_information": float(np.mean(mutual_information))
    })
    context = {"seed": seed, "scenario": "within_dataset", "dataset": source_name, "source": source_name, "target": source_name, "task": task}
    result_store["main"].append({**context, **metrics})
    result_store["runtime"].append({key: {**context, **metrics}.get(key) for key in ["seed", "scenario", "dataset", "task", "model", "training_seconds", "inference_seconds", "milliseconds_per_sample", "peak_memory_mb", "memory_delta_mb"]})
    result_store["per_class"].extend(per_class_rows(y_test, probabilities, class_names, {**context, "model": "Proposed SSL-OpenSet"}))
    for ci in bootstrap_confidence_intervals(y_test, probabilities, task, config.profile.bootstrap_iterations, seed):
        result_store["confidence"].append({**context, "model": "Proposed SSL-OpenSet", **ci})
    before_metrics = classification_metrics(y_test, probabilities_before, task)
    result_store["calibration"].append({**context, "model": "Proposed SSL-OpenSet", "temperature": temperature, "ece_before": before_metrics["ece"], "ece_after": metrics["ece"], "brier_before": before_metrics["brier_score"], "brier_after": metrics["brier_score"]})
    conformal_metrics, set_sizes, prediction_sets = conformal_evaluation(calibration_probabilities, y_calibration, probabilities, y_test, config.alpha_conformal)
    result_store["conformal"].append({**context, "model": "Proposed SSL-OpenSet", **conformal_metrics})
    result_store["training_history"].extend([{**context, "stage": "ssl", **row} for row in ssl_history.to_dict("records")])
    result_store["training_history"].extend([{**context, "stage": "classifier", **row} for row in cls_history.to_dict("records")])
    if seed == config.profile.seeds[0]:
        result_store["within_figure_payloads"].append(compact_figure_payload(y_test, probabilities, probabilities_before, predictive_entropy, class_names, source_name, task, "within_dataset", seed))
        if source_name == "NF-ToN-IoT-v1" and task == "binary":
            plot_embedding(test_embeddings, y_test, class_names, output_dirs["figures"] / "04_learned_representation_embedding.png", min(config.max_plot_points, 6000), seed)
    result_store["ablation"].append({"seed": seed, "dataset": source_name, "task": task, "variant": "No temperature scaling", "macro_f1": before_metrics["macro_f1"], "ece": before_metrics["ece"], "coverage": np.nan, "unknown_auroc": np.nan})
    result_store["ablation"].append({"seed": seed, "dataset": source_name, "task": task, "variant": "SSL + temperature scaling", "macro_f1": metrics["macro_f1"], "ece": metrics["ece"], "coverage": conformal_metrics["coverage"], "unknown_auroc": np.nan})
    if seed == config.profile.seeds[0] and source_name == "NF-ToN-IoT-v1" and task == "binary":
        importance_rows = deep_feature_importance(proposed, x_test, y_test, temperature, source_features, config, seed)
        if "Extra Trees" in fitted_models and hasattr(fitted_models["Extra Trees"], "feature_importances_"):
            for feature_name, value in zip(source_features, fitted_models["Extra Trees"].feature_importances_):
                importance_rows.append({"method": "Extra Trees impurity importance", "feature": feature_name, "importance_mean": float(value), "importance_std": 0.0, "baseline_macro_f1": np.nan})
        result_store["feature_importance"].extend([{"seed": seed, "dataset": source_name, **row} for row in importance_rows])
        plot_feature_importance(pd.DataFrame(importance_rows), output_dirs["figures"] / "05_feature_importance.png")
    if target_frame is not None and target_features is not None:
        source_matched, target_matched = matched_feature_columns(source_frame, target_frame, source_features, target_features)
        if len(source_matched) != len(source_features):
            preprocessor_cross = NetworkPreprocessor()
            x_train_cross = preprocessor_cross.fit_transform(train_frame[source_matched])
            x_validation_cross = preprocessor_cross.transform(validation_frame[source_matched])
            x_calibration_cross = preprocessor_cross.transform(calibration_frame[source_matched])
            cross_preprocessor = preprocessor_cross
        else:
            x_train_cross = x_train
            x_validation_cross = x_validation
            x_calibration_cross = x_calibration
            cross_preprocessor = preprocessor
        if task == "binary":
            target_y = target_frame[label_col].to_numpy(dtype=int)
            target_class_names = class_names
        else:
            target_frame = target_frame[target_frame[label_col].isin(class_names)].copy()
            target_y = encoder.transform(target_frame[label_col])
            target_class_names = class_names
        target_x = cross_preprocessor.transform(target_frame[target_matched])
        cross_baselines = build_baselines(task, n_classes, config.profile, seed)
        for model_name, model in cross_baselines.items():
            logging.info("Cross-dataset training %s | %s to %s | %s | seed %s", model_name, source_name, target_name, task, seed)
            fitted, probabilities_cross, metrics_cross = train_evaluate_baseline(model_name, model, x_train_cross, y_train, target_x, target_y, task, n_classes)
            cross_context = {"seed": seed, "scenario": "cross_dataset", "dataset": f"{source_name}->{target_name}", "source": source_name, "target": target_name, "task": task}
            result_store["main"].append({**cross_context, **metrics_cross})
            result_store["runtime"].append({key: {**cross_context, **metrics_cross}.get(key) for key in ["seed", "scenario", "dataset", "task", "model", "training_seconds", "inference_seconds", "milliseconds_per_sample", "peak_memory_mb", "memory_delta_mb"]})
            result_store["per_class"].extend(per_class_rows(target_y, probabilities_cross, target_class_names, {**cross_context, "model": model_name}))
            for ci in bootstrap_confidence_intervals(target_y, probabilities_cross, task, config.profile.bootstrap_iterations, seed):
                result_store["confidence"].append({**cross_context, "model": model_name, **ci})
        ssl_cross, _ = pretrain_ssl(x_train_cross, x_validation_cross, config, seed)
        proposed_cross, _ = train_deep_classifier(x_train_cross, y_train, x_validation_cross, y_validation, ssl_cross, n_classes, config, seed)
        validation_cross_logits, _, _ = deep_logits_embeddings(proposed_cross, x_validation_cross, config, mc_passes=1)
        temperature_cross = fit_temperature(validation_cross_logits, y_validation, config.device)
        cross_logits, cross_embeddings, cross_output = deep_logits_embeddings(proposed_cross, target_x, config, mc_passes=1)
        cross_probabilities = softmax_temperature(cross_logits, temperature_cross)
        cross_metrics = classification_metrics(target_y, cross_probabilities, task)
        cross_metrics.update({"model": "Proposed SSL-OpenSet", "training_seconds": np.nan, "inference_seconds": cross_output["inference_seconds"], "milliseconds_per_sample": cross_output["milliseconds_per_sample"], "peak_memory_mb": np.nan, "memory_delta_mb": np.nan, "temperature": temperature_cross})
        cross_context = {"seed": seed, "scenario": "cross_dataset", "dataset": f"{source_name}->{target_name}", "source": source_name, "target": target_name, "task": task}
        result_store["main"].append({**cross_context, **cross_metrics})
        result_store["runtime"].append({key: {**cross_context, **cross_metrics}.get(key) for key in ["seed", "scenario", "dataset", "task", "model", "training_seconds", "inference_seconds", "milliseconds_per_sample", "peak_memory_mb", "memory_delta_mb"]})
        result_store["per_class"].extend(per_class_rows(target_y, cross_probabilities, target_class_names, {**cross_context, "model": "Proposed SSL-OpenSet"}))
        for ci in bootstrap_confidence_intervals(target_y, cross_probabilities, task, config.profile.bootstrap_iterations, seed):
            result_store["confidence"].append({**cross_context, "model": "Proposed SSL-OpenSet", **ci})
        if seed == config.profile.seeds[0]:
            result_store["cross_figure_payloads"].append(compact_figure_payload(target_y, cross_probabilities, None, None, target_class_names, f"{source_name}->{target_name}", task, "cross_dataset", seed))
    return {
        "model": proposed,
        "preprocessor": preprocessor,
        "temperature": temperature,
        "class_names": class_names,
        "label_encoder": encoder,
        "test_frame": test_frame,
        "x_test": x_test,
        "y_test": y_test,
        "test_probabilities": probabilities,
        "test_logits": test_logits,
        "test_embeddings": test_embeddings,
        "calibration_frame": calibration_frame,
        "x_calibration": x_calibration,
        "y_calibration": y_calibration,
        "calibration_probabilities": calibration_probabilities,
        "calibration_logits": calibration_logits,
        "calibration_embeddings": calibration_embeddings,
        "ssl_history": ssl_history,
        "cls_history": cls_history
    }

def run_open_set(source_name: str, frame: pd.DataFrame, features: Sequence[str], config: RunConfig, output_dirs: Dict[str, Path], result_store: Dict[str, List[Dict[str, Any]]], seed: int):
    counts = frame["__attack_label__"].value_counts()
    minimum_count = 50 if config.profile.name == "smoke" else 500
    candidates = [name for name, count in counts.items() if name != "Benign" and count >= minimum_count]
    candidates = sorted(candidates, key=lambda name: counts[name], reverse=True)
    limit = min(len(candidates), config.profile.unknown_class_limit)
    if limit >= 3 and len(candidates) >= 3:
        candidates = [candidates[0], candidates[len(candidates) // 2], candidates[-1]][:limit]
    else:
        candidates = candidates[:limit]
    for holdout in candidates:
        logging.info("Open-set holdout %s | %s | seed %s", holdout, source_name, seed)
        known = frame[frame["__attack_label__"] != holdout].copy()
        unknown = frame[frame["__attack_label__"] == holdout].copy()
        train_frame, validation_frame, calibration_frame, known_test = four_way_split(known, "__attack_label__", seed)
        encoder = LabelEncoder().fit(known["__attack_label__"])
        y_train = encoder.transform(train_frame["__attack_label__"])
        y_validation = encoder.transform(validation_frame["__attack_label__"])
        y_calibration = encoder.transform(calibration_frame["__attack_label__"])
        y_known_test = encoder.transform(known_test["__attack_label__"])
        preprocessor = NetworkPreprocessor()
        x_train = preprocessor.fit_transform(train_frame[list(features)])
        x_validation = preprocessor.transform(validation_frame[list(features)])
        x_calibration = preprocessor.transform(calibration_frame[list(features)])
        x_known = preprocessor.transform(known_test[list(features)])
        unknown_sample = unknown.sample(min(len(unknown), len(known_test)), random_state=seed)
        x_unknown = preprocessor.transform(unknown_sample[list(features)])
        ssl_model, _ = pretrain_ssl(x_train, x_validation, config, seed)
        model, _ = train_deep_classifier(x_train, y_train, x_validation, y_validation, ssl_model, len(encoder.classes_), config, seed)
        val_logits, _, _ = deep_logits_embeddings(model, x_validation, config, mc_passes=1)
        temperature = fit_temperature(val_logits, y_validation, config.device)
        cal_logits, cal_embeddings, _ = deep_logits_embeddings(model, x_calibration, config, mc_passes=1)
        cal_probs = softmax_temperature(cal_logits, temperature)
        train_logits, train_embeddings, _ = deep_logits_embeddings(model, x_train, config, mc_passes=1)
        means, variances = fit_prototypes(train_embeddings, y_train, len(encoder.classes_))
        reference, thresholds = fit_open_reference(cal_logits, cal_embeddings, cal_probs, y_calibration, means, variances, config.alpha_open)
        x_combined = np.concatenate([x_known, x_unknown], axis=0)
        y_unknown = np.concatenate([np.zeros(len(x_known), dtype=int), np.ones(len(x_unknown), dtype=int)])
        logits, embeddings, output = deep_logits_embeddings(model, x_combined, config, mc_passes=1)
        probabilities = softmax_temperature(logits, temperature)
        scores = calculate_open_scores(logits, embeddings, probabilities, means, variances, reference)
        score_thresholds = {
            "msp": thresholds["msp"],
            "entropy": thresholds["entropy"],
            "energy": thresholds["energy"],
            "mahalanobis": thresholds["distance"],
            "composite": thresholds["composite"]
        }
        for method, values in scores.items():
            metric = open_set_metrics(y_unknown, values, score_thresholds[method])
            row = {"seed": seed, "dataset": source_name, "holdout_attack": holdout, "method": method, "known_samples": len(x_known), "unknown_samples": len(x_unknown), **metric}
            result_store["open_set"].append(row)
        closed_probabilities = probabilities[:len(x_known)]
        closed_metrics = classification_metrics(y_known_test, closed_probabilities, "multiclass")
        composite_auc = open_set_metrics(y_unknown, scores["composite"], score_thresholds["composite"])["unknown_auroc"]
        result_store["ablation"].append({"seed": seed, "dataset": source_name, "task": "open_set", "variant": f"MSP holdout {holdout}", "macro_f1": closed_metrics["macro_f1"], "ece": closed_metrics["ece"], "coverage": np.nan, "unknown_auroc": open_set_metrics(y_unknown, scores["msp"], score_thresholds["msp"])["unknown_auroc"]})
        result_store["ablation"].append({"seed": seed, "dataset": source_name, "task": "open_set", "variant": f"Composite holdout {holdout}", "macro_f1": closed_metrics["macro_f1"], "ece": closed_metrics["ece"], "coverage": np.nan, "unknown_auroc": composite_auc})
        if seed == config.profile.seeds[0] and holdout == candidates[0]:
            plot_open_set_dashboard(y_unknown, scores, output_dirs["figures"] / "06_open_set_detection_dashboard.png", holdout)

def run_no_ssl_ablation(frame: pd.DataFrame, features: Sequence[str], config: RunConfig, result_store: Dict[str, List[Dict[str, Any]]], seed: int):
    train_frame, validation_frame, calibration_frame, test_frame = four_way_split(frame, "__binary_label__", seed)
    preprocessor = NetworkPreprocessor()
    x_train = preprocessor.fit_transform(train_frame[list(features)])
    x_validation = preprocessor.transform(validation_frame[list(features)])
    x_test = preprocessor.transform(test_frame[list(features)])
    y_train = train_frame["__binary_label__"].to_numpy(dtype=int)
    y_validation = validation_frame["__binary_label__"].to_numpy(dtype=int)
    y_test = test_frame["__binary_label__"].to_numpy(dtype=int)
    model, _ = train_deep_classifier(x_train, y_train, x_validation, y_validation, None, 2, config, seed)
    logits, _, _ = deep_logits_embeddings(model, x_test, config, mc_passes=1)
    probabilities = softmax_temperature(logits, 1.0)
    metrics = classification_metrics(y_test, probabilities, "binary")
    result_store["ablation"].append({"seed": seed, "dataset": "NF-ToN-IoT-v1", "task": "binary", "variant": "Supervised deep model without SSL", "macro_f1": metrics["macro_f1"], "ece": metrics["ece"], "coverage": np.nan, "unknown_auroc": np.nan})

def create_methodology_text(config: RunConfig, path: Path):
    text = f"""Proposed framework

The pipeline applies leakage-aware preprocessing by removing endpoint identifiers and raw timestamps from predictive inputs, converting values to numeric form, imputing medians estimated on the training partition, applying log1p to non-negative highly skewed features, fitting a RobustScaler on training data only, and clipping standardized values to [-10, 10].

Self-supervised representation learning uses a masked tabular autoencoder. Randomly masked flow features are reconstructed while two independently augmented views are aligned using invariance, variance, and covariance regularization. The encoder is then fine-tuned with a class-weighted deep classifier.

Open-set detection combines maximum-softmax uncertainty, predictive entropy, energy, and class-conditional diagonal Mahalanobis distance in the learned embedding space. Each component is robustly standardized using known calibration traffic and the final threshold is estimated only from the calibration partition.

Uncertainty is estimated with temperature-scaled probabilities and Monte Carlo dropout. Split conformal prediction uses alpha={config.alpha_conformal} and abstains when the prediction set is not a singleton. Risk-coverage curves quantify the accuracy-coverage trade-off.

Cross-dataset experiments train on NF-ToN-IoT-v1 and test on NF-BoT-IoT-v1, and then reverse the direction, using only identically named shared NetFlow features. Within-dataset results use independent training, validation, calibration, and test partitions.

Unknown-attack experiments follow leave-one-attack-family-out evaluation. The held-out family is excluded from self-supervised and supervised training and appears only during testing as the unknown class.

Concept drift experiments use NF-ToN-IoT-v3 temporal ordering when a valid flow-start timestamp is available. Page-Hinkley monitors online prediction errors. Controlled prior and attack-family shifts provide known drift boundaries. Adaptation is evaluated with labelled budgets {config.adaptation_budgets}, reporting the minimum number of samples required to recover 95 percent of pre-drift macro F1.

The principal metrics are macro F1, weighted F1, detection recall, false-positive rate, false-negative rate, PR-AUC, MCC, balanced accuracy, unknown AUROC, ECE, Brier score, conformal coverage and risk, inference time, peak memory, and adaptation sample complexity. Confidence intervals are obtained by stratified bootstrap resampling, and model comparisons use Friedman, Nemenyi, and one-sided Wilcoxon signed-rank tests over matched runs.
"""
    path.write_text(text, encoding="utf-8")

def synthetic_frame(rows: int, features: int, classes: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.choice(classes, rows, p=np.array([0.55] + [(0.45 / (classes - 1))] * (classes - 1)))
    x = rng.normal(0, 1, (rows, features)) + y[:, None] * rng.normal(0.35, 0.1, (1, features))
    frame = pd.DataFrame(x, columns=[f"F{i}" for i in range(features)])
    names = ["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"][:classes]
    frame["Attack"] = [names[value] for value in y]
    frame["Label"] = (y != 0).astype(int)
    frame["FLOW_START_MILLISECONDS"] = np.arange(rows)
    frame["__binary_label__"] = frame["Label"]
    frame["__attack_label__"] = frame["Attack"]
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=r"E:\private information\Military service\Article\Security me")
    parser.add_argument("--profile", choices=["quick", "standard", "publication"], default="standard")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    profile = get_profile(args.profile)
    if args.smoke_test:
        profile = Profile("smoke", [42], 1000, 5000, 1, 1, 1, 10, 2, 1, 500, 10, 256, 128, 0)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    project_root = Path(args.project_root)
    paths = discover_paths(project_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{profile.name}_{timestamp}"
    base_output_root = paths.output_root
    if args.smoke_test:
        paths.output_root = Path.cwd() / "security_article_output_smoke_test"
    else:
        paths.output_root = base_output_root / run_id
    if args.overwrite and paths.output_root.exists():
        shutil.rmtree(paths.output_root)
    output_dirs = create_output_tree(paths.output_root)
    configure_logging(paths.output_root)
    batch_size = profile.batch_gpu if device == "cuda" else profile.batch_cpu
    config = RunConfig(profile, paths, device, batch_size, 0.10, 0.05, 0.30, 64, 0.20, 1e-3, 8e-4, 1e-4, [64, 128, 256, 512, 1024, 2048], 8000, run_id)
    logging.info("Run ID: %s", run_id)
    logging.info("Device: %s | Batch size: %s", device, batch_size)
    apply_plot_style()
    result_store = {key: [] for key in ["dataset_summary", "features", "main", "confidence", "per_class", "calibration", "conformal", "open_set", "drift", "drift_summary", "psi", "adaptation", "robustness", "ablation", "statistics", "training_history", "runtime", "feature_importance", "within_figure_payloads", "cross_figure_payloads"]}
    if args.smoke_test:
        ton_v1 = synthetic_frame(1800, 10, 5, 42)
        bot_v1 = synthetic_frame(1600, 10, 4, 43)
        ton_v3 = synthetic_frame(2400, 20, 5, 44)
        ton_v1_features = [c for c in ton_v1.columns if c.startswith("F")]
        bot_v1_features = [c for c in bot_v1.columns if c.startswith("F")]
        ton_v3_features = [c for c in ton_v3.columns if c.startswith("F")]
        ton_v3_timestamps = ["FLOW_START_MILLISECONDS"]
    else:
        validate_paths(paths)
        logging.info("Loading NF-ToN-IoT-v1")
        ton_v1, ton_v1_binary, ton_v1_attack, ton_v1_timestamps = stratified_chunk_sample(paths.ton_v1, profile.v1_per_class, profile.seeds[0])
        logging.info("Loading NF-BoT-IoT-v1")
        bot_v1, bot_v1_binary, bot_v1_attack, bot_v1_timestamps = stratified_chunk_sample(paths.bot_v1, profile.v1_per_class, profile.seeds[0] + 1)
        logging.info("Loading NF-ToN-IoT-v3")
        ton_v3, ton_v3_binary, ton_v3_attack, ton_v3_timestamps = uniform_reservoir_sample(paths.ton_v3, profile.v3_total, profile.seeds[0] + 2)
        ton_v1_schema = read_feature_schema(paths.ton_v1)
        bot_v1_schema = read_feature_schema(paths.bot_v1)
        ton_v3_schema = read_feature_schema(paths.ton_v3)
        ton_v1_features = choose_feature_columns(ton_v1, ton_v1_schema, ton_v1_binary, ton_v1_attack, ton_v1_timestamps)
        bot_v1_features = choose_feature_columns(bot_v1, bot_v1_schema, bot_v1_binary, bot_v1_attack, bot_v1_timestamps)
        ton_v3_features = choose_feature_columns(ton_v3, ton_v3_schema, ton_v3_binary, ton_v3_attack, ton_v3_timestamps)
        add_dataset_summary(result_store, "NF-ToN-IoT-v1", ton_v1, paths.ton_v1, ton_v1_features)
        add_dataset_summary(result_store, "NF-BoT-IoT-v1", bot_v1, paths.bot_v1, bot_v1_features)
        add_dataset_summary(result_store, "NF-ToN-IoT-v3", ton_v3, paths.ton_v3, ton_v3_features)
    plot_class_distribution({"NF-ToN-IoT-v1": ton_v1, "NF-BoT-IoT-v1": bot_v1, "NF-ToN-IoT-v3": ton_v3}, output_dirs["figures"])
    first_seed_artifacts = None
    for seed in profile.seeds:
        logging.info("Starting seed %s", seed)
        binary_ton = run_task("NF-ToN-IoT-v1", ton_v1, ton_v1_features, "NF-BoT-IoT-v1", bot_v1, bot_v1_features, "binary", config, output_dirs, result_store, seed)
        run_task("NF-BoT-IoT-v1", bot_v1, bot_v1_features, "NF-ToN-IoT-v1", ton_v1, ton_v1_features, "binary", config, output_dirs, result_store, seed)
        run_task("NF-ToN-IoT-v1", ton_v1, ton_v1_features, "NF-BoT-IoT-v1", bot_v1, bot_v1_features, "multiclass", config, output_dirs, result_store, seed)
        run_task("NF-BoT-IoT-v1", bot_v1, bot_v1_features, "NF-ToN-IoT-v1", ton_v1, ton_v1_features, "multiclass", config, output_dirs, result_store, seed)
        run_open_set("NF-ToN-IoT-v1", ton_v1, ton_v1_features, config, output_dirs, result_store, seed)
        run_drift_analysis(ton_v3, ton_v3_features, ton_v3_timestamps, config, output_dirs, result_store, seed)
        robustness = constraint_robustness(binary_ton["model"], binary_ton["x_test"], binary_ton["y_test"], binary_ton["temperature"], config, seed)
        result_store["robustness"].extend([{"seed": seed, "dataset": "NF-ToN-IoT-v1", **row} for row in robustness])
        if seed == profile.seeds[0]:
            first_seed_artifacts = binary_ton
            run_no_ssl_ablation(ton_v1, ton_v1_features, config, result_store, seed)
    main_frame = pd.DataFrame(result_store["main"])
    statistics, nemenyi = statistical_tests(main_frame)
    result_store["statistics"].extend(statistics)
    plot_within_dataset_comparison(result_store["within_figure_payloads"], "binary", output_dirs["figures"] / "02_within_dataset_binary_comparison.png")
    plot_within_dataset_comparison(result_store["within_figure_payloads"], "multiclass", output_dirs["figures"] / "03_within_dataset_multiclass_comparison.png")
    plot_cross_dataset_comparison(result_store["cross_figure_payloads"], "binary", output_dirs["figures"] / "08_cross_dataset_binary_comparison.png")
    plot_cross_dataset_comparison(result_store["cross_figure_payloads"], "multiclass", output_dirs["figures"] / "09_cross_dataset_multiclass_comparison.png")
    robustness_frame = pd.DataFrame(result_store["robustness"])
    plot_summary_dashboard(main_frame, pd.DataFrame(result_store["ablation"]), robustness_frame, output_dirs["figures"] / "10_comparative_summary_dashboard.png")
    configuration = {
        "run_id": config.run_id,
        "profile": config.profile.name,
        "seeds": str(config.profile.seeds),
        "device": config.device,
        "batch_size": config.batch_size,
        "ssl_epochs": config.profile.ssl_epochs,
        "classifier_epochs": config.profile.cls_epochs,
        "mask_ratio": config.mask_ratio,
        "latent_dim": config.latent_dim,
        "dropout": config.dropout,
        "alpha_conformal": config.alpha_conformal,
        "alpha_open": config.alpha_open,
        "project_root": str(config.paths.project_root),
        "output_root": str(config.paths.output_root)
    }
    metadata = {"configuration": configuration, "environment": system_environment()}
    save_json(metadata, paths.output_root / "run_metadata.json")
    save_json({"paths": asdict(paths), "configuration": configuration}, paths.output_root / "experiment_configuration.json")
    create_methodology_text(config, paths.output_root / "methodology_for_paper.txt")
    excel_path = output_dirs["excel"] / "Security_AI_Complete_Results.xlsx"
    write_excel(result_store, metadata, excel_path, nemenyi)
    save_json({key: len(value) for key, value in result_store.items() if key not in {"within_figure_payloads", "cross_figure_payloads"}}, paths.output_root / "result_record_counts.json")
    if not args.smoke_test:
        base_output_root.mkdir(parents=True, exist_ok=True)
        (base_output_root / "latest_run.txt").write_text(str(paths.output_root), encoding="utf-8")
    logging.info("Execution completed")
    logging.info("Excel: %s", excel_path)
    logging.info("Figures: %s", output_dirs["figures"])
    cleanup_runtime()

if __name__ == "__main__":
    main()
