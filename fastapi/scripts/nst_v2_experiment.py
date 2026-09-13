#!/usr/bin/env python3
"""NST v2 controlled experiment: does preserving per-turn distribution (instead of
collapsing to a per-call mean) improve natural_speech_termination?

EXPERIMENTAL. Does not touch production /detect, the DETECTORS registry, or the
FAMILIES dict in scripts/train.py - it reuses their underlying building blocks
(train_family, Detector, extract_acoustic_features_v2) without wiring new
families into the live serving path. Artifacts are written to a separate
app/artifacts/_nst_v2_experiments/ directory.

Compares 4 configurations (see app/services/acoustic_features.py):
    A - NST v1 baseline (5 features)
    B - v1 + voiced_frac_near_end distributional stats (9 features)
    C - v1 + distributional stats for all 4 turn-level features (21 features)
    D - C + hard_silence_flag (22 features)

Run from the ``fastapi/`` directory:
    python scripts/nst_v2_experiment.py
"""

from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix, roc_auc_score

FASTAPI_DIR = Path(__file__).resolve().parents[1]
BASE = None  # set below from config
sys.path.insert(0, str(FASTAPI_DIR))

from app.core import config  # noqa: E402
from app.services.acoustic_features import (  # noqa: E402
    NST_V2_CONFIG_A, NST_V2_CONFIG_B, NST_V2_CONFIG_C, NST_V2_CONFIG_D,
    extract_acoustic_features_v2, turns_for_channel,
)
from app.services.classifier import Detector  # noqa: E402
from app.services.turns import caller_audio_and_turns_from_wav  # noqa: E402
from app.api.v1.endpoints.detect import _depad_turns  # noqa: E402

sys.path.insert(0, str(FASTAPI_DIR / "scripts"))
from train import train_family  # noqa: E402

BASE = config.HACKMTY26_DIR
EXPERIMENTS_DIR = config.ARTIFACTS_DIR / "_nst_v2_experiments"

CONFIGS = {
    "A_v1_baseline": NST_V2_CONFIG_A,
    "B_voiced_variability": NST_V2_CONFIG_B,
    "C_all_variability": NST_V2_CONFIG_C,
    "D_all_variability_hard_silence": NST_V2_CONFIG_D,
}

# The 16 confident false negatives identified in the prior live-pipeline investigation,
# all sharing the exact same hard-silence floor_db value.
KNOWN_HARD_SILENCE_FN_IDS = [
    "call_810e3066a7b4", "call_d4518fcb700a", "call_0cf2f4a71328", "call_e3f555297277",
    "call_0847d7417bb1", "call_c04725672945", "call_c49b6face6c3", "call_7b1e73c133f8",
    "call_e329faf2b6a1", "call_fa9e18dd83dd", "call_957a8384caff", "call_36d72e4b1aef",
    "call_181a5049ecae", "call_a3a1f51a4bf9", "call_6058d9c5c82f", "call_fc8bda4b3374",
]


# ============================================================================
# Step 1-2: build the offline dataset (ground-truth turns, matches train.py's
# own methodology) with the full v2 feature set, then determine hard_silence_flag's
# threshold from the TRAIN split only, unsupervised (never touches val labels).
# ============================================================================

def build_v2_dataset() -> pd.DataFrame:
    manifest = pd.read_csv(BASE / "manifest.csv")
    rows = []
    for record in manifest.itertuples(index=False):
        turn_file = BASE / "turns" / f"{record.anon_id}.json"
        audio_file = BASE / "audio" / f"{record.anon_id}.wav"
        if not turn_file.exists() or not audio_file.exists():
            continue
        turns = json.loads(turn_file.read_text())["turns"]
        data, sr = sf.read(audio_file, dtype="float32", always_2d=True)
        caller = data[:, config.CALLER_CHANNEL].astype(np.float64)
        feats = extract_acoustic_features_v2(caller, sr, turns, channel=config.CALLER_CHANNEL)
        if feats is None:
            continue
        feats.update(anon_id=record.anon_id, label=record.label, split=record.split,
                      n_turns_used=len(turns_for_channel(turns, config.CALLER_CHANNEL)))
        rows.append(feats)
    return pd.DataFrame(rows)


print("Building offline v2 dataset (ground-truth turns)...")
df = build_v2_dataset()
print(f"  {len(df)} calls loaded\n")

# Data-driven hard-silence threshold: mode of floor_db on TRAIN split only, unsupervised.
train_floor = df[df.split == "train"]["floor_db"].round(3)
hard_silence_ref = float(train_floor.mode().iloc[0])
hard_silence_count_train = int((train_floor == hard_silence_ref).sum())
print(f"hard_silence_floor_db reference (mode of train-split floor_db, rounded to 3dp): "
      f"{hard_silence_ref}  (appears in {hard_silence_count_train}/{len(train_floor)} train calls)")
df["hard_silence_flag"] = (df["floor_db"].round(3) == round(hard_silence_ref, 3)).astype(float)
print(f"hard_silence_flag prevalence: train={df[df.split=='train'].hard_silence_flag.mean():.3f}  "
      f"val={df[df.split=='val'].hard_silence_flag.mean():.3f}\n")


# ============================================================================
# Step 3: fit + evaluate each configuration (reusing train.py's train_family,
# writing to a separate experimental artifacts dir - production untouched).
# ============================================================================

print("=" * 100)
print("STEP 3-4: controlled ablation (offline, ground-truth turns, same split as production)")
print("=" * 100)

summary_rows = []
fitted = {}

for name, cols in CONFIGS.items():
    out_dir = EXPERIMENTS_DIR / name
    metadata = train_family(df, cols, out_dir)

    det = Detector(name, out_dir).load()
    fitted[name] = (det, cols)

    train_df = df[df.split == "train"].dropna(subset=cols)
    val_df = df[df.split == "val"].dropna(subset=cols)

    def score(sub_df):
        probas = np.array([det.predict(row.to_dict())[2] for _, row in sub_df.iterrows()])
        preds = probas >= 0.5
        truth = (sub_df.label == "synthetic").values
        confs = np.where(preds, probas, 1 - probas)
        return probas, preds, truth, confs

    val_probas, val_preds, val_truth, val_confs = score(val_df)
    train_probas, _, train_truth, _ = score(train_df)

    val_auc = roc_auc_score(val_truth, val_probas)
    train_auc = roc_auc_score(train_truth, train_probas)
    val_acc = (val_preds == val_truth).mean()
    cm = confusion_matrix(val_truth, val_preds)  # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm.ravel()

    summary_rows.append(dict(
        config=name, n_features=len(cols), train_auc=train_auc, val_auc=val_auc,
        val_acc=val_acc, fp=int(fp), fn=int(fn), tn=int(tn), tp=int(tp),
        avg_confidence=float(val_confs.mean()), n_train=len(train_df), n_val=len(val_df),
    ))

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# ---- specifically: are the 16 known hard-silence FNs still misclassified? ----
print("\n--- Hard-silence FN check (offline ground-truth-turn features) ---")
for name, (det, cols) in fitted.items():
    sub = df[df.anon_id.isin(KNOWN_HARD_SILENCE_FN_IDS)].dropna(subset=cols)
    still_wrong = 0
    for _, row in sub.iterrows():
        is_synth, conf, proba = det.predict(row.to_dict())
        if not is_synth:  # truth is always synthetic for these 16
            still_wrong += 1
    print(f"  {name:32s} still predicted HUMAN (wrong): {still_wrong}/{len(sub)}")


# ============================================================================
# Step 4 (live-pipeline check): the 16 known hard-silence FNs were originally
# found via the LIVE VAD+depad path, not the offline ground-truth turns above -
# verify the fix holds there too, reusing the real production functions directly
# (not going through the FastAPI route, so /detect itself is untouched).
# ============================================================================

print("\n" + "=" * 100)
print("STEP 4 (live pipeline): val-split confusion matrix + hard-silence FN check")
print("=" * 100)

manifest_rows = list(csv.DictReader(open(BASE / "manifest.csv")))
val_manifest = [r for r in manifest_rows if r["split"] == "val"]

live_feats_by_config: dict[str, list[dict]] = {name: [] for name in CONFIGS}
for r in val_manifest:
    aid = r["anon_id"]
    wav_path = BASE / "audio" / f"{aid}.wav"
    payload = base64.b64encode(wav_path.read_bytes()).decode()
    caller_audio, sr, turns = caller_audio_and_turns_from_wav(payload, config.CALLER_CHANNEL)
    caller_audio = caller_audio.astype(np.float64)
    acoustic_turns = _depad_turns(turns, config.VAD_SPEECH_PAD_MS / 1000)
    feats = extract_acoustic_features_v2(caller_audio, sr, acoustic_turns, channel=config.CALLER_CHANNEL)
    if feats is None:
        continue
    feats["hard_silence_flag"] = float(abs(feats["floor_db"] - hard_silence_ref) < 0.05)
    feats["anon_id"], feats["label"] = aid, r["label"]
    for name in CONFIGS:
        live_feats_by_config[name].append(feats)

for name, cols in CONFIGS.items():
    det, _ = fitted[name]
    rows_live = [f for f in live_feats_by_config[name] if all(not np.isnan(f.get(c, np.nan)) for c in cols)]
    if not rows_live:
        print(f"  {name:32s} (no usable live-pipeline rows)")
        continue
    truth = np.array([r["label"] == "synthetic" for r in rows_live])
    preds, confs = [], []
    for r in rows_live:
        is_s, conf, _ = det.predict(r)
        preds.append(is_s)
        confs.append(conf)
    preds = np.array(preds)
    acc = (preds == truth).mean()
    cm = confusion_matrix(truth, preds)
    still_wrong = sum(
        1 for r in rows_live
        if r["anon_id"] in KNOWN_HARD_SILENCE_FN_IDS and not det.predict(r)[0]
    )
    n_known = sum(1 for r in rows_live if r["anon_id"] in KNOWN_HARD_SILENCE_FN_IDS)
    print(f"  {name:32s} live_acc={acc:.3f}  avg_conf={np.mean(confs):.3f}  "
          f"cm(TN,FP,FN,TP)={cm.ravel().tolist()}  hard_silence_FNs_still_wrong={still_wrong}/{n_known}")


# ============================================================================
# Step 5: feature-level analysis for configs C and D (standardized coefficients -
# scaler is fit before the model, so raw coef_ IS already in standardized units).
# ============================================================================

print("\n" + "=" * 100)
print("STEP 5: standardized coefficients (LogisticRegression on StandardScaler output)")
print("=" * 100)
for name in ("C_all_variability", "D_all_variability_hard_silence"):
    det, cols = fitted[name]
    coefs = det.model.coef_[0]
    order = np.argsort(-np.abs(coefs))
    print(f"\n--- {name} ---")
    for i in order:
        print(f"  {cols[i]:32s} coef={coefs[i]:+.3f}")


# ============================================================================
# Step 6: turn-count bucket analysis (terciles of n_turns_used on val split),
# comparing v1 baseline vs the best-performing v2 config from the summary above.
# ============================================================================

print("\n" + "=" * 100)
print("STEP 6: performance by turn-count bucket (val split, terciles)")
print("=" * 100)

best_v2_name = summary[summary.config != "A_v1_baseline"].sort_values("val_auc", ascending=False).iloc[0]["config"]
print(f"Best v2 config by val AUC: {best_v2_name}\n")

val_all = df[df.split == "val"].copy()
terciles = val_all["n_turns_used"].quantile([1 / 3, 2 / 3]).values
val_all["turn_bucket"] = pd.cut(
    val_all["n_turns_used"], bins=[-np.inf, terciles[0], terciles[1], np.inf],
    labels=["low", "medium", "high"],
)
print(f"Tercile cut points (n_turns_used): {terciles}\n")

for name in ("A_v1_baseline", best_v2_name):
    det, cols = fitted[name]
    sub = val_all.dropna(subset=cols)
    rows = []
    for bucket in ("low", "medium", "high"):
        b = sub[sub.turn_bucket == bucket]
        if b.empty:
            continue
        preds = np.array([det.predict(row.to_dict())[0] for _, row in b.iterrows()])
        truth = (b.label == "synthetic").values
        rows.append((bucket, len(b), (preds == truth).mean()))
    print(f"  {name}:")
    for bucket, n, acc in rows:
        print(f"    {bucket:8s} n={n:3d}  accuracy={acc:.3f}")

print("\nDone.")
