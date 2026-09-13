# Resonance detector

One signal in the human-vs-synthetic ensemble: flags calls whose formant
(vocal tract resonance) and pitch behavior looks physically implausible
for a real human vocal tract — never looks at what's being said.

Validated on the challenge's train/val split: **AUC 0.94-0.95** on val
(speaker-disjoint, never seen in training), using either the dataset's
`turns.json` or this module's own VAD — both give basically the same
result, so it holds up without the ground-truth turns data that won't
exist at real inference time.

## Start here

Only 4 files matter if you're wiring this into the backend — everything
else is for retraining/research and can be ignored:

```
resonance-detector/
├── detector.py         ← import this
├── resonance_core.py   ← detector.py depends on this
├── vad.py               ← detector.py depends on this
├── model.joblib         ← the trained model (2KB, no retraining needed)
└── training/            ← ignore unless you're retraining, see below
```

```python
from detector import detect_from_base64  # or detect_from_wav_bytes

result = detect_from_base64(wav_b64_string)
# {"is_synthetic": True, "confidence": 0.87}
```

`confidence` is this module's own synthetic-probability score (0-1) —
meant to be combined with the team's other detectors (voting, averaging,
or a meta-classifier), not returned as the final verdict on its own.

CLI smoke test:
```sh
python detector.py path/to/call.wav
```

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Needs the challenge audio unzipped at `../hackmty26/audio/` (see the
main repo's data submodule) — only for retraining/exploration, not for
`detector.py` itself at real inference time.

## `training/` — only if you're retraining

Regenerates the CSVs `detector.py`'s model was built from, and rebuilds
the model. Ordinary use of the detector never touches these.

| File | What it does |
| --- | --- |
| `training/resonance_features.py` | Builds training CSVs using the dataset's ground-truth `turns.json` (only for train/val, not real calls). |
| `training/train_model.py` | Trains the logistic regression on the extracted features, saves `../model.joblib`. |
| `training/build_val_with_vad.py` | One-off script used to confirm VAD-based features match turns.json-based ones (they do). |

If more/different train data shows up, regenerate features and retrain
(run from anywhere, paths resolve on their own):

```sh
python training/resonance_features.py --n-per-class 999 --split train --out resonance_features_train_full.csv
python training/resonance_features.py --n-per-class 999 --split val --out resonance_features_val_full.csv
python training/train_model.py
```

## What each of the 7 features means

See `resonance_core.py` — every feature has a one-line docstring
explaining the physical idea behind it (jitter, formant/pitch coupling,
long-timescale drift, vocoder frame periodicity).
