"""LSTM bake-off model for the NASDAQ-100 next-day direction task.

Trains a small LSTM on 30-day sequences of the shared feature set and evaluates
it on the *same* walk-forward folds and embargo as the rest of the bake-off
(imported from src.evaluation), so its numbers are comparable to the tree models.

Why this doesn't go through evaluate_model()
---------------------------------------------
evaluate_model() masks individual rows, which would shatter time-ordered
sequences across fold boundaries and gives the wrapper no ticker/date info to
rebuild per-ticker windows. Instead this module reuses the harness's FOLDS and
_split_fold() to assign whole sequences (keyed on their last day) to
train/val/test, preserving identical year boundaries + the 5-day embargo, and
writes results in the standard format via save_results().

Anti-leakage guarantees
-----------------------
  * Sequences never span two tickers (built per ticker).
  * A sequence's features come only from days <= its last day; its label is the
    next-day direction of that last day (encoded by build_feature_matrix).
  * Feature scaling is fit on the training fold only, then applied to val/test.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.evaluation import FOLDS, _split_fold, summarize_results, save_results
from src.features import build_feature_matrix

logger = logging.getLogger(__name__)

SEQ_LEN = 30
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 256
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
SEED = 42
MIN_RECALL = 0.40  # recall floor used when tuning the decision threshold


# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------
def build_sequences(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, seq_len=SEQ_LEN):
    """Turn the flat feature matrix into per-ticker sliding windows.

    Returns (seqs, labels, ends):
      seqs   : float32 array (N, seq_len, n_features)
      labels : int64 array  (N,)
      ends   : DataFrame [date, ticker] -- last day of each window, used for
               fold assignment via the shared _split_fold().
    """
    frame = meta.copy()
    frame["_y"] = y.values
    feat_cols = list(X.columns)
    frame[feat_cols] = X.values

    seqs, labels, end_dates, end_tickers = [], [], [], []
    for ticker, g in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        g = g.reset_index(drop=True)
        feats = g[feat_cols].to_numpy(dtype="float32")
        ys = g["_y"].to_numpy(dtype="int64")
        dates = g["date"].to_numpy()
        for i in range(seq_len - 1, len(g)):
            seqs.append(feats[i - seq_len + 1 : i + 1])
            labels.append(ys[i])
            end_dates.append(dates[i])
            end_tickers.append(ticker)

    if not seqs:
        raise ValueError("No sequences built -- need at least seq_len rows per ticker.")

    ends = pd.DataFrame({"date": pd.to_datetime(end_dates), "ticker": end_tickers})
    logger.info("Built %d sequences of shape (%d, %d).", len(seqs), seq_len, len(feat_cols))
    return np.stack(seqs), np.asarray(labels), ends


def _fit_scaler(train_seqs):
    flat = train_seqs.reshape(-1, train_seqs.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype("float32"), std.astype("float32")


def _apply_scaler(seqs, mean, std):
    return (seqs - mean) / std


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def _build_torch():
    import torch
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=HIDDEN,
                num_layers=NUM_LAYERS,
                batch_first=True,
                dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            )
            self.dropout = nn.Dropout(DROPOUT)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(self.dropout(last)).squeeze(-1)

    return torch, nn, LSTMClassifier


def _train_loop(model, opt, loss_fn, Xtr_t, ytr_t, Xva_t, yva_t, torch):
    """Shared training loop with early stopping on validation loss.

    Returns (best_val_loss, best_epoch, val_loss_curve).
    """
    n = len(Xtr_t)
    best_val, best_state, best_epoch, no_improve, curve = float("inf"), None, -1, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()
        curve.append(round(val_loss, 5))
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info("Early stopping at epoch %d (best epoch %d, val_loss %.5f).",
                            epoch, best_epoch, best_val)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, best_epoch, curve


def _train_one_fold(train, val, test):
    """Base LSTM: fixed 0.5 threshold, unweighted BCE. Returns a metrics dict."""
    import torch
    from sklearn.metrics import accuracy_score, precision_score, f1_score

    torch_mod, nn, LSTMClassifier = _build_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr, ytr = train
    Xva, yva = val
    Xte, yte = test
    mean, std = _fit_scaler(Xtr)
    Xtr, Xva, Xte = (_apply_scaler(a, mean, std) for a in (Xtr, Xva, Xte))

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    model = LSTMClassifier(Xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_epoch, curve = _train_loop(
        model, opt, loss_fn, t(Xtr), t(ytr.astype("float32")), t(Xva), t(yva.astype("float32")), torch
    )

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(t(Xte))).numpy()
    y_pred = (probs >= 0.5).astype("int64")
    return {
        "accuracy": float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "f1": float(f1_score(yte, y_pred, zero_division=0)),
        "best_epoch": int(best_epoch),
        "epochs_trained": len(curve),
        "best_val_loss": float(best_val),
        "val_loss_curve": curve,
    }


def _train_one_fold_tuned(train, val, test):
    """Iteration 2: class-weighted loss + validation-tuned decision threshold
    (maximise precision subject to recall >= MIN_RECALL on the val fold)."""
    import torch
    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

    torch_mod, nn, LSTMClassifier = _build_torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr, ytr = train
    Xva, yva = val
    Xte, yte = test
    mean, std = _fit_scaler(Xtr)
    Xtr, Xva, Xte = (_apply_scaler(a, mean, std) for a in (Xtr, Xva, Xte))

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    model = LSTMClassifier(Xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    _train_loop(model, opt, loss_fn, t(Xtr), t(ytr.astype("float32")), t(Xva), t(yva.astype("float32")), torch)

    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(t(Xva))).numpy()
        pt = torch.sigmoid(model(t(Xte))).numpy()
    best_thr, best_p = 0.5, -1.0
    for thr in np.arange(0.30, 0.96, 0.01):
        pred = (pv >= thr).astype(int)
        if pred.sum() == 0 or recall_score(yva, pred, zero_division=0) < MIN_RECALL:
            continue
        p = precision_score(yva, pred, zero_division=0)
        if p > best_p:
            best_p, best_thr = p, thr
    y_pred = (pt >= best_thr).astype("int64")
    return {
        "accuracy": float(accuracy_score(yte, y_pred)),
        "precision": float(precision_score(yte, y_pred, zero_division=0)),
        "f1": float(f1_score(yte, y_pred, zero_division=0)),
        "threshold": float(best_thr),
    }


# --------------------------------------------------------------------------
# Orchestration -- returns results in the shared harness format
# --------------------------------------------------------------------------
def run_lstm_bakeoff(X, y, meta, tuned=False, seq_len=SEQ_LEN):
    """Build sequences and run all walk-forward folds.

    Returns a list of per-fold dicts with the same keys as evaluate_model()
    (fold, train_end_year, test_year, n_train, n_test, accuracy, precision, f1,
    train_time_seconds) so it plugs straight into summarize_results/save_results.
    """
    seqs, labels, ends = build_sequences(X, y, meta, seq_len)
    train_fn = _train_one_fold_tuned if tuned else _train_one_fold

    per_fold = []
    for i, fold in enumerate(FOLDS, start=1):
        tr, va, te = _split_fold(ends, fold)
        if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
            logger.warning("Fold %d skipped — empty split (train=%d val=%d test=%d).",
                           i, tr.sum(), va.sum(), te.sum())
            continue
        logger.info("Fold %d | train=%d val=%d test=%d (test year %d).",
                    i, tr.sum(), va.sum(), te.sum(), fold["test_year"])
        t0 = time.time()
        m = train_fn((seqs[tr], labels[tr]), (seqs[va], labels[va]), (seqs[te], labels[te]))
        elapsed = time.time() - t0
        m.update({
            "fold": i,
            "train_end_year": fold["train_end"],
            "test_year": fold["test_year"],
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "train_time_seconds": float(elapsed),
        })
        per_fold.append(m)
        logger.info("Fold %d done | acc=%.4f prec=%.4f f1=%.4f (%.1fs).",
                    i, m["accuracy"], m["precision"], m["f1"], elapsed)

    if not per_fold:
        raise RuntimeError("No folds produced results.")
    logger.info("All folds complete. Returning %d fold results.", len(per_fold))
    return per_fold


def main():
    """CLI entry point: load data/raw/ via the shared pipeline, run, and save."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    from src.pipeline import build_bakeoff_frame

    logger.info("Loading data via build_bakeoff_frame ...")
    df = build_bakeoff_frame()
    X, y, meta = build_feature_matrix(df)
    logger.info("Feature matrix: %d rows x %d features.", X.shape[0], X.shape[1])
    results = run_lstm_bakeoff(X, y, meta)
    summary = summarize_results(results)
    save_results("lstm", results)
    logger.info("Saved results/metrics/lstm.json | mean precision=%.4f",
                summary["mean_precision"])


if __name__ == "__main__":
    main()
