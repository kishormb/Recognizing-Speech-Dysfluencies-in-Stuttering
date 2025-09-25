#!/usr/bin/env python3
"""
Dysfluency pipeline with feature importance (Streamlit)
- Put audio files under segrigated_samples/<label>/*.wav (or mp3, m4a...)
- Run: streamlit run pipeline_with_feature_importance_and_confusion.py

Additions in this version:
- Confusion matrix plotting (interactive Plotly heatmap) for each model & dataset (before/after)
- Classification report (precision/recall/f1) saved and shown
- Permutation importance for RandomForest (after-cleaning) with CSV export
- Minor robustness fixes and better logging
"""

import os
import re
import warnings
import logging
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# -------------------------
# Directories & logging
# -------------------------
DATA_DIR = "segrigated_samples"
OUTPUT_DIR = "output_results"
CACHE_DIR = "cache_features"
CLEAR_DIR = "clear_audio"

for d in [OUTPUT_DIR, CACHE_DIR, CLEAR_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "pipeline_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# Required third-party libs
# -------------------------
try:
    import librosa
    import noisereduce as nr
    import soundfile as sf
except Exception as e:
    st.error("Please install audio libs: pip install librosa noisereduce soundfile")
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
    from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.inspection import permutation_importance
except Exception:
    st.error("Please install scikit-learn: pip install scikit-learn")
    raise

import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Constants
# -------------------------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
TARGET_SR = 16000
MFCC_N = 20       # number of MFCCs
# audio features: MFCC mean/std (20*2)=40,
# delta mean/std (20*2)=40,
# delta2 mean/std (20*2)=40  -> 120
# chroma mean/std (12*2)=24  -> total audio = 144
AUDIO_FEATURE_LEN = (MFCC_N * 2) * 3 + 12 * 2  # 144
TEXT_FEATURE_LEN = 5
TOTAL_FEATURE_LEN = AUDIO_FEATURE_LEN + TEXT_FEATURE_LEN  # 149

# -------------------------
# Utility helpers
# -------------------------
def list_audio_files(root: str):
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(r, f))
    return sorted(files)


def load_audio(path: str, sr: int = TARGET_SR):
    try:
        y, s = librosa.load(path, sr=sr, mono=True)
        return y, s
    except Exception as e:
        logging.error(f"load_audio fail {path}: {e}")
        return None, None


def cached_path(path: str):
    return os.path.normpath(path)


def save_cache(arr: np.ndarray, cache_path: str):
    np.save(cache_path, arr)


def load_cache(cache_path: str):
    try:
        return np.load(cache_path, allow_pickle=True)
    except Exception:
        return None

# -------------------------
# Audio cleaning + caching
# -------------------------
def clean_audio_and_cache(in_path: str):
    """
    Writes cleaned wav to CLEAR_DIR/<basename>.wav and returns path.
    If cleaned exists already, returns existing cleaned path.
    """
    base = os.path.basename(in_path).rsplit(".", 1)[0]
    out_path = os.path.join(CLEAR_DIR, f"{base}.wav")
    out_path = cached_path(out_path)
    if os.path.exists(out_path):
        return out_path
    y, sr = load_audio(in_path, sr=TARGET_SR)
    if y is None:
        return None
    try:
        y_clean = nr.reduce_noise(y=y, sr=sr)
        y_clean = librosa.util.normalize(y_clean)
        sf.write(out_path, y_clean, sr)
        return out_path
    except Exception as e:
        logging.error(f"clean_audio fail {in_path}: {e}")
        return None

# -------------------------
# Low-level audio metrics
# -------------------------
def snr_db(y: np.ndarray):
    """Segmental energy-based SNR (dB) using frame_length/hop_length correctly."""
    frame_length = int(0.025 * TARGET_SR)  # 25 ms
    hop_length = int(0.010 * TARGET_SR)    # 10 ms
    if y is None or len(y) < frame_length:
        return 0.0
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    # bottom 25% energy frames as noise
    noise_mask = energy < np.percentile(energy, 25)
    if noise_mask.sum() == 0:
        return 0.0
    noise_power = np.mean(energy[noise_mask])
    signal_power = np.mean(energy)
    return 10.0 * np.log10(signal_power / (noise_power + 1e-10))


def spectral_flatness_mean(y: np.ndarray):
    try:
        S = np.abs(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)
        return float(np.mean(flatness))
    except Exception:
        return 0.0


def high_freq_energy_ratio(y: np.ndarray, sr: int):
    try:
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        high_mask = freqs > 4000
        total_energy = np.sum(np.abs(fft)**2)
        high_energy = np.sum(np.abs(fft[high_mask])**2)
        return float(high_energy / (total_energy + 1e-10))
    except Exception:
        return 0.0

# -------------------------
# Text helpers (if transcripts available)
# -------------------------
def repetition_stats_from_text(text: str):
    if not text:
        return {"repetition_count": 0, "repetition_ratio": 0.0, "unique_ratio": 0.0}
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 1:
        return {"repetition_count": 0, "repetition_ratio": 0.0, "unique_ratio": 0.0}
    word_counts = Counter(words)
    repeats = sum(count - 1 for count in word_counts.values() if count > 1)
    ratio = repeats / len(words)
    unique_ratio = len(set(words)) / len(words)
    return {"repetition_count": float(repeats), "repetition_ratio": float(ratio), "unique_ratio": float(unique_ratio)}

# -------------------------
# Feature extraction (audio + text)
# -------------------------
def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Produces deterministic audio vector of length AUDIO_FEATURE_LEN:
    [mfcc_mean(20), mfcc_std(20), delta_mean(20), delta_std(20), delta2_mean(20), delta2_std(20),
     chroma_mean(12), chroma_std(12)]
    => total = (20*2)*3 + 12*2 = 144
    """
    if y is None:
        return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        def stat_pair(mat):
            return np.hstack([np.mean(mat, axis=1), np.std(mat, axis=1)])

        mfcc_stats = stat_pair(mfcc)
        delta_stats = stat_pair(delta)
        delta2_stats = stat_pair(delta2)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stats = np.hstack([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

        feats = np.hstack([mfcc_stats, delta_stats, delta2_stats, chroma_stats]).astype(np.float32)
        # ensure length
        if feats.size != AUDIO_FEATURE_LEN:
            out = np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)
            out[:min(feats.size, AUDIO_FEATURE_LEN)] = feats[:AUDIO_FEATURE_LEN]
            return out
        return feats
    except Exception as e:
        logging.error(f"extract_audio_features error: {e}")
        return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)


def extract_text_features(text: str) -> np.ndarray:
    if not text:
        return np.zeros(TEXT_FEATURE_LEN, dtype=np.float32)
    rep_stats = repetition_stats_from_text(text)
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = float(len(words))
    return np.array([
        float(len(text)),
        word_count,
        rep_stats["repetition_count"],
        rep_stats["repetition_ratio"],
        rep_stats["unique_ratio"]
    ], dtype=np.float32)


def extract_features(y: np.ndarray, sr: int, transcript: str = "") -> np.ndarray:
    audio_feats = extract_audio_features(y, sr)
    text_feats = extract_text_features(transcript)
    feats = np.hstack([audio_feats, text_feats]).astype(np.float32)
    if feats.size != TOTAL_FEATURE_LEN:
        out = np.zeros(TOTAL_FEATURE_LEN, dtype=np.float32)
        out[:min(feats.size, TOTAL_FEATURE_LEN)] = feats[:TOTAL_FEATURE_LEN]
        return out
    return feats

# -------------------------
# Feature name generator for feature importance
# -------------------------
def make_feature_names():
    names = []
    for w in ["mfcc", "delta", "delta2"]:
        names += [f"{w}_mean_{i}" for i in range(MFCC_N)]
        names += [f"{w}_std_{i}" for i in range(MFCC_N)]
    names += [f"chroma_mean_{i}" for i in range(12)]
    names += [f"chroma_std_{i}" for i in range(12)]
    # text features
    names += ["transcript_length", "word_count", "repetition_count", "repetition_ratio", "unique_ratio"]
    # Truncate/pad to match length
    if len(names) > TOTAL_FEATURE_LEN:
        names = names[:TOTAL_FEATURE_LEN]
    elif len(names) < TOTAL_FEATURE_LEN:
        names += [f"pad_{i}" for i in range(TOTAL_FEATURE_LEN - len(names))]
    return names

FEATURE_NAMES = make_feature_names()

# -------------------------
# Plotting helpers
# -------------------------
def plot_accuracies(df: pd.DataFrame):
    fig = px.bar(df, x="model", y="accuracy", color="dataset", barmode="group",
                 title="Model Accuracy Comparison (%)")
    return fig


def plot_test_losses_df(df: pd.DataFrame):
    fig = px.bar(df, x="model", y="test_loss", color="dataset", barmode="group",
                 title="Test Log Loss Comparison")
    return fig


def plot_roc(y_true, y_score_dict, n_classes, class_names, title_suffix=""):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig = go.Figure()
    auc_rows = []
    colors = px.colors.qualitative.Dark24
    for m_idx, (mname, y_score) in enumerate(y_score_dict.items()):
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                trace_name = f"{mname} - {class_names[i]} (AUC = {roc_auc:.2f})"
                color = colors[(m_idx * n_classes + i) % len(colors)]
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=trace_name,
                                         line=dict(color=color, width=2)))
                auc_rows.append({"model": mname, "class": class_names[i], "auc": float(roc_auc)})
            except Exception as e:
                logging.error(f"plot_roc error {mname} class {i}: {e}")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='black', dash='dash'), name='Chance (AUC = 0.5)'))
    fig.update_layout(title=f"Multi-Class ROC Curves {title_suffix}", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", width=900, height=700)
    return fig, pd.DataFrame(auc_rows)


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        # Save CSV
        return cm_df
    except Exception as e:
        logging.error(f"plot_confusion_matrix error: {e}")
        return None

def plot_confusion_heatmap(cm_df, title="Confusion matrix"):
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=list(cm_df.columns),
        y=list(cm_df.index),
        hoverongaps=False,
        text=cm_df.values,
        texttemplate="%{text}",
        colorscale='Blues'))
    fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='Actual', width=700, height=600)
    return fig

# -------------------------
# Main pipeline
# -------------------------

def run_pipeline():
    st.title("Dysfluency Classification & Feature Importance Pipeline")

    files = list_audio_files(DATA_DIR)
    if not files:
        st.error(f"No audio files found under '{DATA_DIR}'. Create subfolders named by class label and add files.")
        return

    st.write(f"Found {len(files)} files. Preprocessing...")

    per_file = []
    raw_paths = []
    clean_paths = []
    transcripts = []  # placeholder - empty transcripts (Whisper not integrated here)
    labels = []
    skipped = 0

    # Preprocess each file: compute quick stats, clean & cache
    for f in tqdm(files, desc="Preprocessing"):
        label = os.path.basename(os.path.dirname(f)) or "unknown"
        y_raw, sr_raw = load_audio(f, sr=TARGET_SR)
        if y_raw is None:
            skipped += 1
            continue

        duration = float(len(y_raw) / sr_raw) if sr_raw else 0.0
        snr_before = float(snr_db(y_raw))
        flat_before = float(spectral_flatness_mean(y_raw))
        hf_before = float(high_freq_energy_ratio(y_raw, sr_raw))

        # cleaning (cached)
        cleaned = clean_audio_and_cache(f)
        if cleaned is None:
            # if cleaning fails, fallback to original path (but note)
            cleaned = f

        y_clean, sr_clean = load_audio(cleaned, sr=TARGET_SR)
        if y_clean is None:
            skipped += 1
            continue

        snr_after = float(snr_db(y_clean))
        flat_after = float(spectral_flatness_mean(y_clean))
        hf_after = float(high_freq_energy_ratio(y_clean, sr_clean))

        # transcripts not provided -> keep empty string
        transcript = ""
        transcripts.append(transcript)

        per_file.append({
            "file": os.path.basename(f),
            "label": label,
            "duration_sec": duration,
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "spectral_flatness_before": flat_before,
            "spectral_flatness_after": flat_after,
            "hf_energy_ratio_before": hf_before,
            "hf_energy_ratio_after": hf_after,
            "transcript": transcript
        })

        raw_paths.append(f)
        clean_paths.append(cleaned)
        labels.append(label)

    st.write(f"Processed {len(per_file)} files, skipped {skipped}.")
    analysis_df = pd.DataFrame(per_file)
    analysis_csv = os.path.join(OUTPUT_DIR, "per_file_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    st.write("Per-file analysis saved to:", analysis_csv)
    st.dataframe(analysis_df)

    # -------------------------
    # Feature extraction caching
    # -------------------------
    def cached_extract_features(path, transcript, suffix):
        """Cache feature vectors to speed repeated runs."""
        base = os.path.basename(path).rsplit(".", 1)[0]
        cache_file = os.path.join(CACHE_DIR, f"{base}_{suffix}_feats.npy")
        cache_file = cached_path(cache_file)
        cached = load_cache(cache_file)
        if cached is not None:
            return np.array(cached)
        y, sr = load_audio(path, sr=TARGET_SR)
        feats = extract_features(y, sr, transcript)
        save_cache(feats, cache_file)
        return feats

    X_before = []
    X_after = []
    y_labels = []

    st.write("Extracting features (may take a while first run)...")
    for raw_p, clean_p, txt, lbl in tqdm(zip(raw_paths, clean_paths, transcripts, labels),
                                        total=len(raw_paths), desc="Feature extraction"):
        feats_raw = cached_extract_features(raw_p, txt, "raw")
        feats_clean = cached_extract_features(clean_p, txt, "clean")
        X_before.append(feats_raw)
        X_after.append(feats_clean)
        y_labels.append(lbl)

    X_before = np.vstack(X_before) if X_before else np.empty((0, TOTAL_FEATURE_LEN))
    X_after = np.vstack(X_after) if X_after else np.empty((0, TOTAL_FEATURE_LEN))

    if X_before.shape[0] == 0 or X_after.shape[0] == 0:
        st.error("No features extracted. Check audio and paths.")
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    st.write(f"Feature shapes -> before: {X_before.shape}, after: {X_after.shape}; classes: {class_names}")

    # Scale features
    scaler_b = StandardScaler().fit(X_before)
    scaler_a = StandardScaler().fit(X_after)
    Xb_scaled = scaler_b.transform(X_before)
    Xa_scaled = scaler_a.transform(X_after)

    # Train-test split (stratified)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    sizes_df = pd.DataFrame({
        "dataset": ["before", "after"],
        "train_size": [len(Xb_train), len(Xa_train)],
        "test_size": [len(Xb_test), len(Xa_test)]
    })
    sizes_df.to_csv(os.path.join(OUTPUT_DIR, "train_test_sizes.csv"), index=False)
    st.write("Train/Test sizes")
    st.dataframe(sizes_df)
    st.plotly_chart(px.bar(sizes_df.melt(id_vars="dataset", value_vars=["train_size", "test_size"],
                                         var_name="split", value_name="count"),
                          x="dataset", y="count", color="split", barmode="group",
                          title="Train vs Test sizes"), use_container_width=True)

    # -------------------------
    # Models setup
    # -------------------------
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    }

    # store results and preds for later analysis
    results = {
        "before": {"probs": {}, "test_loss": {}, "preds": {}},
        "after": {"probs": {}, "test_loss": {}, "preds": {}, "rf_model": None}
    }
    metrics_rows = []

    def train_and_eval(set_name, X_tr, X_te, y_tr, y_te):
        for mname, model in models.items():
            st.write(f"Training {mname} on {set_name} data...")
            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                probs = model.predict_proba(X_te)
                acc = float(accuracy_score(y_te, preds) * 100.0)
                loss = float(log_loss(y_te, probs))
                results[set_name]["probs"][mname] = probs
                results[set_name]["test_loss"][mname] = loss
                results[set_name]["preds"][mname] = preds
                metrics_rows.append({"dataset": set_name, "model": mname, "accuracy": acc, "test_loss": loss})
                st.write(f"{set_name} - {mname}  Acc={acc:.2f}%  LogLoss={loss:.4f}")
                # Save trained RF model for later feature importance (only RF on after)
                if set_name == "after" and mname == "RandomForest":
                    results[set_name]["rf_model"] = model
            except Exception as e:
                logging.error(f"Training error {mname} on {set_name}: {e}")
                st.warning(f"Training failed for {mname} on {set_name}")

    # Train/eval on before & after
    train_and_eval("before", Xb_train, Xb_test, yb_train, yb_test)
    train_and_eval("after", Xa_train, Xa_test, ya_train, ya_test)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    st.write("Model metrics (accuracy% and log loss)")
    st.dataframe(metrics_df)

    # Accuracy bar
    st.plotly_chart(plot_accuracies(metrics_df), use_container_width=True)
    # Test loss bar
    st.plotly_chart(plot_test_losses_df(metrics_df), use_container_width=True)

    # -------------------------
    # ROC curves for before & after
    # -------------------------
    if results["before"]["probs"]:
        try:
            fig_b, auc_b_df = plot_roc(yb_test, results["before"]["probs"], n_classes, class_names, title_suffix="(Before)")
            st.write("ROC curves (Before cleaning)")
            st.plotly_chart(fig_b, use_container_width=True)
            auc_b_df.to_csv(os.path.join(OUTPUT_DIR, "auc_before.csv"), index=False)
            fig_b.write_html(os.path.join(OUTPUT_DIR, "roc_before.html"))
        except Exception as e:
            logging.error(f"ROC before error: {e}")

    if results["after"]["probs"]:
        try:
            fig_a, auc_a_df = plot_roc(ya_test, results["after"]["probs"], n_classes, class_names, title_suffix="(After)")
            st.write("ROC curves (After cleaning)")
            st.plotly_chart(fig_a, use_container_width=True)
            auc_a_df.to_csv(os.path.join(OUTPUT_DIR, "auc_after.csv"), index=False)
            fig_a.write_html(os.path.join(OUTPUT_DIR, "roc_after.html"))
        except Exception as e:
            logging.error(f"ROC after error: {e}")

    # -------------------------
    # Confusion matrices & classification reports
    # -------------------------
    cm_rows = []
    for set_name, (X_te, y_te) in zip(["before", "after"], [(Xb_test, yb_test), (Xa_test, ya_test)]):
        st.write(f"Confusion matrices & reports for {set_name} data")
        for mname in results[set_name]["preds"]:
            try:
                preds = results[set_name]["preds"][mname]
                cm_df = plot_confusion_matrix(y_te, preds, class_names, title=f"{mname} - {set_name}")
                if cm_df is None:
                    continue
                cm_csv = os.path.join(OUTPUT_DIR, f"confusion_{set_name}_{mname}.csv")
                cm_df.to_csv(cm_csv)
                fig_cm = plot_confusion_heatmap(cm_df, title=f"{mname} - {set_name} Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

                # classification report
                crep = classification_report(y_te, preds, target_names=class_names, output_dict=True)
                crep_df = pd.DataFrame(crep).T
                crep_csv = os.path.join(OUTPUT_DIR, f"class_report_{set_name}_{mname}.csv")
                crep_df.to_csv(crep_csv)
                st.write(f"{mname} - {set_name} Classification Report")
                st.dataframe(crep_df)

                # aggregate for summary
                cm_rows.append({"dataset": set_name, "model": mname, "confusion_csv": cm_csv, "report_csv": crep_csv})
            except Exception as e:
                logging.error(f"Confusion/report error for {mname} on {set_name}: {e}")
                st.warning(f"Failed to compute confusion/report for {mname} on {set_name}")

    cm_summary_df = pd.DataFrame(cm_rows)
    if not cm_summary_df.empty:
        cm_summary_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrices_summary.csv"), index=False)

    # -------------------------
    # Feature importance (RandomForest on AFTER features)
    # -------------------------
    if results["after"]["rf_model"] is not None:
        rf = results["after"]["rf_model"]
        try:
            # built-in importances
            importances = rf.feature_importances_
            fname = FEATURE_NAMES[: len(importances)]
            feat_df = pd.DataFrame({"feature": fname, "importance": importances})
            feat_df = feat_df.sort_values("importance", ascending=False).reset_index(drop=True)
            feat_csv = os.path.join(OUTPUT_DIR, "feature_importances_after_rf.csv")
            feat_df.to_csv(feat_csv, index=False)
            st.write("Top-20 feature importances (after-cleaning):")
            st.dataframe(feat_df.head(20))
            st.plotly_chart(px.bar(feat_df.head(20).sort_values("importance"), x="importance", y="feature",
                                   orientation="h", title="Top 20 Feature Importances ( after)"), use_container_width=True)

            # # permutation importance (more robust)
            # st.write("Computing permutation importance (may take a while)...")
            # perm_res = permutation_importance(rf, Xa_test, ya_test, n_repeats=20, random_state=42, n_jobs=1)
            # perm_df = pd.DataFrame({"feature": fname, "perm_importance_mean": perm_res.importances_mean,
            #                          "perm_importance_std": perm_res.importances_std})
            # perm_df = perm_df.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
            # perm_csv = os.path.join(OUTPUT_DIR, "permutation_importances_after_rf.csv")
            # perm_df.to_csv(perm_csv, index=False)
            # st.write("Top-20 permutation importances:")
            # st.dataframe(perm_df.head(20))
            #st.plotly_chart(px.bar(perm_df.head(20).sort_values("perm_importance_mean"), x="perm_importance_mean", y="feature",
             #                      orientation="h", title="Top 20 Permutation Importances (after)"), use_container_width=True)

        except Exception as e:
            logging.error(f"Feature importance error: {e}")
            st.warning("Failed to compute feature importances.")
    else:
        st.info("RandomForest model on 'after' data not available for feature importance.")

    # -------------------------
    # Save summary CSVs
    # -------------------------
    # analysis_df_out = os.path.join(OUTPUT_DIR, "per_file_analysis_with_preds.csv")
    # try:
    #     # If RF after exists, produce predictions/confidence on all after samples
    #     if results["after"].get("rf_model") is not None:
    #         rf = results["after"]["rf_model"]
    #         all_after_scaled = scaler_a.transform(X_after)
    #         all_probs = rf.predict_proba(all_after_scaled)
    #         all_preds = rf.predict(all_after_scaled)
    #         confidence = np.max(all_probs, axis=1)
    #         pred_labels = le.inverse_transform(all_preds)
    #         analysis_df["predicted_label_after_rf"] = pred_labels
    #         analysis_df["confidence_after_rf"] = confidence
    #     analysis_df.to_csv(analysis_csv, index=False)
    # except Exception as e:
    #     logging.error(f"Saving predicted analysis csv error: {e}")

   # st.success("Pipeline complete — check the output_results/ folder for CSVs and ROC HTMLs.")
    #st.write(f"Saved: {os.listdir(OUTPUT_DIR)[:50]}")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    run_pipeline()
