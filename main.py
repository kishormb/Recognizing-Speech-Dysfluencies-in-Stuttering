import os
import io
import json
from typing import List, Tuple, Dict
import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import noisereduce as nr
import soundfile as sf
from collections import Counter

# Assuming OpenAI Whisper is installed: pip install openai-whisper
try:
    import whisper
except ImportError:
    whisper = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================
# Global Directories
# =============================
DATA_DIR = "segrigated_samples"
OUTPUT_DIR = "output_results"
CACHE_DIR = "cache_features"
CLEAR_DIR = "clear_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CLEAR_DIR, exist_ok=True)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
TARGET_SR = 16000

# Load Whisper model if available
try:
    if whisper:
        whisper_model = whisper.load_model("base")
        WHISPER_AVAILABLE = True
    else:
        WHISPER_AVAILABLE = False
        print("Warning: Whisper not installed. Transcriptions will be empty.")
except (ImportError, Exception):
    whisper_model = None
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not installed or failed to load. Transcriptions will be empty.")

# =============================
# Utility Functions
# =============================
def list_audio_files(root: str) -> List[str]:
    """List all audio files without conversion - keep original formats."""
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(r, f))
    print(f"Found {len(files)} audio files in {root}")
    return files

def load_audio_any(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio using librosa only - no pydub conversion. Skip if fails."""
    try:
        y, s = librosa.load(path, sr=sr, mono=True)
        if len(y) < sr * 0.1:  # Skip files shorter than 100ms
            print(f"[Warning] Audio too short: {path}")
            return None, None
        return y, s
    except Exception as e:
        print(f"[Error loading {path}] {e} - Skipping file.")
        return None, None

def cache_or_compute(cache_path: str, func):
    """Cache computation results."""
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data.item() if data.ndim == 0 else data
    result = func()
    if isinstance(result, np.ndarray):
        np.save(cache_path, result)
    else:
        np.save(cache_path, np.array([result], dtype=object))
    return result

# =============================
# Audio Cleaning + Metrics
# =============================
def clean_and_save_audio(file_path: str) -> str:
    """Clean audio using noise reduction and normalization."""
    y, sr = load_audio_any(file_path, sr=TARGET_SR)
    if y is None:
        return None
    try:
        y_clean = nr.reduce_noise(y=y, sr=sr)
        y_clean = librosa.util.normalize(y_clean)
        out_path = os.path.join(CLEAR_DIR, os.path.basename(file_path).rsplit('.', 1)[0] + '.wav')
        sf.write(out_path, y_clean, sr)
        return out_path
    except Exception as e:
        print(f"[Error cleaning {file_path}] {e}")
        return None

# =============================
# Implemented Metric Functions
# =============================
def segmental_snr_db(y: np.ndarray) -> float:
    """Compute segmental SNR in dB."""
    try:
        frame_length = int(0.025 * TARGET_SR)
        hop_length = int(0.010 * TARGET_SR)
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        noise_frames = energy < np.percentile(energy, 25)
        if np.sum(noise_frames) == 0:
            return 0.0
        noise_power = np.mean(frames[:, noise_frames]**2)
        signal_power = np.mean(frames**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr
    except Exception:
        return 0.0

def spectral_flatness_mean(y: np.ndarray, sr: int) -> float:
    """Compute mean spectral flatness."""
    try:
        S = np.abs(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)
        return np.mean(flatness)
    except Exception:
        return 0.0

def high_freq_energy_ratio(y: np.ndarray, sr: int) -> float:
    """Compute high-frequency energy ratio."""
    try:
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        high_mask = freqs > 4000
        total_energy = np.sum(np.abs(fft)**2)
        high_energy = np.sum(np.abs(fft[high_mask])**2)
        return high_energy / (total_energy + 1e-10)
    except Exception:
        return 0.0

def detect_breath_segments(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Breath detection with consistent frame parameters."""
    try:
        frame_length = int(0.050 * sr)
        hop_length = int(0.025 * sr)
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        min_len = min(len(energy), len(flatness))
        energy = energy[:min_len]
        flatness = flatness[:min_len]
        breath_mask = (energy < np.mean(energy) * 0.3) & (flatness > 0.8)
        segments = []
        start = None
        for i, mask in enumerate(breath_mask):
            t = i * (hop_length / sr)
            if mask and start is None:
                start = t
            elif not mask and start is not None:
                segments.append((start, t))
                start = None
        if start is not None:
            segments.append((start, len(y) / sr))
        return [(max(0, s), min(len(y)/sr, e)) for s, e in segments if e - s > 0.1]
    except Exception as e:
        print(f"[Error in breath detection] {e}")
        return []

def detect_plosives(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Plosive detection."""
    try:
        hop_length = int(0.010 * sr)
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])
        high_mask = freqs > 2000
        hf_frames = np.sum(S[high_mask, :], axis=0)
        energy_frames = np.sum(S, axis=0)
        hf_ratio = hf_frames / (energy_frames + 1e-10)
        plos_mask = hf_ratio > np.percentile(hf_ratio, 90)
        segments = []
        start = None
        for i, mask in enumerate(plos_mask):
            t = i * (hop_length / sr)
            if mask and start is None:
                start = t
            elif not mask and start is not None:
                if t - start < 0.05:
                    segments.append((start, t))
                start = None
        if start is not None and (len(y)/sr - start) < 0.05:
            segments.append((start, len(y)/sr))
        return segments
    except Exception:
        return []

def detect_prolongations(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Prolongation detection."""
    try:
        hop_length = int(0.025 * sr)
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        flux = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr)
        low_flux_mask = flux < np.percentile(flux, 20)
        segments = []
        start = None
        for i, mask in enumerate(low_flux_mask):
            t = i * (hop_length / sr)
            if mask and start is None:
                start = t
            elif not mask and start is not None:
                if t - start > 0.2:
                    segments.append((start, t))
                start = None
        if start is not None and (len(y)/sr - start) > 0.2:
            segments.append((start, len(y)/sr))
        return segments
    except Exception:
        return []

def transcribe_with_whisper(file_path: str) -> str:
    """Transcribe using Whisper if available."""
    if not WHISPER_AVAILABLE:
        return ""
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"].strip()
    except Exception as e:
        print(f"[Whisper error {file_path}] {e}")
        return ""

def repetition_stats_from_text(text: str) -> Dict[str, float]:
    """Compute repetition stats."""
    if not text:
        return {"repetition_count": 0, "repetition_ratio": 0.0, "repeated_words": ""}
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:
        return {"repetition_count": 0, "repetition_ratio": 0.0, "repeated_words": ""}
    word_counts = Counter(words)
    repeats = sum(count - 1 for count in word_counts.values() if count > 1)
    ratio = repeats / len(words)
    repeated = [word for word, count in word_counts.items() if count > 1]
    return {
        "repetition_count": repeats,
        "repetition_ratio": ratio,
        "repeated_words": ",".join(repeated[:5])
    }

# =============================
# Feature Extraction
# =============================
def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC, deltas, and chroma features."""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_stats = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1),
        ])
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stats = np.hstack([
            np.mean(chroma, axis=1), np.std(chroma, axis=1)
        ])
        return np.hstack([mfcc_stats, chroma_stats]).astype(np.float32)
    except Exception:
        # Fallback to zero features for failed extraction
        return np.zeros(76, dtype=np.float32)

def extract_text_features(text: str) -> np.ndarray:
    """Extract text-based features."""
    if not text:
        return np.zeros(5, dtype=np.float32)
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    unique_ratio = len(set(words)) / word_count if word_count > 0 else 0
    rep_stats = repetition_stats_from_text(text)
    return np.array([
        len(text),
        word_count,
        rep_stats["repetition_count"],
        rep_stats["repetition_ratio"],
        unique_ratio
    ], dtype=np.float32)

def extract_features(y: np.ndarray, sr: int, transcript: str) -> np.ndarray:
    """Combine audio and text features."""
    audio_feats = extract_audio_features(y, sr)
    text_feats = extract_text_features(transcript)
    return np.hstack([audio_feats, text_feats]).astype(np.float32)

# =============================
# Plotting Functions
# =============================
def plot_roc(y_true, y_score_dict, n_classes, save_path):
    """Plot multi-class ROC curves."""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(y_score_dict)))
    for (name, y_score), color in zip(y_score_dict.items(), colors):
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} Class {i} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================
# Main
# =============================
def main():
    print("=== Improved Dysfluency Classification & Analysis Pipeline (No Format Conversion) ===")

    # 1) Gather files - no conversion
    files = list_audio_files(DATA_DIR)
    if not files:
        print(f"No audio found under {DATA_DIR}. Please check directory and file formats.")
        return

    # 2) Preprocessing - load and clean only what librosa can handle
    per_file_rows = []
    clean_paths = []
    transcripts = []
    labels = []
    processed_count = 0

    for f in tqdm(files, desc="Preprocessing"):
        label = os.path.basename(os.path.dirname(f))
        y_raw, sr_raw = load_audio_any(f, sr=TARGET_SR)
        if y_raw is None:
            continue

        processed_count += 1

        # Pre-cleaning metrics
        snr_before = segmental_snr_db(y_raw)
        flat_before = spectral_flatness_mean(y_raw, TARGET_SR)
        hf_before = high_freq_energy_ratio(y_raw, TARGET_SR)

        # Clean
        out = clean_and_save_audio(f)
        if out is None:
            continue
        clean_paths.append(out)
        labels.append(label)

        # Post-cleaning metrics
        y_clean, _ = load_audio_any(out, sr=TARGET_SR)
        if y_clean is None:
            continue
        snr_after = segmental_snr_db(y_clean)
        flat_after = spectral_flatness_mean(y_clean, TARGET_SR)
        hf_after = high_freq_energy_ratio(y_clean, TARGET_SR)

        # Event detection
        breaths = detect_breath_segments(y_clean, TARGET_SR)
        plos = detect_plosives(y_clean, TARGET_SR)
        prols = detect_prolongations(y_clean, TARGET_SR)

        # Transcription
        text = transcribe_with_whisper(out)
        transcripts.append(text)
        rep_stats = repetition_stats_from_text(text)

        per_file_rows.append({
            "file": os.path.basename(f),
            "label": label,
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "spectral_flatness_before": flat_before,
            "spectral_flatness_after": flat_after,
            "hf_energy_ratio_before": hf_before,
            "hf_energy_ratio_after": hf_after,
            "breath_count": len(breaths),
            "breath_total_sec": sum(max(0.0, e - s) for s, e in breaths),
            "plosive_count": len(plos),
            "prolong_count": len(prols),
            "prolong_total_sec": sum(max(0.0, e - s) for s, e in prols),
            "transcript": text,
            "transcript_length": len(text),
            "word_count": len(re.findall(r'\b\w+\b', text.lower())),
            **rep_stats,
        })

    print(f"Processed {processed_count} files successfully.")

    if not per_file_rows:
        print("No valid audio files processed. Librosa may not support your formats without ffmpeg. Consider installing ffmpeg for MP3 support.")
        return

    # Save per-file analysis
    analysis_df = pd.DataFrame(per_file_rows)
    analysis_csv = os.path.join(OUTPUT_DIR, "per_file_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    print(f"Saved per-file analysis to {analysis_csv}")
    print("\nPer-File Analysis Summary:")
    print(analysis_df.describe())

    # 3) Feature Extraction
    X, y_all = [], []
    for p, lbl, text in tqdm(zip(clean_paths, labels, transcripts), total=len(clean_paths), desc="Feature Extraction"):
        y, sr = load_audio_any(p, sr=TARGET_SR)
        if y is None:
            continue
        cache_path = os.path.join(CACHE_DIR, os.path.basename(p) + "_features.npy")
        feats = cache_or_compute(cache_path, lambda: extract_features(y, sr, text))
        X.append(feats)
        y_all.append(lbl)

    if not X:
        print("No features extracted. Exiting.")
        return

    X = np.vstack(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    print(f"Extracted features shape: {X.shape}")
    print(f"Classes: {le.classes_}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # 4) Classification
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    predictions = {}
    probabilities = {}
    metrics_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds) * 100
        metrics_row = {
            f"{name} Accuracy (%)": round(acc, 2)
        }
        metrics_rows.append(metrics_row)
        predictions[name] = preds
        probabilities[name] = probs

    # Save metrics
    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print("\n=== Model Accuracy (in %) ===")
    print(df_metrics.to_string(index=False))

    # 5) ROC Curves
    y_score_dict = probabilities
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plot_roc(y_test, y_score_dict, n_classes=len(le.classes_), save_path=roc_path)

    # Feature importance for Random Forest
    if "Random Forest" in models:
        importances = models["Random Forest"].feature_importances_
        feat_names = (
            [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_std_{i}" for i in range(20)] +
            [f"delta_mean_{i}" for i in range(20)] + [f"delta_std_{i}" for i in range(20)] +
            [f"delta2_mean_{i}" for i in range(20)] + [f"delta2_std_{i}" for i in range(20)] +
            [f"chroma_mean_{i}" for i in range(12)] + [f"chroma_std_{i}" for i in range(12)] +
            ["transcript_length", "word_count", "repetition_count", "repetition_ratio", "unique_ratio"]
        )[:len(importances)]
        feat_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        feat_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
        print(f"\nTop 10 Important Features:")
        print(feat_df.head(10).to_string(index=False))

    print(f"\n=== Pipeline Complete. Processed {len(per_file_rows)} files. Check {OUTPUT_DIR} for results (CSVs, plots). ===")
    print("Note: For MP3 files, install ffmpeg to enable loading without conversion.")

if __name__ == "__main__":
    main()