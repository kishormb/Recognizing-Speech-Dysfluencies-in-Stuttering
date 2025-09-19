import os
from typing import List, Tuple, Dict
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import logging
from contextlib import redirect_stderr
from io import StringIO

# Set up logging to file
logging.basicConfig(
    filename=os.path.join("output_results", "pipeline_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try importing tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    st.warning("tqdm not installed. Progress bars disabled.")

# Try importing audio-related libraries
try:
    import librosa
    import noisereduce as nr
    import soundfile as sf
except ImportError:
    librosa = nr = sf = None
    st.error("Audio processing libraries (librosa, noisereduce, soundfile) not installed. Audio features disabled.")

# Try importing Whisper
try:
    import whisper
except ImportError:
    whisper = None

# Import sklearn and plotly
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
    from sklearn.metrics import accuracy_score, roc_curve, auc
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    import plotly.graph_objects as go
except ImportError:
    st.error("Required libraries (scikit-learn, plotly) not installed. Classification and plotting disabled.")
    exit(1)

from collections import Counter

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
MAX_DURATION = 30.0  # Limit audio to 30 seconds for faster processing

# Load Whisper model if available
try:
    if whisper:
        whisper_model = whisper.load_model("base")
        WHISPER_AVAILABLE = True
    else:
        WHISPER_AVAILABLE = False
        st.warning("Whisper not installed. Transcriptions will be empty.")
except (ImportError, Exception) as e:
    whisper_model = None
    WHISPER_AVAILABLE = False
    st.warning("Whisper not installed or failed to load. Transcriptions will be empty.")
    logging.error(f"Whisper initialization failed: {e}")

# =============================
# Utility Functions
# =============================
def list_audio_files(root: str) -> List[str]:
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.normpath(os.path.join(r, f)))
    st.write(f"Found {len(files)} audio files in {root}")
    return files

def load_audio_any(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    if not librosa:
        return None, None
    path = os.path.normpath(path)
    stderr = StringIO()
    try:
        with redirect_stderr(stderr):
            y, s = librosa.load(path, sr=sr, mono=True, duration=MAX_DURATION)
        if len(y) < sr * 0.1:
            logging.error(f"Audio too short: {path}")
            return None, None
        return y, s
    except Exception as e:
        logging.error(f"Error loading {path}: {e}\nSTDERR: {stderr.getvalue()}")
        return None, None

def cache_or_compute(cache_path: str, func):
    cache_path = os.path.normpath(cache_path)
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data.item() if data.ndim == 0 else data
        except Exception as e:
            logging.error(f"Error loading cache {cache_path}: {e}")
    result = func()
    try:
        if isinstance(result, np.ndarray):
            np.save(cache_path, result)
        else:
            np.save(cache_path, np.array([result], dtype=object))
    except Exception as e:
        logging.error(f"Error saving cache {cache_path}: {e}")
    return result

# =============================
# Audio Cleaning + Metrics
# =============================
def clean_and_save_audio(file_path: str) -> str:
    if not (librosa and nr and sf):
        return None
    file_path = os.path.normpath(file_path)
    # Check if cleaned file already exists
    safe_name = re.sub(r'[^\w.]', '_', os.path.basename(file_path).rsplit('.', 1)[0]) + '.wav'
    out_path = os.path.normpath(os.path.join(CLEAR_DIR, safe_name))
    if os.path.exists(out_path):
        return out_path
    y, sr = load_audio_any(file_path, sr=TARGET_SR)
    if y is None:
        return None
    try:
        y_clean = nr.reduce_noise(y=y, sr=sr)
        y_clean = librosa.util.normalize(y_clean)
        sf.write(out_path, y_clean, sr)
        return out_path
    except Exception as e:
        logging.error(f"Error cleaning {file_path}: {e}")
        return None

def segmental_snr_db(y: np.ndarray) -> float:
    if not librosa:
        return 0.0
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
    if not librosa:
        return 0.0
    try:
        S = np.abs(librosa.stft(y))
        flatness = librosa.feature.spectral_flatness(S=S)
        return np.mean(flatness)
    except Exception:
        return 0.0

def high_freq_energy_ratio(y: np.ndarray, sr: int) -> float:
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
    if not librosa:
        return []
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
        logging.error(f"Error in breath detection: {e}")
        return []

def detect_plosives(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    if not librosa:
        return []
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
    if not librosa:
        return []
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
    if not WHISPER_AVAILABLE:
        return ""
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        logging.error(f"Whisper error: File not found: {file_path}")
        return ""
    try:
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = whisper_model.transcribe(file_path)
        return result["text"].strip()
    except Exception as e:
        logging.error(f"Whisper error for {file_path}: {e}\nSTDERR: {stderr.getvalue()}")
        return ""

def repetition_stats_from_text(text: str) -> Dict[str, float]:
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
    if not librosa:
        return np.zeros(76, dtype=np.float32)
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
    except Exception as e:
        logging.error(f"Error extracting audio features: {e}")
        return np.zeros(76, dtype=np.float32)

def extract_text_features(text: str) -> np.ndarray:
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
    audio_feats = extract_audio_features(y, sr)
    text_feats = extract_text_features(transcript)
    return np.hstack([audio_feats, text_feats]).astype(np.float32)

# =============================
# Plotting Functions
# =============================
def plot_roc(y_true, y_score_dict, n_classes, class_names):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fig = go.Figure()
    auc_scores = []

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',  # Random Forest
        '#d62728', '#9467bd', '#8c564b',  # MLP
        '#e377c2', '#7f7f7f', '#bcbd22'   # SVM
    ]

    for model_idx, (name, y_score) in enumerate(y_score_dict.items()):
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                trace_name = f"{name} - {class_names[i]} (AUC = {roc_auc:.2f})"
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=trace_name,
                    line=dict(color=colors[model_idx * n_classes + i], width=3, shape='spline'),
                    text=[f"Model: {name}<br>Class: {class_names[i]}<br>AUC: {roc_auc:.2f}<br>FPR: {x:.2f}<br>TPR: {y:.2f}" 
                          for x, y in zip(fpr, tpr)],
                    hoverinfo='text',
                    showlegend=True,
                    opacity=0.8
                ))
                auc_scores.append({
                    "model": name,
                    "class": class_names[i],
                    "auc": roc_auc
                })
            except Exception as e:
                logging.error(f"Error computing ROC for {name} - {class_names[i]}: {e}")

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Chance (AUC = 0.50)',
        opacity=1.0,
        showlegend=True
    ))

    fig.update_layout(
        title=dict(text="Multi-Class ROC Curves", x=0.5, xanchor='center', font=dict(size=20)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], gridcolor='rgba(200,200,200,0.3)', showgrid=True),
        yaxis=dict(range=[0, 1], gridcolor='rgba(200,200,200,0.3)', showgrid=True),
        hovermode='closest',
        legend=dict(x=1.02, y=0.98, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.9)', 
                    bordercolor='black', borderwidth=1, font=dict(size=12)),
        width=900,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig, pd.DataFrame(auc_scores)

# =============================
# Streamlit App
# =============================
def main():
    st.title("Dysfluency Classification & Analysis Pipeline")
    st.write("### Processing audio files for dysfluency detection")

    # Option to disable Whisper or noise reduction
    use_whisper = st.checkbox("Enable Whisper transcription (slower)", value=False)
    use_noise_reduction = st.checkbox("Enable noise reduction (slower)", value=True)

    # 1) Gather files
    files = list_audio_files(DATA_DIR)
    if not files:
        st.error(f"No audio found under {DATA_DIR}. Please check directory and file formats.")
        return

    # 2) Preprocessing
    per_file_rows = []
    clean_paths = []
    transcripts = []
    labels = []
    processed_count = 0
    skipped_files = 0

    for f in tqdm(files, desc="Preprocessing"):
        label = os.path.basename(os.path.dirname(f))
        y_raw, sr_raw = load_audio_any(f, sr=TARGET_SR)
        if y_raw is None:
            skipped_files += 1
            continue

        processed_count += 1

        snr_before = segmental_snr_db(y_raw)
        flat_before = spectral_flatness_mean(y_raw, TARGET_SR)
        hf_before = high_freq_energy_ratio(y_raw, TARGET_SR)

        out = clean_and_save_audio(f) if use_noise_reduction else f
        if out is None:
            skipped_files += 1
            continue
        clean_paths.append(out)
        labels.append(label)

        y_clean, _ = load_audio_any(out, sr=TARGET_SR)
        if y_clean is None:
            skipped_files += 1
            continue
        snr_after = segmental_snr_db(y_clean) if use_noise_reduction else snr_before
        flat_after = spectral_flatness_mean(y_clean, TARGET_SR) if use_noise_reduction else flat_before
        hf_after = high_freq_energy_ratio(y_clean, TARGET_SR) if use_noise_reduction else hf_before

        breaths = detect_breath_segments(y_clean, TARGET_SR)
        plos = detect_plosives(y_clean, TARGET_SR)
        prols = detect_prolongations(y_clean, TARGET_SR)

        text = transcribe_with_whisper(out) if use_whisper else ""
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

    st.write(f"Processed {processed_count} files successfully. Skipped {skipped_files} files due to errors.")

    if not per_file_rows:
        st.error("No valid audio files processed. Install ffmpeg for MP3 support or check file integrity.")
        return

    # Save and display per-file analysis
    analysis_df = pd.DataFrame(per_file_rows)
    analysis_csv = os.path.join(OUTPUT_DIR, "per_file_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    st.write("### Per-File Analysis")
    st.dataframe(analysis_df)
    st.download_button("Download Per-File Analysis", analysis_df.to_csv(index=False), file_name="per_file_analysis.csv")

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
        st.error("No features extracted. Exiting.")
        return

    X = np.vstack(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    st.write(f"Extracted features shape: {X.shape}")
    st.write(f"Classes: {le.classes_}")

    # Normalize and split
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
        try:
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
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            st.warning(f"Failed to train {name}. Skipping.")

    # Save and display metrics
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        df_metrics.to_csv(metrics_csv, index=False)
        st.write("### Model Accuracy (in %)")
        st.dataframe(df_metrics)
        st.download_button("Download Metrics Summary", df_metrics.to_csv(index=False), file_name="metrics_summary.csv")
    else:
        st.warning("No models trained successfully.")

    # 5) ROC Curves
    if probabilities:
        st.write("### Select Model for ROC Curves")
        model_choice = st.selectbox("Choose Model", list(models.keys()) + ["All"])
        y_score_dict = {k: v for k, v in probabilities.items() if model_choice == "All" or k == model_choice}
        roc_fig, auc_df = plot_roc(y_test, y_score_dict, n_classes=len(le.classes_), class_names=le.classes_)
        st.write("### ROC Curves")
        st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.warning("No probabilities available for ROC curves.")

    # 6) Feature Importance
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

        auc_rows = [
            {"feature": f"AUC_{row['model']}_{row['class']}", "importance": row['auc']}
            for _, row in auc_df.iterrows()
        ]
        feat_df = pd.concat([feat_df, pd.DataFrame(auc_rows)], ignore_index=True)

        feat_csv = os.path.join(OUTPUT_DIR, "feature_importances.csv")
        feat_df.to_csv(feat_csv, index=False)
        st.write("### Feature Importances (Top 10)")
        st.dataframe(feat_df.head(10))
        st.download_button("Download Feature Importances", feat_df.to_csv(index=False), file_name="feature_importances.csv")

    # Download error log
    error_log_path = os.path.join(OUTPUT_DIR, "pipeline_errors.log")
    if os.path.exists(error_log_path):
        with open(error_log_path, "r") as f:
            st.download_button("Download Error Log", f.read(), file_name="pipeline_errors.log")

    st.success(f"Pipeline Complete. Processed {len(per_file_rows)} files. Skipped {skipped_files} files. Check {OUTPUT_DIR} for CSVs.")

if __name__ == "__main__":
    main()