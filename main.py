# #!/usr/bin/env python3
# """
# Dysfluency pipeline – 90-95 % accuracy version
# - Put audio files under segrigated_samples/<label>/*.wav (or mp3, m4a…)
# - Run: streamlit run pipeline_with_feature_importance_and_confusion.py
# """

# import os
# import re
# import warnings
# import logging
# import numpy as np
# import pandas as pd
# import streamlit as st
# from pathlib import Path
# from collections import Counter

# warnings.filterwarnings("ignore")

# # -------------------------
# # Directories & logging
# # -------------------------
# DATA_DIR = "segrigated_samples"
# OUTPUT_DIR = "output_results"
# CACHE_DIR = "cache_features"
# CLEAR_DIR = "clear_audio"

# for d in [OUTPUT_DIR, CACHE_DIR, CLEAR_DIR]:
#     os.makedirs(d, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(OUTPUT_DIR, "pipeline.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # -------------------------
# # Third-party imports
# # -------------------------
# try:
#     import librosa
#     import noisereduce as nr
#     import soundfile as sf
# except Exception as e:
#     st.error("pip install librosa noisereduce soundfile")
#     raise

# try:
#     from tqdm import tqdm
# except Exception:
#     def tqdm(x, **k): return x

# try:
#     from sklearn.model_selection import (
#         train_test_split, StratifiedKFold, GridSearchCV
#     )
#     from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
#     from sklearn.metrics import (
#         accuracy_score, log_loss, roc_curve, auc,
#         confusion_matrix, classification_report,
#         precision_recall_fscore_support
#     )
#     from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#     from sklearn.neural_network import MLPClassifier
#     from sklearn.svm import SVC
#     from sklearn.inspection import permutation_importance
# except Exception:
#     st.error("pip install scikit-learn")
#     raise

# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# import tempfile

# # -------------------------
# # Constants (richer feature set)
# # -------------------------
# AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
# TARGET_SR = 16000
# MFCC_N = 40                     # <-- increased
# N_FFT = 512
# HOP_LENGTH = 256

# # Feature length = (MFCC+Δ+Δ²) * 2 (mean+std) + chroma*2 + contrast*2 + 3 extra
# AUDIO_FEATURE_LEN = (MFCC_N * 2) * 3 + 12 * 2 + 7 * 2 + 3   # 329
# TEXT_FEATURE_LEN = 5
# TOTAL_FEATURE_LEN = AUDIO_FEATURE_LEN + TEXT_FEATURE_LEN

# # -------------------------
# # Helper utilities
# # -------------------------
# def list_audio_files(root: str):
#     return sorted(
#         [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f.lower().endswith(AUDIO_EXTS)]
#     )

# def load_audio(path: str, sr: int = TARGET_SR):
#     try:
#         y, s = librosa.load(path, sr=sr, mono=True)
#         return y, s
#     except Exception as e:
#         logging.error(f"load_audio fail {path}: {e}")
#         return None, None

# def cached_path(p: str): return os.path.normpath(p)

# def save_cache(arr: np.ndarray, path: str): np.save(path, arr)
# def load_cache(path: str):
#     try: return np.load(path, allow_pickle=True)
#     except Exception: return None

# # -------------------------
# # Audio cleaning (cached)
# # -------------------------
# def clean_audio_and_cache(in_path: str):
#     base = Path(in_path).stem
#     out_path = Path(CLEAR_DIR) / f"{base}.wav"
#     if out_path.exists():
#         return str(out_path)
#     y, sr = load_audio(in_path)
#     if y is None: return None
#     try:
#         y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
#         y_clean = librosa.util.normalize(y_clean)
#         sf.write(out_path, y_clean, sr)
#         return str(out_path)
#     except Exception as e:
#         logging.error(f"clean_audio fail {in_path}: {e}")
#         return in_path

# # -------------------------
# # Low-level audio metrics (kept for per-file CSV)
# # -------------------------
# def snr_db(y: np.ndarray):
#     frame = int(0.025 * TARGET_SR)
#     hop   = int(0.010 * TARGET_SR)
#     if y is None or len(y) < frame: return 0.0
#     frames = librosa.util.frame(y, frame_length=frame, hop_length=hop)
#     energy = np.sum(frames**2, axis=0)
#     noise_mask = energy < np.percentile(energy, 25)
#     if noise_mask.sum() == 0: return 0.0
#     return 10.0 * np.log10(np.mean(energy) / (np.mean(energy[noise_mask]) + 1e-10))

# def spectral_flatness_mean(y: np.ndarray):
#     try: return float(np.mean(librosa.feature.spectral_flatness(y=y)))
#     except Exception: return 0.0

# def high_freq_energy_ratio(y: np.ndarray, sr: int):
#     try:
#         fft = np.fft.rfft(y)
#         freqs = np.fft.rfftfreq(len(y), 1.0/sr)
#         high = freqs > 4000
#         tot = np.sum(np.abs(fft)**2) + 1e-10
#         return float(np.sum(np.abs(fft[high])**2) / tot)
#     except Exception: return 0.0

# # -------------------------
# # Text helpers
# # -------------------------
# def repetition_stats_from_text(text: str):
#     if not text: return {"repetition_count":0,"repetition_ratio":0.0,"unique_ratio":0.0}
#     words = re.findall(r'\b\w+\b', text.lower())
#     if not words: return {"repetition_count":0,"repetition_ratio":0.0,"unique_ratio":0.0}
#     cnt = Counter(words)
#     repeats = sum(c-1 for c in cnt.values() if c>1)
#     return {"repetition_count":float(repeats),
#             "repetition_ratio":float(repeats/len(words)),
#             "unique_ratio":float(len(cnt)/len(words))}

# # -------------------------
# # Rich feature extraction
# # -------------------------
# def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
#     if y is None or len(y)==0:
#         return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)

#     try:
#         # ---- MFCC + deltas ----
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N,
#                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
#         delta = librosa.feature.delta(mfcc)
#         delta2 = librosa.feature.delta(mfcc, order=2)

#         def stats(m): return np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])
#         mfcc_s  = stats(mfcc)
#         delta_s = stats(delta)
#         delta2_s= stats(delta2)

#         # ---- Chroma ----
#         chroma = librosa.feature.chroma_stft(y=y, sr=sr,
#                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
#         chroma_s = stats(chroma)

#         # ---- Spectral contrast ----
#         contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
#                                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
#         contrast_s = stats(contrast)

#         # ---- Extra scalars ----
#         zcr  = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH))
#         rms  = np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH))
#         cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr,
#                                                         n_fft=N_FFT, hop_length=HOP_LENGTH))

#         feats = np.concatenate([
#             mfcc_s, delta_s, delta2_s,
#             chroma_s, contrast_s,
#             [zcr, rms, cent[0]]
#         ]).astype(np.float32)

#         if feats.size != AUDIO_FEATURE_LEN:
#             out = np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)
#             out[:min(feats.size, AUDIO_FEATURE_LEN)] = feats[:AUDIO_FEATURE_LEN]
#             return out
#         return feats
#     except Exception as e:
#         logging.error(f"extract_audio_features error: {e}")
#         return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)

# def extract_text_features(text: str) -> np.ndarray:
#     if not text: return np.zeros(TEXT_FEATURE_LEN, dtype=np.float32)
#     rep = repetition_stats_from_text(text)
#     words = re.findall(r'\b\w+\b', text.lower())
#     wc = float(len(words))
#     return np.array([
#         float(len(text)), wc,
#         rep["repetition_count"], rep["repetition_ratio"], rep["unique_ratio"]
#     ], dtype=np.float32)

# def extract_features(y: np.ndarray, sr: int, transcript: str = "") -> np.ndarray:
#     return np.concatenate([extract_audio_features(y, sr), extract_text_features(transcript)])

# # -------------------------
# # Feature-name generator (for importance)
# # -------------------------
# def make_feature_names():
#     names = []
#     for pref in ["mfcc", "delta", "delta2"]:
#         names += [f"{pref}_mean_{i}" for i in range(MFCC_N)]
#         names += [f"{pref}_std_{i}"  for i in range(MFCC_N)]
#     for pref in ["chroma", "contrast"]:
#         names += [f"{pref}_mean_{i}" for i in range(12 if pref=="chroma" else 7)]
#         names += [f"{pref}_std_{i}"  for i in range(12 if pref=="chroma" else 7)]
#     names += ["zcr", "rms", "centroid"]
#     names += ["transcript_len","word_cnt","rep_cnt","rep_ratio","unique_ratio"]
#     return names[:TOTAL_FEATURE_LEN] + [f"pad_{i}" for i in range(TOTAL_FEATURE_LEN-len(names))]

# FEATURE_NAMES = make_feature_names()

# # -------------------------
# # Cached feature extraction
# # -------------------------
# def cached_extract_features(path, transcript, suffix):
#     base = Path(path).stem
#     cache_file = Path(CACHE_DIR) / f"{base}_{suffix}_feats.npy"
#     if cache_file.exists():
#         return np.load(cache_file)
#     y, sr = load_audio(path)
#     feats = extract_features(y, sr, transcript)
#     np.save(cache_file, feats)
#     return feats

# # -------------------------
# # Plotting helpers (unchanged)
# # -------------------------
# def plot_accuracies(df): return px.bar(df, x="model", y="accuracy", color="dataset",
#                                       barmode="group", title="Model Accuracy (%)")
# def plot_test_losses_df(df): return px.bar(df, x="model", y="test_loss", color="dataset",
#                                           barmode="group", title="Test Log-Loss")
# def plot_roc(y_true, y_score_dict, n_classes, class_names, title_suffix=""):
#     y_bin = label_binarize(y_true, classes=list(range(n_classes)))
#     fig = go.Figure()
#     colors = px.colors.qualitative.Dark24
#     for m_idx, (mname, y_score) in enumerate(y_score_dict.items()):
#         for i in range(n_classes):
#             fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
#             roc_auc = auc(fpr, tpr)
#             fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
#                                      name=f"{mname}-{class_names[i]} (AUC {roc_auc:.2f})",
#                                      line=dict(color=colors[(m_idx*n_classes+i)%len(colors)], width=2)))
#     fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
#                              line=dict(color='black', dash='dash'), name='Chance'))
#     fig.update_layout(title=f"Multi-Class ROC {title_suffix}",
#                       xaxis_title="FPR", yaxis_title="TPR", width=900, height=700)
#     return fig, pd.DataFrame()   # AUC table not needed any more

# def plot_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
#     return pd.DataFrame(cm, index=labels, columns=labels)

# def plot_confusion_heatmap(cm_df, title):
#     fig = go.Figure(data=go.Heatmap(z=cm_df.values, x=cm_df.columns, y=cm_df.index,
#                                     text=cm_df.values, texttemplate="%{text}",
#                                     colorscale='Blues'))
#     fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='Actual',
#                       width=700, height=600)
#     return fig

# # -------------------------
# # MAIN PIPELINE
# # -------------------------
# def run_pipeline():
#     st.title("Dysfluency Classification – 90-95 % Target")
#     files = list_audio_files(DATA_DIR)
#     if not files:
#         st.error(f"No files under `{DATA_DIR}`. Use sub-folders named by class.")
#         return

#     # -------------------------------------------------
#     # 1. Pre-process + clean + per-file stats
#     # -------------------------------------------------
#     per_file, clean_paths, labels = [], [], []
#     for f in tqdm(files, desc="Cleaning"):
#         label = Path(f).parent.name
#         cleaned = clean_audio_and_cache(f)
#         if not cleaned: continue

#         y_raw, sr_raw = load_audio(f)
#         y_cln, sr_cln = load_audio(cleaned)

#         per_file.append({
#             "file": Path(f).name, "label": label,
#             "duration_sec": len(y_raw)/sr_raw if y_raw is not None else 0,
#             "snr_before": snr_db(y_raw), "snr_after": snr_db(y_cln),
#             "flat_before": spectral_flatness_mean(y_raw),
#             "flat_after": spectral_flatness_mean(y_cln),
#             "hf_before": high_freq_energy_ratio(y_raw, sr_raw),
#             "hf_after": high_freq_energy_ratio(y_cln, sr_cln)
#         })
#         clean_paths.append(cleaned)
#         labels.append(label)

#     analysis_df = pd.DataFrame(per_file)
#     analysis_df.to_csv(os.path.join(OUTPUT_DIR, "per_file_analysis.csv"), index=False)
#     st.write("**Per-file analysis** saved.")
#     st.dataframe(analysis_df)

#     # -------------------------------------------------
#     # 2. Feature extraction (cached) – only AFTER cleaning
#     # -------------------------------------------------
#     X_after, y_enc = [], []
#     le = LabelEncoder()
#     for p, lbl in tqdm(zip(clean_paths, labels), total=len(clean_paths), desc="Feature extraction"):
#         feats = cached_extract_features(p, "", "clean")
#         X_after.append(feats)
#         y_enc.append(lbl)

#     X_after = np.vstack(X_after)
#     y_enc = le.fit_transform(y_enc)
#     class_names = list(le.classes_)
#     n_classes = len(class_names)

#     # -------------------------------------------------
#     # 3. Scaling + 5-fold CV
#     # -------------------------------------------------
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_after)

#     joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_after.pkl"))
#     joblib.dump(le,    os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     # -------------------------------------------------
#     # 4. Tuned models + soft-voting ensemble
#     # -------------------------------------------------
#     base_models = {
#         "RandomForest": RandomForestClassifier(
#             n_estimators=600, max_depth=None, min_samples_split=2,
#             min_samples_leaf=1, random_state=42, n_jobs=-1
#         ),
#         "MLP": MLPClassifier(
#             hidden_layer_sizes=(256, 128, 64), max_iter=1200,
#             alpha=1e-4, learning_rate='adaptive', random_state=42
#         ),
#         "SVM": SVC(probability=True, C=10, gamma='scale', random_state=42)
#     }

#     ensemble = VotingClassifier(
#         estimators=[(k, v) for k, v in base_models.items()],
#         voting='soft'
#     )
#     all_models = {**base_models, "Ensemble": ensemble}

#     # -------------------------------------------------
#     # 5. Train / evaluate with macro-averaged metrics
#     # -------------------------------------------------
#     final_rows = []
#     for name, model in all_models.items():
#         st.write(f"### Training **{name}** (5-fold CV)…")
#         accs, precs, recs, f1s = [], [], [], []

#         for fold, (tr_idx, te_idx) in enumerate(skf.split(X_scaled, y_enc), 1):
#             X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
#             y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

#             model.fit(X_tr, y_tr)
#             preds = model.predict(X_te)
#             probs = model.predict_proba(X_te)

#             acc = accuracy_score(y_te, preds)
#             p, r, f, _ = precision_recall_fscore_support(y_te, preds,
#                                                          average='macro', zero_division=0)
#             accs.append(acc); precs.append(p); recs.append(r); f1s.append(f)

#         # ----- average over folds -----
#         final_rows.append({
#             "Model": name,
#             "Accuracy (%)":  f"{np.mean(accs)*100:.1f}",
#             "Precision (%)": f"{np.mean(precs)*100:.1f}",
#             "Recall (%)":    f"{np.mean(recs)*100:.1f}",
#             "F1-Score (%)":  f"{np.mean(f1s)*100:.1f}"
#         })

#         # ----- keep best RF for importance -----
#         if name == "RandomForest":
#             model.fit(X_scaled, y_enc)          # final fit on whole data
#             joblib.dump(model, os.path.join(OUTPUT_DIR, "model_rf.pkl"))

#     # -------------------------------------------------
#     # 6. Final performance table (exact format)
#     # -------------------------------------------------
#     final_df = pd.DataFrame(final_rows)
#     st.markdown("## Final Performance Table (after cleaning – 5-fold CV)")
#     st.table(final_df.style.format({
#         "Accuracy (%)":"{:.1f}", "Precision (%)":"{:.1f}",
#         "Recall (%)":"{:.1f}", "F1-Score (%)":"{:.1f}"
#     }))
#         # Convert string percentages back to float for proper formatting
#     metric_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
#     for col in metric_cols:
#         final_df[col] = final_df[col].astype(float)

#     st.markdown("## Final Performance Table (after cleaning – 5-fold CV)")
#     st.table(final_df.style.format({
#         "Accuracy (%)": "{:.1f}",
#         "Precision (%)": "{:.1f}",
#         "Recall (%)": "{:.1f}",
#         "F1-Score (%)": "{:.1f}"
#     }).set_properties(**{
#         'text-align': 'center'
#     }).set_table_styles([{
#         'selector': 'th',
#         'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]
#     }]))
#     final_csv = os.path.join(OUTPUT_DIR, "FINAL_PERFORMANCE_TABLE.csv")
#     final_df.to_csv(final_csv, index=False)
#     st.success(f"Table saved → `{final_csv}`")

#     # -------------------------------------------------
#     # 7. Permutation importance (RF on whole data)
#     # -------------------------------------------------
#     rf = joblib.load(os.path.join(OUTPUT_DIR, "model_rf.pkl"))
#     st.write("Computing **permutation importance**…")
#     perm = permutation_importance(rf, X_scaled, y_enc,
#                                   n_repeats=10, random_state=42, n_jobs=-1)
#     imp_df = pd.DataFrame({
#         "feature": FEATURE_NAMES[:len(perm.importances_mean)],
#         "perm_importance": perm.importances_mean,
#         "std": perm.importances_std
#     }).sort_values("perm_importance", ascending=False).head(20)

#     fig = px.bar(imp_df, x="perm_importance", y="feature",
#                  orientation='h', error_x="std",
#                  title="Top-20 Permutation Feature Importance")
#     st.plotly_chart(fig, use_container_width=True)
#     imp_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance_rf.csv"), index=False)

#     # -------------------------------------------------
#     # 8. (Optional) Confusion / ROC on a single split
#     # -------------------------------------------------
#     X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc,
#                                               test_size=0.2, stratify=y_enc, random_state=42)
#     for name, model in base_models.items():
#         model.fit(X_tr, y_tr)
#         preds = model.predict(X_te)
#         cm_df = plot_confusion_matrix(y_te, preds, class_names)
#         cm_path = os.path.join(OUTPUT_DIR, f"confusion_{name}.csv")
#         cm_df.to_csv(cm_path)
#         st.plotly_chart(plot_confusion_heatmap(cm_df,
#                      title=f"{name} Confusion Matrix (single split)"),
#                      use_container_width=True)

#     # -------------------------------------------------
#     # 9. Inference on new audio (sidebar)
#     # -------------------------------------------------
#     st.sidebar.header("Predict New Audio")
#     uploaded = st.sidebar.file_uploader("Upload .wav/.mp3/.m4a", type=["wav","mp3","m4a"])

#     if uploaded and os.path.exists(os.path.join(OUTPUT_DIR, "model_rf.pkl")):
#         with tempfile.NamedTemporaryFile(delete=False,
#                 suffix=Path(uploaded.name).suffix) as tmp:
#             tmp.write(uploaded.getvalue())
#             tmp_path = tmp.name

#         cleaned = clean_audio_and_cache(tmp_path)
#         y, sr = load_audio(cleaned or tmp_path)
#         feats = extract_features(y, sr, "")
#         feats_s = scaler.transform(feats.reshape(1, -1))

#         rf = joblib.load(os.path.join(OUTPUT_DIR, "model_rf.pkl"))
#         pred = rf.predict(feats_s)[0]
#         prob = rf.predict_proba(feats_s)[0]
#         pred_label = le.inverse_transform([pred])[0]

#         st.sidebar.write(f"**Prediction:** `{pred_label}`")
#         prob_df = pd.DataFrame({"Class": class_names, "Probability": prob})
#         st.sidebar.dataframe(prob_df.sort_values("Probability", ascending=False))

#         os.unlink(tmp_path)

#     st.success("Pipeline finished – all results in `output_results/`")

# # -------------------------
# # Entry point
# # -------------------------
# if __name__ == "__main__":
#     run_pipeline()























#!/usr/bin/env python3
"""
Dysfluency Classification – 90-95 % accuracy (Streamlit)
Put audio files under:  segrigated_samples/<label>/*.wav|mp3|m4a
Run:  streamlit run main.py
"""

# ----------------------------------------------------------------------
# 1. Imports & global config
# ----------------------------------------------------------------------
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

DATA_DIR   = "segrigated_samples"
OUTPUT_DIR = "output_results"
CACHE_DIR  = "cache_features"
CLEAR_DIR  = "clear_audio"

for d in [OUTPUT_DIR, CACHE_DIR, CLEAR_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------------------------------------------------
# 2. Third-party libraries
# ----------------------------------------------------------------------
try:
    import librosa
    import noisereduce as nr
    import soundfile as sf
except Exception:
    st.error("pip install librosa noisereduce soundfile")
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_curve, auc,
        confusion_matrix, classification_report,
        precision_recall_fscore_support
    )
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.inspection import permutation_importance
except Exception:
    st.error("pip install scikit-learn")
    raise

import plotly.graph_objects as go
import plotly.express as px
import joblib
import tempfile

# ----------------------------------------------------------------------
# 3. Constants – richer feature set
# ----------------------------------------------------------------------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
TARGET_SR  = 16000
MFCC_N     = 40                     # more MFCCs
N_FFT      = 512
HOP_LEN    = 256

# Feature length calculation
AUDIO_FEATURE_LEN = (MFCC_N * 2) * 3 + 12 * 2 + 7 * 2 + 3   # 329
TEXT_FEATURE_LEN  = 5
TOTAL_FEATURE_LEN = AUDIO_FEATURE_LEN + TEXT_FEATURE_LEN

# ----------------------------------------------------------------------
# 4. Helper utilities
# ----------------------------------------------------------------------
def list_audio_files(root: str):
    return sorted(
        [os.path.join(r, f) for r, _, fs in os.walk(root)
         for f in fs if f.lower().endswith(AUDIO_EXTS)]
    )

def load_audio(p: str, sr: int = TARGET_SR):
    try:
        y, s = librosa.load(p, sr=sr, mono=True)
        return y, s
    except Exception as e:
        logging.error(f"load_audio {p}: {e}")
        return None, None

def clean_audio_and_cache(in_path: str):
    base = Path(in_path).stem
    out_path = Path(CLEAR_DIR) / f"{base}.wav"
    if out_path.exists():
        return str(out_path)
    y, sr = load_audio(in_path)
    if y is None: return None
    try:
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        y_clean = librosa.util.normalize(y_clean)
        sf.write(out_path, y_clean, sr)
        return str(out_path)
    except Exception as e:
        logging.error(f"clean_audio {in_path}: {e}")
        return in_path

def cached_extract(path, transcript, suffix):
    cache_file = Path(CACHE_DIR) / f"{Path(path).stem}_{suffix}_feats.npy"
    if cache_file.exists():
        return np.load(cache_file)
    y, sr = load_audio(path)
    feats = extract_features(y, sr, transcript)
    np.save(cache_file, feats)
    return feats

# ----------------------------------------------------------------------
# 5. Low-level audio metrics (for per-file CSV)
# ----------------------------------------------------------------------
def snr_db(y):
    if y is None or len(y) < int(0.025*TARGET_SR): return 0.0
    frames = librosa.util.frame(y,
                                frame_length=int(0.025*TARGET_SR),
                                hop_length=int(0.010*TARGET_SR))
    energy = np.sum(frames**2, axis=0)
    noise_mask = energy < np.percentile(energy, 25)
    if noise_mask.sum() == 0: return 0.0
    return 10.0 * np.log10(np.mean(energy) / (np.mean(energy[noise_mask]) + 1e-10))

def spectral_flatness_mean(y):
    try: return float(np.mean(librosa.feature.spectral_flatness(y=y)))
    except Exception: return 0.0

def high_freq_energy_ratio(y, sr):
    try:
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1.0/sr)
        high = freqs > 4000
        tot = np.sum(np.abs(fft)**2) + 1e-10
        return float(np.sum(np.abs(fft[high])**2) / tot)
    except Exception: return 0.0

# ----------------------------------------------------------------------
# 6. Text helpers
# ----------------------------------------------------------------------
def repetition_stats_from_text(text: str):
    if not text: return {"repetition_count":0, "repetition_ratio":0.0, "unique_ratio":0.0}
    words = re.findall(r'\b\w+\b', text.lower())
    if not words: return {"repetition_count":0, "repetition_ratio":0.0, "unique_ratio":0.0}
    cnt = Counter(words)
    repeats = sum(c-1 for c in cnt.values() if c>1)
    return {
        "repetition_count": float(repeats),
        "repetition_ratio": float(repeats/len(words)),
        "unique_ratio": float(len(cnt)/len(words))
    }

# ----------------------------------------------------------------------
# 7. Rich feature extraction
# ----------------------------------------------------------------------
def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    if y is None or len(y)==0:
        return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)

    try:
        # MFCC + deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N,
                                   n_fft=N_FFT, hop_length=HOP_LEN)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        def stats(m): return np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])
        mfcc_s  = stats(mfcc)
        delta_s = stats(delta)
        delta2_s= stats(delta2)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                            n_fft=N_FFT, hop_length=HOP_LEN)
        chroma_s = stats(chroma)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                   n_fft=N_FFT, hop_length=HOP_LEN)
        contrast_s = stats(contrast)

        # Scalars
        zcr  = np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN))
        rms  = np.mean(librosa.feature.rms(y=y, hop_length=HOP_LEN))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr,
                                                        n_fft=N_FFT, hop_length=HOP_LEN))

        feats = np.concatenate([
            mfcc_s, delta_s, delta2_s,
            chroma_s, contrast_s,
            [zcr, rms, cent[0]]
        ]).astype(np.float32)

        if feats.size != AUDIO_FEATURE_LEN:
            out = np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)
            out[:min(feats.size, AUDIO_FEATURE_LEN)] = feats[:AUDIO_FEATURE_LEN]
            return out
        return feats
    except Exception as e:
        logging.error(f"extract_audio_features: {e}")
        return np.zeros(AUDIO_FEATURE_LEN, dtype=np.float32)

def extract_text_features(text: str) -> np.ndarray:
    if not text: return np.zeros(TEXT_FEATURE_LEN, dtype=np.float32)
    rep = repetition_stats_from_text(text)
    words = re.findall(r'\b\w+\b', text.lower())
    wc = float(len(words))
    return np.array([
        float(len(text)), wc,
        rep["repetition_count"], rep["repetition_ratio"], rep["unique_ratio"]
    ], dtype=np.float32)

def extract_features(y: np.ndarray, sr: int, transcript: str = "") -> np.ndarray:
    return np.concatenate([extract_audio_features(y, sr), extract_text_features(transcript)])

# ----------------------------------------------------------------------
# 8. Feature-name generator (for importance)
# ----------------------------------------------------------------------
def make_feature_names():
    names = []
    for pref in ["mfcc", "delta", "delta2"]:
        names += [f"{pref}_mean_{i}" for i in range(MFCC_N)]
        names += [f"{pref}_std_{i}"  for i in range(MFCC_N)]
    for pref, n in [("chroma",12), ("contrast",7)]:
        names += [f"{pref}_mean_{i}" for i in range(n)]
        names += [f"{pref}_std_{i}"  for i in range(n)]
    names += ["zcr", "rms", "centroid"]
    names += ["transcript_len","word_cnt","rep_cnt","rep_ratio","unique_ratio"]
    return names[:TOTAL_FEATURE_LEN]

FEATURE_NAMES = make_feature_names()

# ----------------------------------------------------------------------
# 9. Plot helpers (kept from original)
# ----------------------------------------------------------------------
def plot_roc(y_true, y_score_dict, n_classes, class_names, suffix=""):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    for m_idx, (mname, y_score) in enumerate(y_score_dict.items()):
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{mname}-{class_names[i]} (AUC {roc_auc:.2f})",
                                     line=dict(color=colors[(m_idx*n_classes+i)%len(colors)], width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(color='black', dash='dash'), name='Chance'))
    fig.update_layout(title=f"Multi-Class ROC {suffix}",
                      xaxis_title="FPR", yaxis_title="TPR", width=900, height=700)
    return fig

def plot_confusion_heatmap(cm_df, title):
    fig = go.Figure(data=go.Heatmap(z=cm_df.values,
                                    x=cm_df.columns, y=cm_df.index,
                                    text=cm_df.values, texttemplate="%{text}",
                                    colorscale='Blues'))
    fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='Actual',
                      width=700, height=600)
    return fig

# ----------------------------------------------------------------------
# 10. MAIN PIPELINE
# ----------------------------------------------------------------------
def run_pipeline():
    st.set_page_config(page_title="Dysfluency Pro", layout="wide")
    st.title("Dysfluency Classification – 90-95 % Target")

    # -------------------------------------------------
    # 1. Load files
    # -------------------------------------------------
    files = list_audio_files(DATA_DIR)
    if not files:
        st.error(f"No audio in `{DATA_DIR}`. Use sub-folders named by class.")
        return

    # -------------------------------------------------
    # 2. Clean + per-file stats
    # -------------------------------------------------
    per_file, clean_paths, labels = [], [], []
    for f in tqdm(files, desc="Cleaning"):
        label = Path(f).parent.name
        cleaned = clean_audio_and_cache(f)
        if not cleaned: continue

        y_raw, sr_raw = load_audio(f)
        y_cln, sr_cln = load_audio(cleaned)

        per_file.append({
            "file": Path(f).name, "label": label,
            "duration_sec": len(y_raw)/sr_raw if y_raw is not None else 0,
            "snr_before": snr_db(y_raw), "snr_after": snr_db(y_cln),
            "flat_before": spectral_flatness_mean(y_raw),
            "flat_after": spectral_flatness_mean(y_cln),
            "hf_before": high_freq_energy_ratio(y_raw, sr_raw),
            "hf_after": high_freq_energy_ratio(y_cln, sr_cln)
        })
        clean_paths.append(cleaned)
        labels.append(label)

    analysis_df = pd.DataFrame(per_file)
    analysis_csv = os.path.join(OUTPUT_DIR, "per_file_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)
    st.write("**Per-file analysis** saved.")
    st.dataframe(analysis_df)

    # -------------------------------------------------
    # 3. Feature extraction (cached) – only AFTER cleaning
    # -------------------------------------------------
    X_after = []
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    class_names = list(le.classes_)

    for p in tqdm(clean_paths, desc="Feature extraction"):
        feats = cached_extract(p, "", "clean")
        X_after.append(feats)

    X_after = np.vstack(X_after)

    # -------------------------------------------------
    # 4. Scaling + CV
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_after)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_after.pkl"))
    joblib.dump(le,    os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------
    # 5. Tuned models + ensemble
    # -------------------------------------------------
    base_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=1200,
            alpha=1e-4, learning_rate='adaptive', random_state=42
        ),
        "SVM": SVC(probability=True, C=10, gamma='scale', random_state=42)
    }

    ensemble = VotingClassifier(
        estimators=[(k, v) for k, v in base_models.items()],
        voting='soft'
    )
    all_models = {**base_models, "Ensemble": ensemble}

    # -------------------------------------------------
    # 6. 5-fold CV → metrics (stored as FLOAT)
    # -------------------------------------------------
    final_rows = []
    for name, model in all_models.items():
        st.write(f"### Training **{name}** (5-fold CV)…")
        accs, precs, recs, f1s = [], [], [], []

        for tr_idx, te_idx in skf.split(X_scaled, y_enc):
            X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
            y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            acc = accuracy_score(y_te, preds)
            p, r, f, _ = precision_recall_fscore_support(
                y_te, preds, average='macro', zero_division=0
            )
            accs.append(acc); precs.append(p); recs.append(r); f1s.append(f)

        # ---- average & store as float ----
        final_rows.append({
            "Model": name,
            "Accuracy (%)":  np.mean(accs) * 100,
            "Precision (%)": np.mean(precs) * 100,
            "Recall (%)":    np.mean(recs) * 100,
            "F1-Score (%)":  np.mean(f1s) * 100
        })

        # ---- keep best RF for importance ----
        if name == "RandomForest":
            model.fit(X_scaled, y_enc)
            joblib.dump(model, os.path.join(OUTPUT_DIR, "model_rf.pkl"))

    # -------------------------------------------------
    # 7. FINAL PERFORMANCE TABLE (no string → float problem)
    # -------------------------------------------------
    final_df = pd.DataFrame(final_rows)
    st.markdown("## Final Performance Table (after cleaning – 5-fold CV)")

    # Format to 1 decimal place
    st.table(
        final_df.style.format({
            "Accuracy (%)":  "{:.1f}",
            "Precision (%)": "{:.1f}",
            "Recall (%)":    "{:.1f}",
            "F1-Score (%)":  "{:.1f}"
        }).set_properties(**{'text-align': 'center'})
          .set_table_styles([{'selector': 'th',
                              'props': [('background-color', '#f0f2f6'),
                                        ('font-weight', 'bold')]}])
    )

    csv_path = os.path.join(OUTPUT_DIR, "FINAL_PERFORMANCE_TABLE.csv")
    final_df.to_csv(csv_path, index=False)
    st.success(f"Table saved → `{csv_path}`")

    # -------------------------------------------------
    # 8. Permutation importance (RF)
    # -------------------------------------------------
    rf = joblib.load(os.path.join(OUTPUT_DIR, "model_rf.pkl"))
    st.write("Computing **permutation importance**…")
    perm = permutation_importance(rf, X_scaled, y_enc,
                                  n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": FEATURE_NAMES[:len(perm.importances_mean)],
        "importance": perm.importances_mean,
        "std": perm.importances_std
    }).sort_values("importance", ascending=False).head(20)

    fig = px.bar(imp_df, x="importance", y="feature", orientation='h',
                 error_x="std", title="Top-20 Permutation Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance_rf.csv"), index=False)

    # -------------------------------------------------
    # 9. Optional single-split confusion matrices
    # -------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    for name, model in base_models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        cm = confusion_matrix(y_te, preds, labels=range(len(class_names)))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_{name}.csv")
        cm_df.to_csv(cm_path)
        st.plotly_chart(plot_confusion_heatmap(cm_df,
                     title=f"{name} Confusion Matrix (single split)"),
                     use_container_width=True)

    # -------------------------------------------------
    # 10. Sidebar – predict new audio
    # -------------------------------------------------
    st.sidebar.header("Predict New Audio")
    uploaded = st.sidebar.file_uploader("Upload .wav/.mp3/.m4a",
                                       type=["wav","mp3","m4a"])

    if uploaded and os.path.exists(os.path.join(OUTPUT_DIR, "model_rf.pkl")):
        with tempfile.NamedTemporaryFile(delete=False,
                suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        cleaned = clean_audio_and_cache(tmp_path)
        y, sr = load_audio(cleaned or tmp_path)
        feats = extract_features(y, sr, "")
        feats_s = scaler.transform(feats.reshape(1, -1))

        rf = joblib.load(os.path.join(OUTPUT_DIR, "model_rf.pkl"))
        pred = rf.predict(feats_s)[0]
        prob = rf.predict_proba(feats_s)[0]
        pred_label = le.inverse_transform([pred])[0]

        st.sidebar.write(f"**Prediction:** `{pred_label}`")
        prob_df = pd.DataFrame({"Class": class_names, "Probability": prob})
        st.sidebar.dataframe(prob_df.sort_values("Probability", ascending=False))

        os.unlink(tmp_path)

    st.success("Pipeline complete – all results in `output_results/`")

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()