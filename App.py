import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.metrics import auc, roc_curve

from brain_mri.config import (CLASS_COLORS, CLASS_NAMES, DEFAULT_CHECKPOINT, DISPLAY_NAMES, FIGURES_DIR,
                              REPORTS_DIR, SAMPLES_DIR)
from brain_mri.gradcam import overlay_heatmap
from brain_mri.inference import Predictor

st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.pred-card {border-radius: 14px; padding: 18px 20px; color: white; margin-bottom: 0.6rem;}
.pred-card h2 {margin: 0; font-size: 1.6rem; color: white;}
.pred-card p {margin: 4px 0 0; opacity: 0.92;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
       background: rgba(255,255,255,0.22); margin-right: 6px;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return Predictor(DEFAULT_CHECKPOINT)


@st.cache_data
def load_reports():
    metrics_path = REPORTS_DIR / "metrics.json"
    preds_path = REPORTS_DIR / "test_predictions.csv"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    preds = pd.read_csv(preds_path) if preds_path.exists() else None
    return metrics, preds


def sample_images():
    if not SAMPLES_DIR.exists():
        return {}
    return {p.stem.replace("_", " ").title(): p for p in sorted(SAMPLES_DIR.glob("*.jpg"))}


def probability_chart(probs: dict):
    names = [DISPLAY_NAMES[c] for c in probs]
    values = [v * 100 for v in probs.values()]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", text=[f"{v:.1f}%" for v in values],
                           textposition="outside", cliponaxis=False, marker_color=[CLASS_COLORS[c] for c in probs]))
    fig.update_layout(height=230, margin=dict(l=10, r=60, t=10, b=10), xaxis=dict(range=[0, 125], visible=False),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(value: float, title: str, color: str):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value * 100, number={"suffix": "%"},
                                 title={"text": title, "font": {"size": 14}},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "steps": [{"range": [0, 60], "color": "#F4F4F6"},
                                                  {"range": [60, 85], "color": "#E9EAF0"},
                                                  {"range": [85, 100], "color": "#DDE0EA"}]}))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=40, b=0))
    return fig


def draw_focus(image: Image.Image, box, size: int):
    if box is None:
        return image
    sx, sy = image.width / size, image.height / size
    out = image.copy()
    ImageDraw.Draw(out).rectangle([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], outline="#00E5FF",
                                  width=max(2, image.width // 150))
    return out


def prediction_card(pred, threshold: float):
    color = CLASS_COLORS[pred.label]
    if not pred.looks_like_mri:
        status = "Input does not look like a grayscale MRI scan"
    elif pred.confidence < threshold:
        status = "Low confidence - review manually"
    else:
        status = "Tumor pattern detected" if pred.has_tumor else "No tumor pattern detected"
    st.markdown(f"""
    <div class="pred-card" style="background: linear-gradient(135deg, {color}, {color}CC);">
      <p>{status}</p>
      <h2>{pred.display_label}</h2>
      <p><span class="pill">Confidence {pred.confidence:.1%}</span>
         <span class="pill">Uncertainty {pred.entropy:.2f}</span>
         {'<span class="pill">TTA</span>' if pred.tta else ''}</p>
    </div>""", unsafe_allow_html=True)


def render_single(name: str, image: Image.Image, pred, settings, img_size: int):
    overlay = overlay_heatmap(image, pred.cam, alpha=settings["alpha"], cmap=settings["cmap"])
    if settings["show_box"]:
        overlay = draw_focus(overlay, pred.focus_box, img_size)
    col1, col2, col3 = st.columns([1, 1, 1.15])
    col1.image(image, caption=f"Input - {name}", use_container_width=True)
    col2.image(overlay, caption=f"{settings['cam_label']} attention heatmap", use_container_width=True)
    with col3:
        prediction_card(pred, settings["threshold"])
        if not pred.is_reliable(settings["threshold"]):
            st.warning("The model is unsure or the image may not be a brain MRI. Treat this result with caution.")
        st.plotly_chart(probability_chart(pred.probs), use_container_width=True, config={"displayModeBar": False})
    with st.expander("How to read the heatmap"):
        st.write("Grad-CAM highlights the regions that most increased the score of the predicted class. "
                 "Red/yellow areas influenced the decision most. For tumor classes these should overlap the lesion; "
                 "attention on the skull or background suggests an unreliable prediction.")


def analyze_tab(predictor, settings):
    st.subheader("Analyze MRI scans")
    source = st.radio("Image source", ["Upload scans", "Try a sample"], horizontal=True, label_visibility="collapsed")
    images = []
    if source == "Upload scans":
        files = st.file_uploader("Upload one or more 2D brain MRI slices", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True)
        images = [(f.name, Image.open(io.BytesIO(f.getvalue())).convert("RGB")) for f in files or []]
    else:
        samples = sample_images()
        if not samples:
            st.info("No sample images found in assets/samples.")
        chosen = st.multiselect("Sample scans", list(samples), default=list(samples)[:1])
        images = [(name, Image.open(samples[name]).convert("RGB")) for name in chosen]

    if not images:
        st.caption("Supported: axial, coronal or sagittal T1/T2 MRI slices as JPG or PNG.")
        return

    progress = st.progress(0.0, text="Running inference...")
    results = []
    for i, (name, image) in enumerate(images):
        results.append((name, image, predictor.predict(image, tta=settings["tta"], cam_method=settings["cam"])))
        progress.progress((i + 1) / len(images), text=f"Analyzed {i + 1}/{len(images)}")
    progress.empty()

    if len(results) == 1:
        render_single(*results[0], settings, predictor.meta["img_size"])
        return

    table = pd.DataFrame([{
        "file": name, "prediction": p.display_label, "confidence": round(p.confidence * 100, 1),
        "uncertainty": round(p.entropy, 3),
        "status": "OK" if p.is_reliable(settings["threshold"]) else "Review",
        **{f"p_{c}": round(v * 100, 1) for c, v in p.probs.items()},
    } for name, _, p in results])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scans analyzed", len(results))
    c2.metric("Tumor detected", int(sum(p.has_tumor for _, _, p in results)))
    c3.metric("Need review", int((table["status"] == "Review").sum()))
    c4.metric("Mean confidence", f"{table['confidence'].mean():.1f}%")

    left, right = st.columns([1.6, 1])
    left.dataframe(table[["file", "prediction", "confidence", "uncertainty", "status"]], use_container_width=True,
                   hide_index=True, column_config={"confidence": st.column_config.ProgressColumn(
                       "confidence", format="%.1f%%", min_value=0, max_value=100)})
    counts = table["prediction"].value_counts().reset_index()
    counts.columns = ["prediction", "count"]
    right.plotly_chart(px.pie(counts, names="prediction", values="count", hole=0.55,
                              color="prediction", color_discrete_map={DISPLAY_NAMES[k]: v for k, v in CLASS_COLORS.items()})
                       .update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0)), use_container_width=True)
    st.download_button("Download results (CSV)", table.to_csv(index=False), "mri_predictions.csv", "text/csv")

    for name, image, pred in results:
        with st.expander(f"{name} - {pred.display_label} ({pred.confidence:.1%})"):
            render_single(name, image, pred, settings, predictor.meta["img_size"])


def performance_tab(predictor):
    metrics, preds = load_reports()
    if metrics is None:
        st.info("Run `python scripts/evaluate.py --test-dir <path>/Testing` to generate performance reports.")
        return

    st.subheader(f"Test-set performance - {metrics['arch']} on {metrics['n_samples']:,} held-out scans")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, [
        ("Accuracy", f"{metrics['accuracy']:.2%}"), ("Macro F1", f"{metrics['macro_f1']:.3f}"),
        ("Macro AUC", f"{metrics['macro_auc']:.3f}"), ("Cohen's kappa", f"{metrics['cohen_kappa']:.3f}"),
        ("Calibration (ECE)", f"{metrics['ece']:.3f}"), ("Latency", f"{metrics['ms_per_image']:.0f} ms/img")]):
        col.metric(label, value)

    labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    left, right = st.columns(2)
    with left:
        normalize = st.toggle("Normalize confusion matrix", value=True)
        cm = np.array(metrics["confusion_matrix"], dtype=float)
        z = cm / cm.sum(1, keepdims=True) if normalize else cm
        text = [[f"{v:.1%}" if normalize else f"{int(v)}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="Blues", text=text, texttemplate="%{text}",
                                   hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>"))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis_autorange="reversed", height=430, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        rep = metrics["report"]
        df = pd.DataFrame([{"class": DISPLAY_NAMES[c], "metric": m.title().replace("-Score", " score"),
                            "value": rep[c][m]} for c in CLASS_NAMES for m in ("precision", "recall", "f1-score")])
        fig = px.bar(df, x="class", y="value", color="metric", barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#5B6EE1", "#29B6A4", "#F3A712"])
        fig.update_layout(title="Precision / recall / F1 per class", yaxis_range=[min(0.6, df.value.min() - 0.05), 1.02],
                          height=470, legend_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    if preds is not None:
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for c in CLASS_NAMES:
                fpr, tpr, _ = roc_curve(preds["label"] == c, preds[f"p_{c}"])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=CLASS_COLORS[c], width=2.5),
                                         name=f"{DISPLAY_NAMES[c]} (AUC {auc(fpr, tpr):.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                                     showlegend=False))
            fig.update_layout(title="ROC curves (one-vs-rest)", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", height=430, legend=dict(x=0.45, y=0.05))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            conf = preds[[f"p_{c}" for c in CLASS_NAMES]].max(axis=1)
            outcome = np.where(preds["label"] == preds["pred"], "Correct", "Wrong")
            fig = px.histogram(x=conf, color=outcome, nbins=40, log_y=True, barmode="overlay", opacity=0.8,
                               color_discrete_map={"Correct": "#29B6A4", "Wrong": "#E4572E"})
            fig.update_layout(title="Confidence of correct vs wrong predictions", xaxis_title="Confidence",
                              yaxis_title="Images (log)", height=430, legend_title=None)
            st.plotly_chart(fig, use_container_width=True)

    history = predictor.meta.get("history") or {}
    if history.get("train_loss"):
        epochs = list(range(1, len(history["train_loss"]) + 1))
        hist_df = pd.DataFrame({"epoch": epochs * 2, "split": ["Train"] * len(epochs) + ["Validation"] * len(epochs),
                                "loss": history["train_loss"] + history["val_loss"],
                                "accuracy": history["train_acc"] + history["val_acc"]})
        left, right = st.columns(2)
        left.plotly_chart(px.line(hist_df, x="epoch", y="loss", color="split", markers=True, title="Loss"),
                          use_container_width=True)
        right.plotly_chart(px.line(hist_df, x="epoch", y="accuracy", color="split", markers=True, title="Accuracy"),
                           use_container_width=True)

    gallery = [(p, t) for p, t in [("gradcam_examples.png", "Grad-CAM examples"),
                                   ("tsne_embeddings.png", "t-SNE of learned embeddings"),
                                   ("misclassified.png", "Most confident mistakes"),
                                   ("calibration.png", "Calibration")] if (FIGURES_DIR / p).exists()]
    if gallery:
        st.subheader("Visual diagnostics")
        tabs = st.tabs([t for _, t in gallery])
        for tab, (p, _) in zip(tabs, gallery):
            tab.image(str(FIGURES_DIR / p), use_container_width=True)


def about_tab(predictor):
    st.subheader("About this project")
    st.markdown(f"""
A deep-learning system that classifies 2D brain MRI slices into **glioma**, **meningioma**, **pituitary tumor**
or **no tumor**, and explains each prediction with a **Grad-CAM heatmap**.

| | |
|---|---|
| Model | `{predictor.meta['arch']}` fine-tuned from ImageNet ({predictor.num_params / 1e6:.1f}M parameters) |
| Data | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (CC BY 4.0) |
| Training | Stratified train/validation split, augmentation, label smoothing, AdamW + warmup-cosine LR, early stopping on validation macro F1 |
| Explainability | Grad-CAM / Grad-CAM++ on the last convolutional block |
| Safety checks | Confidence threshold, predictive entropy and a grayscale check to flag non-MRI inputs |
""")
    st.error("**Research prototype - not a medical device.** Predictions must not be used for diagnosis or treatment "
             "decisions. Always consult a qualified radiologist.")


def sidebar(predictor):
    with st.sidebar:
        st.markdown("## 🧠 Brain MRI Classifier")
        metrics, _ = load_reports()
        st.caption(f"Model: **{predictor.meta['arch']}** · {predictor.num_params / 1e6:.1f}M params · "
                   f"running on {predictor.device.type.upper()}")
        if metrics:
            a, b = st.columns(2)
            a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
            b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.divider()
        st.markdown("#### Settings")
        threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.75, 0.01,
                              help="Predictions below this confidence are flagged for review.")
        tta = st.toggle("Test-time augmentation", value=True, help="Averages predictions over the image and its mirror.")
        cam_label = st.selectbox("Explanation method", ["Grad-CAM", "Grad-CAM++"])
        alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
        cmap = st.selectbox("Heatmap colors", ["jet", "inferno", "turbo", "viridis"])
        show_box = st.toggle("Outline focus region", value=True)
        st.divider()
        st.caption("Research prototype - not for clinical use.")
    return {"threshold": threshold, "tta": tta, "cam": cam_label.lower(), "cam_label": cam_label, "alpha": alpha,
            "cmap": cmap, "show_box": show_box}


def main():
    predictor = get_predictor()
    settings = sidebar(predictor)
    st.title("Brain MRI Tumor Classifier")
    st.caption("Upload MRI slices to classify tumor type and see where the model is looking.")
    analyze, performance, about = st.tabs(["🔬 Analyze", "📊 Model performance", "ℹ️ About"])
    with analyze:
        analyze_tab(predictor, settings)
    with performance:
        performance_tab(predictor)
    with about:
        about_tab(predictor)


main()
