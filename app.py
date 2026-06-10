"""
SPECT & Scintigrafia Planare - Controllo Qualità
Basato su pylinac.nuclear (IAEA NMQC toolkit reimplementation)
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
import io
import traceback
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPECT/Scintigrafia QC",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] * { color: #e0e8f0 !important; }
  .main-title { font-size:2.2rem; font-weight:700; color:#00b4d8; margin-bottom:0; }
  .sub-title  { font-size:1rem;   color:#90e0ef; margin-top:0; margin-bottom:1.5rem; }
  .metric-card {
    background:#0d1b2a; border-radius:10px; padding:16px 20px;
    border-left:4px solid #00b4d8; margin-bottom:10px;
  }
  .metric-label { font-size:.8rem; color:#90e0ef; font-weight:600; letter-spacing:.05em; }
  .metric-value { font-size:1.6rem; font-weight:700; color:#caf0f8; }
  .metric-unit  { font-size:.85rem; color:#90e0ef; }
  .badge-pass { background:#1a3a2a; color:#40c878; border-radius:6px; padding:2px 10px; font-size:.85rem; font-weight:600; }
  .badge-fail { background:#3a1a1a; color:#ff6b6b; border-radius:6px; padding:2px 10px; font-size:.85rem; font-weight:600; }
  .badge-info { background:#1a2a3a; color:#74c7ec; border-radius:6px; padding:2px 10px; font-size:.85rem; font-weight:600; }
  .stAlert    { border-radius:10px !important; }
  .section-header { color:#00b4d8; font-size:1.15rem; font-weight:700; border-bottom:1px solid #00b4d8; padding-bottom:4px; margin-top:1.2rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_upload(uploaded_file) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def metric_card(label, value, unit="", status=None):
    badge = ""
    if status == "pass":
        badge = '<span class="badge-pass">✓ PASS</span>'
    elif status == "fail":
        badge = '<span class="badge-fail">✗ FAIL</span>'
    elif status == "info":
        badge = '<span class="badge-info">ℹ INFO</span>'
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label} {badge}</div>
      <div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ☢️ SPECT / Scintigrafia QC")
    st.markdown("---")

    test_choice = st.radio(
        "**Seleziona il Test**",
        options=[
            "🔆 Contrasto Tomografico (SPECT)",
            "📊 Uniformità Planare",
            "↻ Centro di Rotazione",
            "🔍 Risoluzione Tomografica",
            "📈 Max Count Rate",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📁 Carica immagine DICOM")
    uploaded = st.file_uploader(
        "File `.dcm`",
        type=["dcm", "DCM"],
        help="Immagine DICOM acquisita con gamma-camera / SPECT",
    )

    st.markdown("---")
    st.caption("Basato su **pylinac.nuclear** · IAEA NMQC toolkit · v3.45")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">☢️ Controllo Qualità SPECT & Scintigrafia Planare</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisi automatica basata su IAEA No. 6 · pylinac · Implementazione NMQC</p>', unsafe_allow_html=True)

# ── Import check ──────────────────────────────────────────────────────────────
try:
    from pylinac import nuclear as pnuc
    PYLINAC_OK = True
except ImportError:
    PYLINAC_OK = False

if not PYLINAC_OK:
    st.error("""
**pylinac non trovato.** Installa con:
```bash
pip install pylinac
```
""")
    st.stop()

# ── No file uploaded yet ───────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Carica un file DICOM dalla barra laterale per iniziare l'analisi.")

    st.markdown("---")
    st.markdown("### 📋 Test disponibili")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🔆 Contrasto Tomografico** *(IAEA test 4.3.9)*
- Misura il contrasto del sistema SPECT
- Rileva automaticamente slice di uniformità e sfere
- Calcola contrasto per ogni sfera

**📊 Uniformità Planare** *(IAEA test 2.3.3)*
- Uniformità integrale e differenziale
- Analisi UFOV e CFOV per ogni frame
- Sliding window per uniformità differenziale
""")
    with col2:
        st.markdown("""
**↻ Centro di Rotazione** *(IAEA test 4.3.6)*
- Deviazione dal centro di rotazione atteso
- Fit sinusoidale delle proiezioni

**🔍 Risoluzione Tomografica** *(IAEA test 4.3.4)*
- FWHM/FWTM calcolato dal fit gaussiano
- Profili su tre assi (x, y, z)

**📈 Max Count Rate** *(IAEA test 2.3.11.4)*
- Count rate massimo per frame
- Plot count rate vs numero frame
""")
    st.stop()

# ── File uploaded — run analysis ───────────────────────────────────────────────
dcm_path = save_upload(uploaded)

st.success(f"✅ File caricato: `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TEST:  CONTRASTO TOMOGRAFICO
# ══════════════════════════════════════════════════════════════════════════════
if "Contrasto Tomografico" in test_choice:
    st.markdown('<div class="section-header">🔆 Contrasto Tomografico (IAEA 4.3.9)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parametri analisi", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sphere_diameters_str = st.text_input(
                "Diametri sfere (mm) — separati da virgola",
                value="38, 31.8, 25, 19.1, 12.7, 9.5",
                help="Es: 38, 31.8, 25, 19.1, 12.7, 9.5"
            )
        with col2:
            sphere_angles_str = st.text_input(
                "Angoli sfere (°) — separati da virgola",
                value="-10, -70, -130, 170, 110, 50",
                help="Un angolo per ogni sfera"
            )
        with col3:
            ufov_ratio   = st.slider("UFOV ratio", 0.80, 1.00, 0.95, 0.01)
            search_win   = st.number_input("Search window (px)", 1, 20, 5)
            search_slices = st.number_input("Search slices", 1, 10, 3)

    if st.button("🚀 Esegui Analisi Contrasto Tomografico", type="primary"):
        try:
            sphere_diameters = [float(x.strip()) for x in sphere_diameters_str.split(",")]
            sphere_angles    = [float(x.strip()) for x in sphere_angles_str.split(",")]

            if len(sphere_diameters) != len(sphere_angles):
                st.error("Il numero di diametri deve corrispondere al numero di angoli!")
                st.stop()

            with st.spinner("Analisi in corso..."):
                tc = pnuc.TomographicContrast(dcm_path)
                tc.analyze(
                    sphere_diameters_mm=tuple(sphere_diameters),
                    sphere_angles=tuple(sphere_angles),
                    ufov_ratio=ufov_ratio,
                    search_window_px=search_win,
                    search_slices=search_slices,
                )
                results = tc.results_data()

            st.markdown("#### 📊 Risultati")

            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                metric_card("Baseline Uniformità", f"{results.uniformity_baseline:.4f}", "", "info")
                st.markdown("**Contrasto per sfera:**")
                for name, sphere in results.spheres.items():
                    contrast_val = sphere.contrast if hasattr(sphere, "contrast") else getattr(sphere, "contrast_value", "N/A")
                    status = "pass" if isinstance(contrast_val, float) and contrast_val > 0 else "info"
                    metric_card(
                        f"Sfera {name}",
                        f"{contrast_val:.3f}" if isinstance(contrast_val, float) else str(contrast_val),
                        "",
                        status
                    )

            with col_r2:
                st.markdown("**📈 Visualizzazione ROI**")
                fig = tc.plot(show=False)
                if fig is None:
                    fig, _ = plt.subplots()
                    tc.plot()
                    fig = plt.gcf()
                st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close("all")
                st.download_button(
                    "⬇️ Scarica immagine analisi",
                    data=fig_to_bytes(fig),
                    file_name="tomographic_contrast.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            with st.expander("🐛 Dettaglio errore"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  UNIFORMITÀ PLANARE
# ══════════════════════════════════════════════════════════════════════════════
elif "Uniformità Planare" in test_choice:
    st.markdown('<div class="section-header">📊 Uniformità Planare (IAEA 2.3.3)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parametri analisi", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ufov_ratio = st.slider("UFOV ratio", 0.80, 1.00, 0.95, 0.01)
        with col2:
            cfov_ratio = st.slider("CFOV ratio", 0.50, 0.90, 0.75, 0.01)
        with col3:
            window_size = st.number_input("Window size (px)", 3, 15, 5)
        with col4:
            threshold = st.slider("Threshold", 0.50, 1.00, 0.75, 0.05)

    if st.button("🚀 Esegui Analisi Uniformità Planare", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                pu = pnuc.PlanarUniformity(dcm_path)
                pu.analyze(
                    ufov_ratio=ufov_ratio,
                    cfov_ratio=cfov_ratio,
                    window_size=window_size,
                    threshold=threshold,
                )
                results = pu.results_data()

            st.markdown("#### 📊 Risultati")

            # Display results - handle both list and dict structures
            frames_data = None
            if hasattr(results, "frames"):
                frames_data = results.frames
            elif hasattr(results, "__dict__"):
                frames_data = results.__dict__

            if frames_data:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if isinstance(frames_data, list):
                        for i, frame in enumerate(frames_data):
                            st.markdown(f"**Frame {i+1}**")
                            if hasattr(frame, "ufov_integral_uniformity"):
                                iu = frame.ufov_integral_uniformity
                                du = frame.ufov_differential_uniformity
                                ciu = frame.cfov_integral_uniformity
                                cdu = frame.cfov_differential_uniformity
                                metric_card(f"F{i+1} UFOV Integrale", f"{iu:.2f}", "%",
                                            "pass" if iu < 5 else "fail")
                                metric_card(f"F{i+1} UFOV Differenziale", f"{du:.2f}", "%",
                                            "pass" if du < 5 else "fail")
                                metric_card(f"F{i+1} CFOV Integrale", f"{ciu:.2f}", "%",
                                            "pass" if ciu < 5 else "fail")
                                metric_card(f"F{i+1} CFOV Differenziale", f"{cdu:.2f}", "%",
                                            "pass" if cdu < 5 else "fail")
                    else:
                        st.json(frames_data if isinstance(frames_data, dict) else str(frames_data))

                with col2:
                    st.markdown("**📈 Visualizzazione UFOV/CFOV**")
                    pu.plot(show=False)
                    fig = plt.gcf()
                    st.image(fig_to_bytes(fig), use_container_width=True)
                    plt.close("all")
            else:
                st.info("Analisi completata. Visualizzazione:")
                pu.plot(show=False)
                fig = plt.gcf()
                st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close("all")

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            with st.expander("🐛 Dettaglio errore"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  CENTRO DI ROTAZIONE
# ══════════════════════════════════════════════════════════════════════════════
elif "Centro di Rotazione" in test_choice:
    st.markdown('<div class="section-header">↻ Centro di Rotazione (IAEA 4.3.6)</div>', unsafe_allow_html=True)

    st.info("Il test analizza la deviazione del pannello SPECT dal centro di rotazione atteso, tramite fit sinusoidale delle proiezioni.")

    if st.button("🚀 Esegui Analisi Centro di Rotazione", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                cor = pnuc.CenterOfRotation(dcm_path)
                cor.analyze()
                results = cor.results_data()

            st.markdown("#### 📊 Risultati")

            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(results, "x_offset_mm"):
                    status_x = "pass" if abs(results.x_offset_mm) < 2.0 else "fail"
                    metric_card("Deviazione X", f"{results.x_offset_mm:.3f}", "mm", status_x)
                if hasattr(results, "y_offset_mm"):
                    status_y = "pass" if abs(results.y_offset_mm) < 2.0 else "fail"
                    metric_card("Deviazione Y", f"{results.y_offset_mm:.3f}", "mm", status_y)
                if hasattr(results, "__dict__"):
                    for k, v in results.__dict__.items():
                        if isinstance(v, (int, float)) and k not in ("x_offset_mm", "y_offset_mm"):
                            metric_card(k.replace("_", " ").title(), f"{v:.4f}", "", "info")

            with col2:
                st.markdown("**📈 Fit Sinusoidale & Residui**")
                cor.plot(show=False)
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close("all")

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            with st.expander("🐛 Dettaglio errore"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  RISOLUZIONE TOMOGRAFICA
# ══════════════════════════════════════════════════════════════════════════════
elif "Risoluzione Tomografica" in test_choice:
    st.markdown('<div class="section-header">🔍 Risoluzione Tomografica (IAEA 4.3.4)</div>', unsafe_allow_html=True)

    st.info("Misura la risoluzione SPECT tramite FWHM/FWTM calcolati dal fit gaussiano su tre assi.")

    if st.button("🚀 Esegui Analisi Risoluzione Tomografica", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                tr = pnuc.TomographicResolution(dcm_path)
                tr.analyze()
                results = tr.results_data()

            st.markdown("#### 📊 Risultati")
            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(results, "__dict__"):
                    for k, v in results.__dict__.items():
                        if isinstance(v, (int, float)):
                            unit = "mm" if "mm" in k else ""
                            metric_card(k.replace("_", " ").title(), f"{v:.3f}", unit, "info")

            with col2:
                st.markdown("**📈 Profili e Fit Gaussiani (X, Y, Z)**")
                tr.plot(show=False)
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close("all")

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            with st.expander("🐛 Dettaglio errore"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  MAX COUNT RATE
# ══════════════════════════════════════════════════════════════════════════════
elif "Max Count Rate" in test_choice:
    st.markdown('<div class="section-header">📈 Max Count Rate (IAEA 2.3.11.4)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parametri analisi", expanded=True):
        frame_resolution = st.number_input(
            "Risoluzione frame (sec/frame)",
            min_value=0.1, max_value=60.0, value=0.5, step=0.1,
            help="Intervallo temporale tra frame acquisiti"
        )

    if st.button("🚀 Esegui Analisi Max Count Rate", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                mcr = pnuc.MaxCountRate(dcm_path)
                mcr.analyze(frame_resolution=frame_resolution)
                results = mcr.results_data()

            st.markdown("#### 📊 Risultati")
            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(mcr, "max_countrate"):
                    metric_card("Max Count Rate", f"{mcr.max_countrate:,.0f}", "counts/s", "info")
                if hasattr(mcr, "max_frame"):
                    metric_card("Frame Massimo", str(mcr.max_frame), "", "info")
                if hasattr(mcr, "max_time"):
                    metric_card("Tempo Massimo", f"{mcr.max_time:.1f}", "s", "info")
                if hasattr(results, "__dict__"):
                    for k, v in results.__dict__.items():
                        if isinstance(v, (int, float)) and k not in ("max_countrate", "max_frame", "max_time"):
                            metric_card(k.replace("_", " ").title(), f"{v:.3f}", "", "info")

            with col2:
                st.markdown("**📈 Count Rate vs Frame**")
                mcr.plot(show=False)
                fig = plt.gcf()
                st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close("all")

        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")
            with st.expander("🐛 Dettaglio errore"):
                st.code(traceback.format_exc())

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>⚕️ Uso esclusivamente per controllo qualità interno · Basato su <b>pylinac.nuclear</b> "
    "(reimplementazione IAEA NMQC toolkit) · Non sostituisce la valutazione del fisico medico.</small>",
    unsafe_allow_html=True
)

# Cleanup temp file
try:
    os.unlink(dcm_path)
except Exception:
    pass
