"""
SPECT & Scintigrafia Planare - Controllo Qualità
Basato su pylinac.nuclear (IAEA NMQC toolkit reimplementation)
Supporta serie multi-file DICOM (es. Siemens I100000W, I100001W, ...)
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile, os, io, traceback, copy
from pathlib import Path

import numpy as np
import pydicom

# ── Page config ───────────────────────────────────────────────────────────────
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
  .sub-title  { font-size:1rem; color:#90e0ef; margin-top:0; margin-bottom:1.5rem; }
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
  .section-header { color:#00b4d8; font-size:1.15rem; font-weight:700;
    border-bottom:1px solid #00b4d8; padding-bottom:4px;
    margin-top:1.2rem; margin-bottom:.8rem; }
  .file-chip {
    display:inline-block; background:#1a2a3a; color:#90e0ef;
    border-radius:6px; padding:2px 8px; font-size:.78rem;
    margin:2px; font-family:monospace;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_dicom_safe(data: bytes) -> pydicom.Dataset:
    """Read DICOM from bytes, with force=True for files without preamble."""
    return pydicom.dcmread(io.BytesIO(data), force=True)


def merge_to_multiframe(datasets: list[pydicom.Dataset]) -> pydicom.Dataset:
    """
    Merge a list of single-frame NM DICOM datasets into one multi-frame dataset.
    Sorted by InstanceNumber so frame order is correct.
    """
    # Sort by InstanceNumber (default to index if missing)
    datasets = sorted(
        datasets,
        key=lambda d: int(d.get("InstanceNumber", 0))
    )

    # Stack pixel arrays → shape (N, rows, cols)
    arrays = []
    for d in datasets:
        arr = d.pixel_array
        if arr.ndim == 2:
            arrays.append(arr)
        else:
            # already multi-frame — flatten into list
            for i in range(arr.shape[0]):
                arrays.append(arr[i])

    stacked = np.stack(arrays, axis=0)

    # Build merged dataset from first file's metadata
    merged = copy.deepcopy(datasets[0])
    merged.NumberOfFrames = len(arrays)

    # Write pixel data
    merged.PixelData = stacked.astype(stacked.dtype).tobytes()
    merged["PixelData"].VR = "OB"

    return merged, len(arrays)


def save_merged_dcm(datasets: list[pydicom.Dataset]) -> tuple[str, int]:
    """Merge datasets and save to a temp .dcm file. Returns (path, n_frames)."""
    merged, n_frames = merge_to_multiframe(datasets)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    pydicom.dcmwrite(tmp.name, merged)
    tmp.close()
    return tmp.name, n_frames


def load_uploaded_files(uploaded_files) -> tuple[str, list, dict]:
    """
    Load one or more uploaded files, validate as NM DICOM, merge if needed.
    Returns (dcm_path, datasets, meta_info).
    """
    datasets = []
    errors   = []

    for uf in uploaded_files:
        data = uf.read()
        try:
            ds = read_dicom_safe(data)
            if ds.get("Modality", "") != "NM":
                errors.append(f"⚠️ `{uf.name}` non è NM (Modality={ds.get('Modality','?')})")
                continue
            datasets.append(ds)
        except Exception as e:
            errors.append(f"⚠️ `{uf.name}` non è DICOM valido: {e}")

    if errors:
        for e in errors:
            st.warning(e)

    if not datasets:
        return None, [], {}

    # Collect meta from first dataset
    d0 = datasets[0]
    meta = {
        "studio":  d0.get("StudyDescription", ""),
        "serie":   d0.get("SeriesDescription", ""),
        "modello": d0.get("ManufacturerModelName", d0.get("Manufacturer", "")),
        "data":    d0.get("StudyDate", ""),
        "paziente": str(d0.get("PatientName", "")),
    }

    if len(datasets) == 1:
        # Single file — save directly
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        raw = datasets[0]._write_to_bytes() if hasattr(datasets[0], "_write_to_bytes") else None
        if raw is None:
            pydicom.dcmwrite(tmp.name, datasets[0])
        tmp.close()
        n_frames = int(datasets[0].get("NumberOfFrames", 1))
        dcm_path = tmp.name
    else:
        dcm_path, n_frames = save_merged_dcm(datasets)

    meta["n_files"]  = len(datasets)
    meta["n_frames"] = n_frames

    return dcm_path, datasets, meta


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


def show_all_figs():
    """Render all open matplotlib figures into Streamlit."""
    for fn in plt.get_fignums():
        fig = plt.figure(fn)
        st.image(fig_to_bytes(fig), use_container_width=True)
    plt.close("all")


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
    st.markdown("### 📁 Carica file DICOM")
    st.caption("Puoi caricare **uno o più file** della stessa serie (es. `I100000W`, `I100001W`, …). Verranno fusi automaticamente in un'unica serie multi-frame.")

    uploaded_files = st.file_uploader(
        "File DICOM (con o senza estensione)",
        type=None,
        accept_multiple_files=True,
        help="Seleziona tutti i file della serie. Ordine non importa — vengono ordinati per InstanceNumber.",
    )

    st.markdown("---")
    st.caption("Basato su **pylinac.nuclear** · IAEA NMQC toolkit · v3.45")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">☢️ Controllo Qualità SPECT & Scintigrafia Planare</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analisi automatica basata su IAEA No. 6 · pylinac · Implementazione NMQC</p>', unsafe_allow_html=True)

try:
    from pylinac import nuclear as pnuc
except ImportError:
    st.error("**pylinac non trovato.** Installa con: `pip install pylinac`")
    st.stop()

# ── No files uploaded yet ─────────────────────────────────────────────────────
if not uploaded_files:
    st.info("👈 Carica uno o più file DICOM dalla barra laterale per iniziare.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🔆 Contrasto Tomografico** *(IAEA 4.3.9)*  
Contrasto sfere SPECT, baseline uniformità

**📊 Uniformità Planare** *(IAEA 2.3.3)*  
Uniformità integrale/differenziale UFOV & CFOV
""")
    with col2:
        st.markdown("""
**↻ Centro di Rotazione** *(IAEA 4.3.6)*  
Deviazione dal centro di rotazione atteso

**🔍 Risoluzione Tomografica** *(IAEA 4.3.4)*  
FWHM/FWTM da fit gaussiano su 3 assi

**📈 Max Count Rate** *(IAEA 2.3.11.4)*  
Count rate massimo per frame
""")
    st.stop()

# ── Load & merge files ────────────────────────────────────────────────────────
dcm_path, datasets, meta = load_uploaded_files(uploaded_files)

if dcm_path is None:
    st.error("Nessun file DICOM NM valido trovato tra quelli caricati.")
    st.stop()

# ── File summary banner ───────────────────────────────────────────────────────
n_files  = meta["n_files"]
n_frames = meta["n_frames"]

if n_files == 1:
    st.success(f"✅ **1 file caricato** · {uploaded_files[0].name} · Frame: {n_frames}")
else:
    st.success(f"✅ **{n_files} file caricati** e fusi in {n_frames} frame")
    chips = "".join(f'<span class="file-chip">{uf.name}</span>' for uf in uploaded_files)
    st.markdown(chips, unsafe_allow_html=True)

if meta["studio"] or meta["serie"]:
    st.info(f"📋 **Studio:** {meta['studio']}  |  **Serie:** {meta['serie']}  |  **Scanner:** {meta['modello']}")

st.markdown("---")

_tmp_files = [dcm_path]  # track for cleanup


# ══════════════════════════════════════════════════════════════════════════════
# ANTEPRIMA IMMAGINE CON ROI
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🖼️ Anteprima immagine & ROI", expanded=True):
    try:
        import warnings
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import PowerNorm

        ds_preview = pydicom.dcmread(dcm_path, force=True)
        pixel_array = ds_preview.pixel_array.astype(float)
        spacing = float(ds_preview.get("PixelSpacing", [4.8, 4.8])[0])

        if pixel_array.ndim == 3:
            n_fr = pixel_array.shape[0]
            frame_idx = st.slider("Frame da visualizzare", 0, n_fr - 1, 0, format="Frame %d")
            arr = pixel_array[frame_idx]
        else:
            arr = pixel_array
            frame_idx = 0

        rows, cols = arr.shape

        # Controls in a compact row
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
        ufov_r   = ctrl1.slider("UFOV ratio", 0.80, 1.00, 0.95, 0.01, key="prev_ufov")
        cfov_r   = ctrl2.slider("CFOV ratio", 0.50, 0.90, 0.75, 0.01, key="prev_cfov")
        thresh   = ctrl3.slider("Threshold", 0.50, 1.00, 0.75, 0.05, key="prev_thresh")
        colormap = ctrl4.selectbox("Colormap", ["hot","gray","inferno","plasma","viridis","bone"], index=0, key="prev_cmap")
        log_scale = ctrl5.checkbox("Log scale", value=False, key="prev_log")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pu_prev = pnuc.PlanarUniformity(dcm_path)
            pu_prev.analyze(ufov_ratio=ufov_r, cfov_ratio=cfov_r, threshold=thresh)

        fr_key = str(frame_idx + 1)
        if fr_key not in pu_prev.frame_results:
            fr_key = list(pu_prev.frame_results.keys())[min(frame_idx, len(pu_prev.frame_results)-1)]
        fr_data = pu_prev.frame_results[fr_key]

        ufov_bx = fr_data["ufov"].boundary_x
        ufov_by = fr_data["ufov"].boundary_y
        cfov_bx = fr_data["cfov"].boundary_x
        cfov_by = fr_data["cfov"].boundary_y
        binned  = fr_data["binned_frame"]

        BG = "#0d1b2a"
        TICK = "#90e0ef"
        TITLE_C = "#00b4d8"

        fig = plt.figure(figsize=(17, 5.5), facecolor=BG)
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        def style_ax(ax, title):
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_color(TICK)
            ax.tick_params(colors=TICK, labelsize=8)
            ax.xaxis.label.set_color(TICK)
            ax.yaxis.label.set_color(TICK)
            ax.set_title(title, color=TITLE_C, fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("mm", fontsize=8)
            ax.set_ylabel("mm", fontsize=8)

        extent = [0, cols * spacing, rows * spacing, 0]
        norm = (PowerNorm(gamma=0.5, vmin=arr[arr>0].min() if arr[arr>0].size else 0, vmax=arr.max())
                if log_scale else None)
        cmap_obj = plt.colormaps[colormap]

        # Panel 1 — raw
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(arr, cmap=cmap_obj, origin="upper", extent=extent, norm=norm, interpolation="nearest")
        style_ax(ax1, "Immagine Raw")
        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(colors=TICK, labelsize=7)

        # Panel 2 — raw + UFOV/CFOV
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(arr, cmap=cmap_obj, origin="upper", extent=extent, norm=norm, interpolation="nearest")
        ax2.plot(np.append(ufov_bx, ufov_bx[0]) * spacing,
                 np.append(ufov_by, ufov_by[0]) * spacing,
                 color="#00ffff", lw=2.5, label="UFOV", zorder=5)
        ax2.plot(np.append(cfov_bx, cfov_bx[0]) * spacing,
                 np.append(cfov_by, cfov_by[0]) * spacing,
                 color="#ffdd00", lw=2.5, linestyle="--", label="CFOV", zorder=5)
        style_ax(ax2, "ROI: UFOV & CFOV")
        ax2.legend(loc="upper right", facecolor="#1a2a3a", edgecolor=TICK, labelcolor="white", fontsize=9)

        # Panel 3 — processed
        ax3 = fig.add_subplot(gs[2])
        b_rows, b_cols = binned.shape
        b_extent = [0, b_cols * spacing, b_rows * spacing, 0]
        scale = b_cols / cols
        im3 = ax3.imshow(binned, cmap=cmap_obj, origin="upper", extent=b_extent, interpolation="nearest")
        ax3.plot(np.append(ufov_bx, ufov_bx[0]) * spacing * scale,
                 np.append(ufov_by, ufov_by[0]) * spacing * scale,
                 color="#00ffff", lw=2.5, label="UFOV", zorder=5)
        ax3.plot(np.append(cfov_bx, cfov_bx[0]) * spacing * scale,
                 np.append(cfov_by, cfov_by[0]) * spacing * scale,
                 color="#ffdd00", lw=2.5, linestyle="--", label="CFOV", zorder=5)
        style_ax(ax3, "Immagine Processata (NMQC)")
        cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.ax.tick_params(colors=TICK, labelsize=7)
        ax3.legend(loc="upper right", facecolor="#1a2a3a", edgecolor=TICK, labelcolor="white", fontsize=9)

        # Stats
        nonzero = arr[arr > 0]
        st.markdown(
            f"**Dimensioni:** {rows}×{cols} px &nbsp;|&nbsp; "
            f"**Pixel size:** {spacing:.2f} mm &nbsp;|&nbsp; "
            f"**FOV:** {cols*spacing:.0f}×{rows*spacing:.0f} mm &nbsp;|&nbsp; "
            f"**Max counts:** {arr.max():.0f} &nbsp;|&nbsp; "
            f"**Mean (ROI):** {nonzero.mean():.1f} &nbsp;|&nbsp; "
            f"**Pixel attivi:** {len(nonzero)}"
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
        buf.seek(0)
        img_bytes = buf.read()
        plt.close(fig)

        st.image(img_bytes, use_container_width=True)
        st.download_button("⬇️ Scarica anteprima ROI", img_bytes, "preview_roi.png", "image/png")

    except Exception as e:
        st.warning(f"Anteprima non disponibile: {e}")
        with st.expander("🐛 Dettaglio"):
            st.code(traceback.format_exc())

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
                "Diametri sfere (mm) — virgola",
                value="38, 31.8, 25, 19.1, 12.7, 9.5",
            )
        with col2:
            sphere_angles_str = st.text_input(
                "Angoli sfere (°) — virgola",
                value="-10, -70, -130, 170, 110, 50",
            )
        with col3:
            ufov_ratio    = st.slider("UFOV ratio", 0.80, 1.00, 0.95, 0.01)
            search_win    = st.number_input("Search window (px)", 1, 20, 5)
            search_slices = st.number_input("Search slices", 1, 10, 3)

    if st.button("🚀 Esegui Analisi", type="primary"):
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

            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                metric_card("Baseline Uniformità", f"{results.uniformity_baseline:.4f}", "", "info")
                st.markdown("**Contrasto per sfera:**")
                for name, sphere in results.spheres.items():
                    cv = getattr(sphere, "contrast", getattr(sphere, "contrast_value", "N/A"))
                    status = "pass" if isinstance(cv, float) and cv > 0 else "info"
                    metric_card(f"Sfera {name}", f"{cv:.3f}" if isinstance(cv, float) else str(cv), "", status)
            with col_r2:
                tc.plot(show=False)
                fig = plt.gcf()
                img_bytes = fig_to_bytes(fig)
                plt.close("all")
                st.image(img_bytes, use_container_width=True)
                st.download_button("⬇️ Scarica immagine", img_bytes, "tomographic_contrast.png", "image/png")

        except Exception as e:
            st.error(f"Errore: {e}")
            with st.expander("🐛 Dettaglio"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  UNIFORMITÀ PLANARE
# ══════════════════════════════════════════════════════════════════════════════
elif "Uniformità Planare" in test_choice:
    st.markdown('<div class="section-header">📊 Uniformità Planare (IAEA 2.3.3)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parametri analisi", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        ufov_ratio  = c1.slider("UFOV ratio",   0.80, 1.00, 0.95, 0.01)
        cfov_ratio  = c2.slider("CFOV ratio",   0.50, 0.90, 0.75, 0.01)
        window_size = c3.number_input("Window size (px)", 3, 15, 5)
        threshold   = c4.slider("Threshold",    0.50, 1.00, 0.75, 0.05)

    if st.button("🚀 Esegui Analisi", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                pu = pnuc.PlanarUniformity(dcm_path)
                pu.analyze(ufov_ratio=ufov_ratio, cfov_ratio=cfov_ratio,
                           window_size=window_size, threshold=threshold)
                results = pu.results_data()

            col1, col2 = st.columns([1, 2])
            with col1:
                frames_data = getattr(results, "frames", None)
                if frames_data and isinstance(frames_data, list):
                    for i, frame in enumerate(frames_data):
                        st.markdown(f"**Frame {i+1}**")
                        for attr, label in [
                            ("ufov_integral_uniformity",    "UFOV Integrale"),
                            ("ufov_differential_uniformity","UFOV Differenziale"),
                            ("cfov_integral_uniformity",    "CFOV Integrale"),
                            ("cfov_differential_uniformity","CFOV Differenziale"),
                        ]:
                            if hasattr(frame, attr):
                                v = getattr(frame, attr)
                                metric_card(f"F{i+1} {label}", f"{v:.2f}", "%",
                                            "pass" if v < 5 else "fail")
            with col2:
                pu.plot(show=False)
                show_all_figs()

        except Exception as e:
            st.error(f"Errore: {e}")
            with st.expander("🐛 Dettaglio"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  CENTRO DI ROTAZIONE
# ══════════════════════════════════════════════════════════════════════════════
elif "Centro di Rotazione" in test_choice:
    st.markdown('<div class="section-header">↻ Centro di Rotazione (IAEA 4.3.6)</div>', unsafe_allow_html=True)
    st.info("Analizza la deviazione del pannello SPECT dal centro di rotazione tramite fit sinusoidale.")

    if st.button("🚀 Esegui Analisi", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                cor = pnuc.CenterOfRotation(dcm_path)
                cor.analyze()
                results = cor.results_data()

            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(results, "__dict__"):
                    for k, v in results.__dict__.items():
                        if isinstance(v, (int, float)):
                            unit   = "mm" if "mm" in k else ""
                            status = "pass" if abs(v) < 2 else "fail"
                            metric_card(k.replace("_", " ").title(), f"{v:.3f}", unit, status)
            with col2:
                cor.plot(show=False)
                show_all_figs()

        except Exception as e:
            st.error(f"Errore: {e}")
            with st.expander("🐛 Dettaglio"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  RISOLUZIONE TOMOGRAFICA
# ══════════════════════════════════════════════════════════════════════════════
elif "Risoluzione Tomografica" in test_choice:
    st.markdown('<div class="section-header">🔍 Risoluzione Tomografica (IAEA 4.3.4)</div>', unsafe_allow_html=True)
    st.info("FWHM/FWTM calcolati dal fit gaussiano su tre assi.")

    if st.button("🚀 Esegui Analisi", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                tr = pnuc.TomographicResolution(dcm_path)
                tr.analyze()
                results = tr.results_data()

            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(results, "__dict__"):
                    for k, v in results.__dict__.items():
                        if isinstance(v, (int, float)):
                            metric_card(k.replace("_", " ").title(), f"{v:.3f}",
                                        "mm" if "mm" in k else "", "info")
            with col2:
                tr.plot(show=False)
                show_all_figs()

        except Exception as e:
            st.error(f"Errore: {e}")
            with st.expander("🐛 Dettaglio"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TEST:  MAX COUNT RATE
# ══════════════════════════════════════════════════════════════════════════════
elif "Max Count Rate" in test_choice:
    st.markdown('<div class="section-header">📈 Max Count Rate (IAEA 2.3.11.4)</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parametri analisi", expanded=True):
        frame_resolution = st.number_input(
            "Risoluzione frame (sec/frame)", 0.1, 60.0, 0.5, 0.1)

    if st.button("🚀 Esegui Analisi", type="primary"):
        try:
            with st.spinner("Analisi in corso..."):
                mcr = pnuc.MaxCountRate(dcm_path)
                mcr.analyze(frame_resolution=frame_resolution)
                results = mcr.results_data()

            col1, col2 = st.columns([1, 2])
            with col1:
                if hasattr(mcr, "max_countrate"):
                    metric_card("Max Count Rate", f"{mcr.max_countrate:,.0f}", "counts/s", "info")
                if hasattr(mcr, "max_frame"):
                    metric_card("Frame Massimo", str(mcr.max_frame), "", "info")
                if hasattr(mcr, "max_time"):
                    metric_card("Tempo Massimo", f"{mcr.max_time:.1f}", "s", "info")
            with col2:
                mcr.plot(show=False)
                show_all_figs()

        except Exception as e:
            st.error(f"Errore: {e}")
            with st.expander("🐛 Dettaglio"):
                st.code(traceback.format_exc())


# ── Footer & cleanup ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>⚕️ Uso esclusivamente per controllo qualità interno · "
    "pylinac.nuclear · IAEA NMQC toolkit · Non sostituisce la valutazione del fisico medico.</small>",
    unsafe_allow_html=True,
)
for f in _tmp_files:
    try:
        os.unlink(f)
    except Exception:
        pass
