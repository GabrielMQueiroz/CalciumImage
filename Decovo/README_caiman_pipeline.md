# 🧠 Decovo v2: CaImAn CNMF-E 1-Photon Pipeline (TIFF & AVI | Local & Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrielMQueiroz/CalciumImage/blob/main/Decovo/notebooks/Decovo_Colab.ipynb)

Decovo v2 is a unified, resumable CaImAn CNMF-E pipeline for 1-photon miniscope calcium imaging. It processes **both multi-page `.tif`/`.tiff` stacks and `msCam*.avi` chunks** using a single validated parameter preset (`presets/miniscope_1p_default.yaml`) across:

1. **☁️ Google Colab Notebook (`notebooks/Decovo_Colab.ipynb`)** — Form-based workflow with interactive threshold sliders for users without local setup.
2. **🖥️ Local Interactive Dashboard (`caiman_interactive.py`)** — Browser-based GUI with two-stage caching (adjusting thresholds re-runs only CNMF-E, not motion correction).
3. **💻 Local Command-Line Interface (`decovo` / `caiman_auto_pipeline.py`)** — Headless single-session and batch execution.

All three front-ends produce outputs directly compatible with the **CleanMiniscope Cell Curator GUI** (`analysis_results.hdf5` with `Cn` and `dims` attached) as well as MATLAB (`ms.mat` with both `RawTraces` and `DeconvTraces` oriented as `time × neurons`, plus `SFP.mat`).

---

## ☁️ Option 1: Run on Google Colab (Easiest — No Local Installation)

1. Upload your recording folder (containing either one multi-page `.tif`/`.tiff` file or `msCam*.avi` chunks) to **Google Drive**.
2. Click the **Open in Colab** badge above (or upload `notebooks/Decovo_Colab.ipynb` to Google Colab).
3. Run **Cells 1–3** to install CondaColab, CaImAn, and `decovo`, and to mount your Google Drive.
4. In **Cell 4**, paste your Drive folder path and set your `frame_rate` (Hz).
5. Run **Cell 5** to check the recording: if the TIFF header contains a frame rate, it is displayed so you can verify it by hand.
6. Run **Cell 6 (Stage A)** to run preprocessing, motion correction, and summary images (`Cn` and `PNR`). Checkpoints are automatically saved to your Drive.
7. In **Cell 7 (Stage B)**, drag the `min_corr` and `min_pnr` sliders to preview neuron seeds and click **Extract with these** — this re-runs only CNMF-E on the cached movie without redoing motion correction.
8. Run **Cell 9** to confirm all files (`analysis_results.hdf5`, `ms.mat`, `SFP.mat`, `summary_figure.png`, `config_used.yaml`) are synced to `<your_folder>/results_decovo/` on Google Drive.

---

## 🖥️ Option 2: Run Locally (Dashboard or CLI)

### One-Time Local Setup

Activate your `caiman` Conda environment and install `decovo` in editable mode:

```powershell
conda activate caiman
cd D:\Master_Gabriel\Progaming\CalciumImage\Decovo
pip install -e .
```

### Launch the Local Interactive Dashboard

```powershell
panel serve caiman_interactive.py --show
```

1. Enter the path to your folder (containing `.tif`/`.tiff` or `msCam*.avi`) or a `.tif` file directly.
2. Verify the **Frame Rate (Hz)** (automatically populated if present in TIFF metadata).
3. Click **🚀 Run Pipeline** (or **⚡ Stage A Only**).
4. In the **Threshold Inspector**, adjust `min_corr` and `min_pnr` and click **✅ Apply & Extract Neurons (Stage B only)** — motion correction is cached and skipped!

### Run from the Command Line (CLI)

```powershell
# Single session (auto-detects .tif/.tiff stack OR msCam*.avi chunks)
python caiman_auto_pipeline.py --path "D:\data\Animal1\pre" --frame-rate 20

# Manual thresholds (skips Stage A automatically if already cached!)
python caiman_auto_pipeline.py --path "D:\data\Animal1\pre" --min-corr 0.84 --min-pnr 8.0

# Force a fresh run (ignore cached stages)
python caiman_auto_pipeline.py --path "D:\data\Animal1\pre" --force

# Batch mode across multiple animals/sessions
python caiman_auto_pipeline.py --path "D:\data" --batch --batch-pattern "Animal*/pre,Animal*/post/d*"
```

---

## ⚙️ Default Validated Parameters (`presets/miniscope_1p_default.yaml`)

| Group | Parameter | Default | Notes |
|---|---|---|---|
| Acquisition | `frame_rate` | User-selected (default `20.0` Hz) | Read from TIFF metadata if present & verified by user |
| Preprocess | `spatial_downsample` | `0.6` | Chunk-streamed (`chunk_frames: 1000`) |
| Motion | `max_shifts` | `[4, 4]` | Rigid NoRMCorre with `gSig_filt: [3, 3]` |
| CNMF-E | `gSig` / `gSiz` | `3.0` / `13` | `gSiz = 4 * gSig + 1` |
| CNMF-E | `rf` / `stride` | `40` / `20` | Patch half-size & overlap |
| CNMF-E | `merge_thr` | `0.7` | Temporal correlation merge threshold |
| Evaluation | `min_SNR` / `rval_thr` | `3.0` / `0.95` | `use_cnn: false` (1-photon) |
