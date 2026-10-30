#!/usr/bin/env python3
"""
show_df.py – Visualise every feature-extraction stage of DeepFilterNet.

Usage (from the project root):
    conda activate DFNet2_fixed
    export PYTHONPATH=$PWD/DeepFilterNet
    python tools/show_df.py path/to/audio.wav [--model-base-dir /path/to/model]

Produces a 4-panel figure:
    1. Complex spectrogram  (magnitude in dB)  – from STFT via df_state.analysis()
    2. Log-power spectrogram                    – 10·log10(|X|²)
    3. ERB spectrogram                          – ERB filterbank applied (dB)
    4. Normalised ERB spectrogram               – exponential unit-norm (as fed to the model)
"""

import argparse
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
from df.config import config
from df.enhance import df_features
from df.io import load_audio
from df.model import ModelParams
from libdf import DF, erb

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 35,
    'axes.labelsize': 35,
    'axes.titlesize': 35,
    'xtick.labelsize': 25,  # Size of x-axis tick labels
    'ytick.labelsize': 25,
})

output_ext = "svg"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DeepFilterNet feature-extraction stages."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to a .wav (or any torchaudio-supported) audio file.",
    )
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default=None,
        help="Model directory containing config.ini (needed for DF / ERB params). "
        "Leave empty to use the default DeepFilterNet3 config.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Directory where the 4 PDF files will be saved. Defaults to current dir.",
    )
    return parser.parse_args()


def load_config(model_base_dir):
    """Load a config.ini so that ModelParams / DfParams are available."""
    import os

    from df.enhance import get_model_basedir, PRETRAINED_MODELS

    if model_base_dir is None or model_base_dir in PRETRAINED_MODELS:
        model_base_dir = get_model_basedir(model_base_dir or "DeepFilterNet3")
    config_path = os.path.join(model_base_dir, "config.ini")
    config.load(
        config_path,
        config_must_exist=True,
        allow_defaults=True,
        allow_reload=True,
    )


def build_df_state():
    """Create a libdf.DF state from the loaded config."""
    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    return df_state, p


def extract_features(audio_path, df_state, p):
    """
    Run the DeepFilterNet feature-extraction pipeline step by step.

    Returns
    -------
    spec_mag_db : np.ndarray [F, T]   – magnitude spectrogram in dB
    log_pow_db  : np.ndarray [F, T]   – log-power spectrogram in dB
    erb_db      : np.ndarray [B, T]   – ERB spectrogram (dB)
    erb_normed  : np.ndarray [B, T]   – normalised ERB spectrogram
    sr          : int                  – sampling rate used
    hop         : int                  – hop size
    fft_size    : int                  – FFT size
    """
    sr = p.sr
    hop = p.hop_size
    fft_size = p.fft_size

    # ── load & resample ──────────────────────────────────────────────────
    audio, _meta = load_audio(audio_path, sr=sr)  # [C, T]

    # ── 1. STFT via df_state.analysis() ──────────────────────────────────
    # analysis expects numpy [C, T] → returns complex numpy [C, T', F]
    spec = df_state.analysis(audio.numpy())  # complex64, shape [C, T', F]
    spec = spec[0]  # take first channel → [T', F]

    # magnitude (linear scale – the raw STFT output)
    spec_mag = np.abs(spec)  # [T', F]

    # ── 2. Log-power spectrogram ─────────────────────────────────────────
    power = spec_mag ** 2
    log_pow_db = 10.0 * np.log10(power + 1e-12)  # [T', F]

    # ── 3. ERB spectrogram ───────────────────────────────────────────────
    erb_fb = df_state.erb_widths()
    # erb() from libdf expects [C, T', F] complex input, returns [C, T', B]
    # db=True  → output is in dB
    erb_spec = erb(spec[np.newaxis], erb_fb, db=True)  # [1, T', B]
    erb_spec = erb_spec[0]  # [T', B]

    # ── 4. Normalised ERB spectrogram ────────────────────────────────────
    # Use df_features() directly to match the exact model input.
    # df_features calls erb(spec, erb_fb) with default db=True, then erb_norm.
    _, erb_normed, _ = df_features(audio, df_state, p.nb_df, device="cpu")
    # erb_normed shape: [B, 1, T', nb_erb] → squeeze to [T', nb_erb]
    erb_normed = erb_normed.squeeze(0).squeeze(0).numpy()  # [T', B]

    # Transpose everything to [F/B, T] for plotting (frequency on y-axis)
    return (
        spec_mag.T,
        log_pow_db.T,
        erb_spec.T,
        erb_normed.T,
        sr,
        hop,
        fft_size,
    )


def _plot_single(data, time_axis, y_axis, title, ylabel, output_path,
                 cmap="inferno", vmin=None, vmax=None, cb_format=None,
                 cb_label=None, hgrid_ticks=None):
    """Save a single square spectrogram as a PDF."""
    fig_size = 7  # inches – square figure
    fig, ax = plt.subplots(figsize=(fig_size+1, fig_size), constrained_layout=True)

    im = ax.pcolormesh(time_axis, y_axis, data,
                       shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
                       rasterized=True)
    ax.set_aspect("auto")          # fill the square canvas
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [s]")
    #ax.set_title(title)
    fig.colorbar(im, ax=ax, format=cb_format, pad=0.02, shrink=0.85)
    # cbar = fig.colorbar(im, ax=ax, format=cb_format, pad=0.02, shrink=0.85)
    # if cb_label is not None:
    #     cbar.set_label(cb_label)

    # Thin horizontal gridlines at each band boundary
    if hgrid_ticks is not None:
        ax.set_yticks(hgrid_ticks, minor=True)
        ax.grid(axis="y", which="minor", color="white", linewidth=0.3, alpha=0.6)
        ax.tick_params(axis="y", which="minor", length=0)  # hide minor tick marks

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_features(spec_mag_db, log_pow_db, erb_db, erb_normed,
                  sr, hop, fft_size, output_dir="."):
    """Create 4 separate square-figure PDFs, one per extraction stage."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_frames = spec_mag_db.shape[1]
    time_axis = np.arange(n_frames) * hop / sr  # seconds

    n_freq_bins = spec_mag_db.shape[0]
    freq_axis_khz = np.linspace(0, sr / 2, n_freq_bins) / 1000.0

    n_erb_bands = erb_db.shape[0]
    erb_axis = np.arange(n_erb_bands)

    cmap = "inferno"

    # ── 1. STFT magnitude spectrogram (linear) ────────────────────────────
    _plot_single(
        spec_mag_db, time_axis, freq_axis_khz,
        title="STFT Magnitude Spectrogram (linear)",
        ylabel="Frequency [kHz]",
        output_path=os.path.join(output_dir, f"1_stft_spectrogram.{output_ext}"),
        cmap=cmap, cb_label="Magnitude",
    )

    # ── 2. Log-power spectrogram ─────────────────────────────────────────
    vmax = log_pow_db.max()
    vmin = max(vmax - 100, log_pow_db.min())
    _plot_single(
        log_pow_db, time_axis, freq_axis_khz,
        title="Log-Power Spectrogram",
        ylabel="Frequency [kHz]",
        output_path=os.path.join(output_dir, f"2_log_power_spectrogram.{output_ext}"),
        cmap=cmap, vmin=vmin, vmax=vmax, cb_format="%+2.0f", cb_label="Power [dB]",
    )

    # ── 3. ERB spectrogram ───────────────────────────────────────────────
    vmax_erb = erb_db.max()
    vmin_erb = max(vmax_erb - 80, erb_db.min())
    erb_grid = np.arange(n_erb_bands) - 0.5  # band boundaries
    _plot_single(
        erb_db, time_axis, erb_axis,
        title="ERB Spectrogram",
        ylabel="ERB Band Index",
        output_path=os.path.join(output_dir, f"3_erb_spectrogram.{output_ext}"),
        cmap=cmap, vmin=vmin_erb, vmax=vmax_erb, cb_format="%+2.0f", cb_label="Energy [dB]",
        hgrid_ticks=erb_grid,
    )

    # ── 4. Normalised ERB spectrogram ────────────────────────────────────
    _plot_single(
        erb_normed, time_axis, erb_axis,
        title="Normalised ERB Spectrogram (model input)",
        ylabel="ERB Band Index",
        output_path=os.path.join(output_dir, f"4_erb_normed_spectrogram.{output_ext}"),
        cmap=cmap, cb_label="Normalised energy (a.u.)",
        hgrid_ticks=erb_grid,
    )


def main():
    args = parse_args()

    # Load project config (needed by ModelParams / DfParams)
    load_config(args.model_base_dir)

    # Build libdf state
    df_state, p = build_df_state()

    # Extract features step-by-step
    (spec_mag_db, log_pow_db, erb_db, erb_normed,
     sr, hop, fft_size) = extract_features(args.audio_path, df_state, p)

    print(f"Audio loaded at {sr} Hz | FFT={fft_size} | Hop={hop}")
    print(f"  STFT shape:           {spec_mag_db.shape}  (F × T)  [linear]")
    print(f"  Log-power shape:      {log_pow_db.shape}  (F × T)")
    print(f"  ERB shape:            {erb_db.shape}  (B × T)")
    print(f"  Normalised ERB shape: {erb_normed.shape}  (B × T)")

    # Plot – one PDF per feature
    plot_features(spec_mag_db, log_pow_db, erb_db, erb_normed,
                  sr, hop, fft_size, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
