#!/usr/bin/env python3
"""
show_df_complex.py – Visualise the complex spectrogram feature-extraction
                     pipeline of DeepFilterNet (the Deep Filtering branch).

Usage (from the project root):
    conda activate DFNet2_fixed
    export PYTHONPATH=$PWD/DeepFilterNet
    python tools/show_df_complex.py path/to/audio.wav --model-base-dir /path/to/model

Produces 3 separate plots:
    1. Full STFT spectrogram       – magnitude in dB (all F bins)
    2. Frequency-cropped spectrum  – first nb_df bins only  (the "f_df" cut)
    3. Unit-normalised spectrum    – after unit_norm()  (model input for DF coefs)
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
from df.config import config
from df.enhance import df_features
from df.io import load_audio
from df.model import ModelParams
from df.utils import as_complex
from libdf import DF

plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 35,
    'axes.labelsize': 35,
    'axes.titlesize': 35,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
})

output_ext = "svg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DeepFilterNet complex spectrogram feature-extraction stages."
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
        help="Model directory containing config.ini. "
        "Leave empty to use the default DeepFilterNet3 config.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Directory where the 3 plot files will be saved. Defaults to current dir.",
    )
    return parser.parse_args()


def load_config(model_base_dir):
    """Load a config.ini so that ModelParams / DfParams are available."""
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
    Run the complex-spectrogram feature-extraction pipeline step by step.
    All outputs are in **linear magnitude** (no dB), matching the actual
    complex branch which never applies a log transformation.

    Returns
    -------
    stft_mag   : np.ndarray [F, T]      – full STFT linear magnitude
    crop_mag   : np.ndarray [nb_df, T]  – cropped to first nb_df bins
    norm_mag   : np.ndarray [nb_df, T]  – after unit_norm, linear magnitude
    sr, hop, fft_size, nb_df : int       – parameters
    """
    sr = p.sr
    hop = p.hop_size
    fft_size = p.fft_size
    nb_df = p.nb_df

    # ── load & resample ──────────────────────────────────────────────────
    audio, _meta = load_audio(audio_path, sr=sr)  # [C, T]

    # ── 1. Full STFT ─────────────────────────────────────────────────────
    spec_np = df_state.analysis(audio.numpy())  # complex [C, T', F]
    spec_ch0 = spec_np[0]  # [T', F]
    stft_mag = np.abs(spec_ch0)  # [T', F]  — linear magnitude

    # ── 2. Frequency-band cut (first nb_df bins) ─────────────────────────
    spec_crop = spec_ch0[:, :nb_df]  # [T', nb_df]
    crop_mag = np.abs(spec_crop)  # [T', nb_df]  — linear magnitude

    # ── 3. Unit-normalised complex spectrum (via df_features) ────────────
    # df_features returns spec_feat as real-valued [B, 1, T', nb_df, 2]
    _, _, spec_feat = df_features(audio, df_state, nb_df, device="cpu")
    # Convert back to complex to get magnitude
    spec_feat_complex = as_complex(spec_feat.squeeze(0).squeeze(0))  # [T', nb_df]
    norm_mag = spec_feat_complex.abs().numpy()  # [T', nb_df]  — linear magnitude

    # Transpose to [F, T] for plotting
    return (
        stft_mag.T,
        crop_mag.T,
        norm_mag.T,
        sr,
        hop,
        fft_size,
        nb_df,
    )


def _plot_single(data, time_axis, y_axis, title, ylabel, output_path,
                 cmap="inferno", vmin=None, vmax=None, cb_format=None,
                 cb_label=None):
    """Save a single square spectrogram as a PDF/SVG."""
    fig_size = 7
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size), constrained_layout=True)

    im = ax.pcolormesh(time_axis, y_axis, data,
                       shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
                       rasterized=True)
    ax.set_aspect("auto")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [s]")
    #ax.set_title(title)
    fig.colorbar(im, ax=ax, format=cb_format, pad=0.02, shrink=0.85)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_features(stft_mag, crop_mag, norm_mag,
                  sr, hop, fft_size, nb_df, output_dir="."):
    """Create 3 separate plots, one per complex-spec extraction stage."""
    os.makedirs(output_dir, exist_ok=True)

    n_frames = stft_mag.shape[1]
    time_axis = np.arange(n_frames) * hop / sr  # seconds

    # Full frequency axis (all F bins)
    n_freq_bins = stft_mag.shape[0]
    freq_axis_khz = np.linspace(0, sr / 2, n_freq_bins) / 1000.0

    # Cropped frequency axis (first nb_df bins only)
    freq_axis_crop_khz = np.linspace(0, sr / 2, n_freq_bins)[:nb_df] / 1000.0

    cmap = "inferno"

    # ── 1. Full STFT with f_df cutoff highlighted ──────────────────────────
    f_df_khz = freq_axis_khz[nb_df - 1]  # cutoff frequency in kHz
    fig_size = 7
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size), constrained_layout=True)

    im = ax.pcolormesh(time_axis, freq_axis_khz, stft_mag,
                       shading="nearest", cmap=cmap, rasterized=True)
    ax.set_aspect("auto")
    ax.set_ylabel("Frequency [kHz]")
    ax.set_xlabel("Time [s]")
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)

    # Gray out the region above f_df
    ax.axhspan(f_df_khz, freq_axis_khz[-1], color="black", alpha=0.5, zorder=2)

    # Dashed line at f_df cutoff
    ax.axhline(f_df_khz, color="red", linestyle="--", linewidth=2, zorder=3)
    ax.text(time_axis[-1] * 0.98, f_df_khz + 0.3, f"$f_{{df}}$ = {f_df_khz:.1f} kHz",
            color="red", fontsize=25, ha="right", va="bottom", zorder=4)

    fig.savefig(os.path.join(output_dir, f"1_complex_stft.{output_ext}"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.join(output_dir, f'1_complex_stft.{output_ext}')}")

    # ── 2. Cropped spectrogram (first nb_df bins only) ───────────────────
    _plot_single(
        crop_mag, time_axis, freq_axis_crop_khz,
        title=f"Cropped Spectrogram (first {nb_df} bins)",
        ylabel="Frequency [kHz]",
        output_path=os.path.join(output_dir, f"2_complex_cropped.{output_ext}"),
        cmap=cmap, cb_label="Magnitude",
    )

    # ── 3. Unit-normalised spectrogram (linear magnitude) ────────────────
    _plot_single(
        norm_mag, time_axis, freq_axis_crop_khz,
        title="Unit-Normalised Spectrogram (model input)",
        ylabel="Frequency [kHz]",
        output_path=os.path.join(output_dir, f"3_complex_unit_normed.{output_ext}"),
        cmap=cmap, cb_label="Normalised magnitude (a.u.)",
    )


def main():
    args = parse_args()

    # Load project config
    load_config(args.model_base_dir)

    # Build libdf state
    df_state, p = build_df_state()

    # Extract features
    (stft_mag, crop_mag, norm_mag,
     sr, hop, fft_size, nb_df) = extract_features(args.audio_path, df_state, p)

    print(f"Audio loaded at {sr} Hz | FFT={fft_size} | Hop={hop} | nb_df={nb_df}")
    print(f"  Full STFT shape:       {stft_mag.shape}  (F × T)  [linear]")
    print(f"  Cropped shape:         {crop_mag.shape}  (nb_df × T)  [linear]")
    print(f"  Unit-normed shape:     {norm_mag.shape}  (nb_df × T)  [linear]")

    # Plot
    plot_features(stft_mag, crop_mag, norm_mag,
                  sr, hop, fft_size, nb_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
