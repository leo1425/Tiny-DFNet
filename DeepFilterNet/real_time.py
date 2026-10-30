"""
real_time.py – Simulate real-time, frame-by-frame execution of DeepFilterNet models.

Instead of passing the full spectrogram tensor at once, this script:
  1. Pre-computes STFT features for the entire audio (STFT is not the bottleneck).
  2. Iterates over time frames **one at a time**, feeding a small sliding window
     to the model to satisfy convolution/lookahead context requirements.
  3. Manually carries GRU hidden states between frames via forward hooks.
  4. Measures per-frame latency and reports Real-Time Factor (RTF) statistics.

Usage:
    python real_time.py --model-base-dir <path> --noisy-audio <audio.wav> [--out output.wav]
"""

import argparse
import os
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F

from df.config import config
from df.enhance import df_features, init_df, setup_df_argument_parser
from df.io import load_audio, resample, save_audio
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real


# ─────────────────────────────────────────────────────────────────────
# GRU state management via hooks
# ─────────────────────────────────────────────────────────────────────


def _collect_gru_modules(model: nn.Module) -> OrderedDict:
    """Recursively find all nn.GRU leaf modules in the model."""
    grus: OrderedDict = OrderedDict()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.GRU):
            grus[name] = mod
    return grus


def _init_hidden_states(
    gru_modules: OrderedDict, batch_size: int, device: torch.device
) -> Dict[str, Tensor]:
    """Create zero-initialized hidden states for all GRU modules."""
    states: Dict[str, Tensor] = {}
    for name, gru in gru_modules.items():
        num_dir = 2 if gru.bidirectional else 1
        h = torch.zeros(
            gru.num_layers * num_dir, batch_size, gru.hidden_size, device=device
        )
        states[name] = h
    return states


class GRUStateManager:
    """Manages GRU hidden states across frame-by-frame forward passes.

    Registers forward pre-hooks and post-hooks on every nn.GRU in the model to:
      - Inject the stored hidden state *before* each GRU forward call
      - Capture the updated hidden state *after* each GRU forward call

    This allows the model to be called repeatedly with small temporal windows
    while maintaining recurrent state continuity.
    """

    def __init__(self, model: nn.Module, batch_size: int, device: torch.device):
        self.gru_modules = _collect_gru_modules(model)
        self.states = _init_hidden_states(self.gru_modules, batch_size, device)
        self._hooks = []
        self._register_hooks()
        logger.info(f"GRUStateManager: tracking {len(self.gru_modules)} GRU modules")

    def _register_hooks(self):
        for name, gru in self.gru_modules.items():
            # Pre-hook: inject stored hidden state
            def make_pre_hook(n):
                def pre_hook(module, args):
                    # nn.GRU.forward(input, hx=None)
                    inp = args[0]
                    return (inp, self.states[n])
                return pre_hook

            # Post-hook: capture updated hidden state (only the LAST time step)
            def make_post_hook(n):
                def post_hook(module, args, output):
                    # output is (output_tensor, h_n)
                    _, h_n = output
                    self.states[n] = h_n.detach()
                    return output
                return post_hook

            h1 = gru.register_forward_pre_hook(make_pre_hook(name))
            h2 = gru.register_forward_hook(make_post_hook(name))
            self._hooks.extend([h1, h2])

    def reset(self, batch_size: int, device: torch.device):
        """Reset all hidden states to zero."""
        self.states = _init_hidden_states(self.gru_modules, batch_size, device)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────
# Determine the model's temporal context requirements
# ─────────────────────────────────────────────────────────────────────


def _get_model_context(model: nn.Module) -> int:
    """Determine how many past context frames the model needs.

    The model uses nn.ConstantPad2d / nn.ConstantPad3d with negative left
    padding for lookahead.  We need to pass enough frames so that after this
    padding, at least 1 valid output frame remains.

    Returns the minimum number of frames (window size) to feed per step.
    """
    conv_lookahead = 0
    df_lookahead = 0

    # Try to read from model attributes
    if hasattr(model, "pad_feat"):
        # DFNet3/4: pad_feat = ConstantPad2d((0, 0, -conv_la, conv_la), 0)
        pad_mod = model.pad_feat
        if isinstance(pad_mod, nn.ConstantPad2d):
            # padding is (left, right, top, bottom) for 2d
            # here it's used as (freq_l, freq_r, time_l, time_r)
            p = pad_mod.padding
            conv_lookahead = max(0, -p[2])  # negative = lookahead shift
    elif hasattr(model, "pad"):
        # DFNet2: pad = ConstantPad2d((0, 0, -conv_la, conv_la), 0)
        pad_mod = model.pad
        if isinstance(pad_mod, nn.ConstantPad2d):
            p = pad_mod.padding
            conv_lookahead = max(0, -p[2])

    if hasattr(model, "pad_spec"):
        pad_mod = model.pad_spec
        if isinstance(pad_mod, nn.ConstantPad3d):
            p = pad_mod.padding
            df_lookahead = max(0, -p[4])

    lookahead = max(conv_lookahead, df_lookahead)

    # Also account for the DF operation's temporal context (df_order frames)
    df_order = getattr(model, "df_order", 5)

    # The total window must be large enough that:
    # 1) After pad_feat removes `lookahead` frames from the left, T >= 1
    # 2) The DF operation has df_order frames of history
    # We use max(df_order, lookahead + 1) as the window size
    window = max(df_order, lookahead + 1)

    logger.info(
        f"Model context: conv_lookahead={conv_lookahead}, "
        f"df_lookahead={df_lookahead}, df_order={df_order} → window={window}"
    )
    return window


# ─────────────────────────────────────────────────────────────────────
# Core real-time simulation
# ─────────────────────────────────────────────────────────────────────

from torch.profiler import profile, record_function, ProfilerActivity
@torch.no_grad()
def simulate_real_time(
    model: nn.Module,
    df_state,
    audio: Tensor,
    pad: bool = True,
    atten_lim_db: Optional[float] = None,
    warmup_frames: int = 5,
) -> Tuple[Tensor, Dict[str, float]]:
    """Simulate real-time frame-by-frame enhancement.

    Args:
        model: A loaded DeepFilterNet model (1, 2, 3 or 4).
        df_state: DF state for STFT/ISTFT/ERB.
        audio: Time-domain audio [C, T].
        pad: Pad audio to compensate for STFT delay.
        atten_lim_db: Optional noise attenuation limit (dB).
        warmup_frames: Number of initial frames to skip for timing statistics.

    Returns:
        enhanced_audio: Enhanced time-domain audio [C, T].
        stats: Dictionary of latency/RTF statistics.
    """
    model.eval()
    device = get_device()
    bs = audio.shape[0]

    orig_len = audio.shape[-1]
    n_fft, hop = 0, 0
    if pad:
        n_fft, hop = df_state.fft_size(), df_state.hop_size()
        audio = F.pad(audio, (0, n_fft))

    # ── Pre-compute full features (STFT not the bottleneck) ──────────
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=device)
    # spec: [B, 1, T, F, 2]  |  erb_feat: [B, 1, T, E]  |  spec_feat: [B, 1, T, F', 2]
    num_frames = spec.shape[2]
    sr = ModelParams().sr
    hop_size = ModelParams().hop_size
    frame_dur_ms = hop_size / sr * 1000.0

    logger.info(f"Total frames: {num_frames}  |  Frame duration: {frame_dur_ms:.2f} ms")
    logger.info(f"Warmup frames: {warmup_frames}  |  Device: {device}")

    # ── Determine sliding window size ────────────────────────────────
    window = _get_model_context(model)

    # Pre-pad features with zeros on the left so the sliding window is valid
    # even for the first frames
    pad_frames = window - 1
    if pad_frames > 0:
        spec = F.pad(spec, (0, 0, 0, 0, pad_frames, 0))        # pad T dim
        erb_feat = F.pad(erb_feat, (0, 0, pad_frames, 0))       # pad T dim
        # spec_feat has shape [B, 1, T, F', 2] – pad the T dimension
        spec_feat = F.pad(spec_feat, (0, 0, 0, 0, pad_frames, 0))

    # ── Set up GRU state management ──────────────────────────────────
    gru_mgr = GRUStateManager(model, bs, device)

    # ── Frame-by-frame processing loop ───────────────────────────────
    enhanced_frames: List[Tensor] = []
    frame_times: List[float] = []
    for t in range(num_frames): #
        # Extract sliding window of features ending at frame t
        # (with pre-padding, index t in original == index t+pad_frames in padded)
        win_start = t  # = t + pad_frames - (window - 1)  = t
        win_end = t + window  # = t + pad_frames + 1

        spec_win = spec[:, :, win_start:win_end, :, :]
        erb_win = erb_feat[:, :, win_start:win_end, :]
        spec_feat_win = spec_feat[:, :, win_start:win_end]

        # ── Timed forward pass ───────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(spec_win.clone(), erb_win, spec_feat_win)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        frame_times.append(t1 - t0)

        # Extract the LAST frame from the output (that's the current frame)
        out_spec = out[0]  # enhanced spec
        out_frame = out_spec[:, :, -1:, :, :]  # [B, 1, 1, F, 2]
        enhanced_frames.append(out_frame.cpu())

    # ── Remove hooks ─────────────────────────────────────────────────
    gru_mgr.remove_hooks()

    # ── Reconstruct full enhanced spectrogram ────────────────────────
    enhanced_spec = torch.cat(enhanced_frames, dim=2)  # [B, 1, T, F, 2]
    enhanced_spec = as_complex(enhanced_spec.squeeze(1))

    # Optional attenuation limit
    if atten_lim_db is not None and abs(atten_lim_db) > 0:
        lim = 10 ** (-abs(atten_lim_db) / 20)
        # Use original (unpadded) spec for mixing
        orig_spec = spec[:, :, pad_frames:, :, :]
        orig_complex = as_complex(orig_spec.squeeze(1).cpu())
        enhanced_spec = orig_complex * lim + enhanced_spec * (1 - lim)

    # ── ISTFT (synthesis) ────────────────────────────────────────────
    enhanced_audio = torch.as_tensor(df_state.synthesis(enhanced_spec.numpy()))
    if pad:
        assert n_fft % hop == 0
        d = n_fft - hop
        enhanced_audio = enhanced_audio[:, d : orig_len + d]

    # ── Compute statistics ───────────────────────────────────────────
    measured_times = frame_times[warmup_frames:] if warmup_frames < len(frame_times) else frame_times
    measured_times_np = np.array(measured_times) * 1000.0  # ms

    total_proc_time = sum(frame_times)
    audio_duration = orig_len / sr
    rtf = total_proc_time / audio_duration if audio_duration > 0 else float("inf")

    stats = {
        "num_frames": num_frames,
        "frame_duration_ms": frame_dur_ms,
        "window_size": window,
        "warmup_frames": warmup_frames,
        "measured_frames": len(measured_times),
        "mean_latency_ms": float(np.mean(measured_times_np)),
        "std_latency_ms": float(np.std(measured_times_np)),
        "median_latency_ms": float(np.median(measured_times_np)),
        "p95_latency_ms": float(np.percentile(measured_times_np, 95)),
        "p99_latency_ms": float(np.percentile(measured_times_np, 99)),
        "max_latency_ms": float(np.max(measured_times_np)),
        "min_latency_ms": float(np.min(measured_times_np)),
        "total_processing_time_s": total_proc_time,
        "audio_duration_s": audio_duration,
        "rtf": rtf,
        "meets_realtime": rtf < 1.0,
    }

    return enhanced_audio, stats


# ─────────────────────────────────────────────────────────────────────
# Pretty-print statistics
# ─────────────────────────────────────────────────────────────────────


def print_stats(stats: Dict[str, float]):
    """Print a nicely formatted summary table of the latency statistics."""
    budget = stats["frame_duration_ms"]
    header = "═" * 60
    print(f"\n{header}")
    print(f"  Real-Time Simulation Results")
    print(f"{header}")
    print(f"  Audio duration        : {stats['audio_duration_s']:.3f} s")
    print(f"  Total frames          : {stats['num_frames']:.0f}")
    print(f"  Sliding window size   : {stats['window_size']:.0f} frames")
    print(f"  Measured frames       : {stats['measured_frames']:.0f}  (excl. {stats['warmup_frames']:.0f} warmup)")
    print(f"  Frame duration budget : {budget:.2f} ms")
    print(f"{'─' * 60}")
    print(f"  Per-frame latency:")
    print(f"    Mean   : {stats['mean_latency_ms']:8.3f} ms   {'✓' if stats['mean_latency_ms'] < budget else '✗'}")
    print(f"    Median : {stats['median_latency_ms']:8.3f} ms   {'✓' if stats['median_latency_ms'] < budget else '✗'}")
    print(f"    Std    : {stats['std_latency_ms']:8.3f} ms")
    print(f"    P95    : {stats['p95_latency_ms']:8.3f} ms   {'✓' if stats['p95_latency_ms'] < budget else '✗'}")
    print(f"    P99    : {stats['p99_latency_ms']:8.3f} ms   {'✓' if stats['p99_latency_ms'] < budget else '✗'}")
    print(f"    Max    : {stats['max_latency_ms']:8.3f} ms")
    print(f"    Min    : {stats['min_latency_ms']:8.3f} ms")
    print(f"{'─' * 60}")
    print(f"  Total processing time : {stats['total_processing_time_s']:.3f} s")
    rtf_str = f"{stats['rtf']:.4f}"
    rt_emoji = "✓ REAL-TIME" if stats["meets_realtime"] else "✗ NOT REAL-TIME"
    print(f"  Real-Time Factor (RTF): {rtf_str}   {rt_emoji}")
    print(f"{header}\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = setup_df_argument_parser()
    parser.add_argument(
        "--noisy-audio",
        type=str,
        required=True,
        help="Path to a single noisy audio file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for the enhanced audio. "
        "Defaults to <input>_rt_enhanced.wav in the same directory.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of initial frames to skip for timing statistics (default: 5).",
    )
    parser.add_argument(
        "--no-delay-compensation",
        dest="compensate_delay",
        action="store_false",
        help="Don't pad to compensate the STFT delay.",
    )
    parser.add_argument(
        "--atten-lim",
        type=float,
        default=None,
        help="Attenuation limit in dB by mixing enhanced with noisy signal.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the output audio (only report timing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load model ───────────────────────────────────────────────────
    model, df_state, suffix, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    df_sr = ModelParams().sr

    # ── Load audio ───────────────────────────────────────────────────
    if not os.path.isfile(args.noisy_audio):
        logger.error(f"File not found: {args.noisy_audio}")
        exit(1)
    audio, meta = load_audio(args.noisy_audio, df_sr)
    logger.info(
        f"Loaded audio: {args.noisy_audio}  |  "
        f"{audio.shape[-1] / df_sr:.2f}s  |  "
        f"channels={audio.shape[0]}  |  sr={df_sr}"
    )

    # ── Run real-time simulation ─────────────────────────────────────
    enhanced, stats = simulate_real_time(
        model,
        df_state,
        audio,
        pad=args.compensate_delay,
        atten_lim_db=args.atten_lim,
        warmup_frames=args.warmup_frames,
    )

    # ── Print results ────────────────────────────────────────────────
    print_stats(stats)

    # ── Save output ──────────────────────────────────────────────────
    if not args.no_save:
        enhanced = resample(enhanced.to("cpu"), df_sr, meta.sample_rate)
        if args.out is None:
            base, ext = os.path.splitext(args.noisy_audio)
            args.out = f"{base}_rt_enhanced{ext}"
        save_audio(args.out, enhanced, sr=meta.sample_rate, log=True)
        logger.info(f"Enhanced audio saved to: {args.out}")


if __name__ == "__main__":
    main()
