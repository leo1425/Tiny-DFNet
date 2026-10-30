# Script to measure metrics: PESQ, STOI, MACs, and Parameter Count for DeepFilterNet
# metrics.py
# Run with: PYTHONPATH=./DeepFilterNet conda run -n DFNET2 python metrics.py --model_dir ownexec --num_samples 100 --boxplot
# python -W ignore metrics.py --model_dir '/Users/leopoldh/Library/CloudStorage/GoogleDrive-notreplystopstop@gmail.com/My Drive/Memoire/OwnModels/ownexec_vad_pretrain_change_vad_loss' --dataset_dir '/Users/leopoldh/DOCUMENTS_LOCAUX/ECOLE/Memoire/dataset/hdf5/all/' --num_samples 30 --boxplot --plot_output Nesy_hyper.pdf
#from IPython.core import pylabtools
import os
import re
import sys
import time
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm
import h5py
from pesq import pesq
import pystoi
from types import SimpleNamespace
#import noisereduce as nr
import scipy.io.wavfile as wav
from df.loss import SiSdr

# Ensure DeepFilterNet folder is in python import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepFilterNet"))
from df.enhance import init_df, enhance, df_features

def rms(x):
    """Calculate the Root Mean Square (RMS) of a signal tensor."""
    return torch.sqrt(torch.mean(x ** 2) + 1e-10)

def mix_at_snr(clean, noise, snr_db):
    """Mix clean speech and noise at a target SNR in dB."""
    # Ensure they have the same number of channels and samples
    clean_channels, clean_len = clean.shape
    noise_channels, noise_len = noise.shape
    
    # 1. Align channels
    if noise_channels != clean_channels:
        if noise_channels > clean_channels:
            noise = noise[:clean_channels, :]
        else:
            noise = noise.repeat(clean_channels, 1)
            
    # 2. Align length
    if noise_len < clean_len:
        reps = (clean_len + noise_len - 1) // noise_len
        noise = noise.repeat(1, reps)[:, :clean_len]
    elif noise_len > clean_len:
        noise = noise[:, :clean_len]

    rms_clean = rms(clean)
    rms_noise = rms(noise)
    if rms_noise > 1e-10:
        target_rms_noise = rms_clean * (10.0 ** (-snr_db / 20.0))
        factor = target_rms_noise / rms_noise
        noisy = clean + factor * noise
    else:
        noisy = clean.clone()
    return noisy

def load_hdf5_signal(group, key, codec):
    """Load and optionally decode an audio signal from an HDF5 group."""
    ds = group[key]
    if codec == "pcm":
        sample = torch.from_numpy(ds[...]).float()
        # Scale to -1.0 to 1.0 depending on dtype
        if ds.dtype == np.int16:
            sample = sample / 32768.0
        elif ds.dtype == np.int32:
            sample = sample / 2147483648.0
        if sample.dim() == 1:
            sample.unsqueeze_(0)
        return sample
    else:
        import io
        wav, _ = torchaudio.load(io.BytesIO(ds[...].tobytes()), format=codec.lower())
        return wav

def main(ex=False, model_dir="", dataset_dir="", nb=10):
    if ex ==True:
        args = SimpleNamespace()
        args.model_dir = model_dir
        args.dataset_dir = dataset_dir
        args.num_samples = nb
        args.boxplot = True
        args.boxplot_snrs = "-5,5,20"
        args.default_snr = 10.0
        args.plot_output = "metrics_boxplot.png"
        args.csv_output = "metrics_data.csv"
        args.plot_from_csv = None
        args.epoch = "best"
        args.output_audio_path = None
        args.valentini = True

    else:
        parser = argparse.ArgumentParser(description="Measure PESQ, STOI, MACs, and Parameter Count of DeepFilterNet.")
        parser.add_argument('--model_dir', type=str, default="ownexec", help="Path to DeepFilterNet model directory.")
        parser.add_argument('--dataset_dir', type=str, default="dataset/hdf5", help="Path to dataset/hdf5 directory.")
        parser.add_argument('--num_samples', type=int, default=-1, help="Number of test set samples to evaluate (-1 for all).")
        parser.add_argument('--boxplot', action='store_true', help="Generate boxplots of PESQ and STOI across SNRs.")
        parser.add_argument('--boxplot_snrs', type=str, default="-5,0,5,10,20", help="Comma-separated SNRs for the boxplot.")
        parser.add_argument('--default_snr', type=float, default=10.0, help="Default SNR to mix at if --boxplot is not set.")
        parser.add_argument('--plot_output', type=str, default="metrics_boxplot.pdf", help="Path to save the generated boxplot.")
        parser.add_argument('--csv_output', type=str, default="metrics_data.csv", help="Path to save the evaluation data CSV.")
        parser.add_argument('--plot_from_csv', type=str, default=None, help="Path to an existing evaluation CSV to generate boxplots from directly.")
        parser.add_argument('--epoch', type=str, default="best", help="Epoch/checkpoint to load.")
        parser.add_argument('--output_audio_path', type=str, default=None, help="Export all audio (clean/noise/enhanced) files for visuzalization.")
        parser.add_argument("--valentini", action="store_true", help="Use Valentini's test set.", default=False)
        
        args = parser.parse_args()

    if args.plot_from_csv:
        if not os.path.exists(args.plot_from_csv):
            print("Error: CSV file not found at: %s" % args.plot_from_csv)
            sys.exit(1)
            
        print("Generating boxplot from existing CSV data: %s..." % args.plot_from_csv)
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = pd.read_csv(args.plot_from_csv)
        
        # Setup aesthetic styling for a premium feel
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Helvetica Neue', 'Arial']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        pesq_color = "#3498db"
        stoi_color = "#2ecc71"
        
        sns.boxplot(x='SNR', y='PESQ', data=df, ax=axes[0], width=0.4, color=pesq_color,
                    boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))
        axes[0].set_title('PESQ Score distribution', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_xlabel('SNR (dB)', fontsize=12, labelpad=10)
        axes[0].set_ylabel('PESQ Score (Wideband)', fontsize=12, labelpad=10)
        axes[0].tick_params(labelsize=10)
        
        sns.boxplot(x='SNR', y='STOI', data=df, ax=axes[1], width=0.4, color=stoi_color,
                    boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))
        axes[1].set_title('STOI Score distribution', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xlabel('SNR (dB)', fontsize=12, labelpad=10)
        axes[1].set_ylabel('STOI Score', fontsize=12, labelpad=10)
        axes[1].tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(args.plot_output, dpi=300)
        print("Saved boxplots to %s" % args.plot_output)
        
        # Print summary per SNR
        print("\n--- Evaluation Results Summary by SNR ---")
        for snr in sorted(df['SNR'].unique()):
            snr_df = df[df['SNR'] == snr]
            avg_pesq = snr_df['PESQ'].dropna().mean()
            avg_stoi = snr_df['STOI'].dropna().mean()
            avg_sisdr = snr_df['Si-SDR'].dropna().mean()
            avg_rtf = snr_df['RTF'].dropna().mean() if 'RTF' in df.columns else None
            rtf_str = (" | Avg RTF: %.4f" % avg_rtf) if avg_rtf is not None and not pd.isna(avg_rtf) else ""
            print("SNR: %2d dB | Avg PESQ: %.4f | Avg STOI: %.4f | Avg Si-SDR: %.4f%s" % 
                  (snr, avg_pesq if not pd.isna(avg_pesq) else float('nan'), avg_stoi if not pd.isna(avg_stoi) else float('nan'), avg_sisdr if not pd.isna(avg_sisdr) else float('nan'), rtf_str))
        return

    # 1. Load config and instantiate model
    print("Instantiating DeepFilterNet model from directory: %s..." % args.model_dir)
    model, df_state, suffix, epoch_num = init_df(
        model_base_dir=args.model_dir,
        epoch=args.epoch,
        config_allow_defaults=True,
        log_file=None
    )
    print("Loaded checkpoint from epoch: %s" % str(epoch_num))

    # 2. Computational Complexity Metrics: Params and MACs
    print("\n--- Measuring Computational Complexity ---")
    
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    parms = {"enc":0, "erb_dec": 0, "df_dec":0}
    for n, p in model.named_parameters():
        if n.startswith("enc"):
            parms["enc"] += p.numel()
        elif n.startswith("erb_dec"):
            parms["erb_dec"] += p.numel()
        elif n.startswith("df_dec"):
            parms["df_dec"] += p.numel()
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: %d" % total_params)
    print("Trainable Parameters: %d" % trainable_params)
    print("Encoder Parameters: %d" % parms["enc"])
    print("ERB Decoder Parameters: %d" % parms["erb_dec"])
    print("DF Decoder Parameters: %d" % parms["df_dec"])
    
    # MACs (using THOP)
    try:
        import thop
        device = next(model.parameters()).device
        dummy_audio = torch.zeros(1, 48000, device='cpu')  # 1 second of audio at 48kHz
        spec, erb_feat, spec_feat = df_features(dummy_audio, df_state, getattr(model, "nb_df", 96), device=device)
        
        macs, params = thop.profile(model, inputs=(spec, erb_feat, spec_feat), verbose=False)
        print("MACs (Multiply-Accumulates per 1s audio): %.2f M" % (macs / 1e6))
        print("MACs (GFLOPs equivalent per 1s audio): %.4f GFLOPs" % (2 * macs / 1e9))
    except ImportError:
        print("Note: 'thop' package is not installed or import failed. Skipping MACs estimation.")
    except Exception as e:
        print("Error measuring MACs via thop: %s" % str(e))

    # 3. Acoustic Quality Metrics on Testset: PESQ & STOI
    print("\n--- Evaluating Test Set Performance ---")

    sisdr = SiSdr()
    
    speech_h5_path = os.path.join(args.dataset_dir, "TEST_SET_SPEECH.hdf5")
    noise_h5_path = os.path.join(args.dataset_dir, "TEST_SET_NOISE.hdf5")
    
    if not os.path.exists(speech_h5_path) or not os.path.exists(noise_h5_path):
        print("Error: HDF5 files not found in: %s" % args.dataset_dir)
        sys.exit(1)
        
    with h5py.File(speech_h5_path, "r", libver="latest", swmr=True) as speech_h5, \
         h5py.File(noise_h5_path, "r", libver="latest", swmr=True) as noise_h5:
         
        speech_grp = speech_h5["speech"]
        noise_grp = noise_h5["noise"]
        
        speech_keys = sorted(list(speech_grp.keys()))
        noise_keys = sorted(list(noise_grp.keys()))
        
        num_test_files = len(speech_keys)
        print("Total test set files available: %d" % num_test_files)
        
        if args.num_samples <= 0 or args.num_samples > num_test_files:
            num_eval_samples = num_test_files
        else:
            num_eval_samples = args.num_samples
            
        sample_rate = df_state.sr()
        
        # Setup Resamplers: Wideband PESQ expects 16kHz, STOI expects 10kHz
        resample_16k = T.Resample(orig_freq=sample_rate, new_freq=16000)
        resample_10k = T.Resample(orig_freq=sample_rate, new_freq=10000)

        #stats = {'SNR': [], 'gate_erb_mean': [], 'gate_df_mean': [], 'gate_vad_mean': []}#'bypass_rate': [], 'erb_only_rate': [], 'full_rate': [], 'temperature': []}
        stats = {'SNR': []}
        if args.boxplot:
            target_snrs = [float(x.strip()) for x in args.boxplot_snrs.split(',')]
            print("Evaluating PESQ & STOI across SNRs %s on %d test samples..." % (target_snrs, num_eval_samples))
            
            boxplot_data = []
            if args.valentini:
                    print("-"*5, "Using Valentini tesset", "-"*5)
                    target_snrs = [0]
                    path_clean = "/Users/leopoldh/DOCUMENTS_LOCAUX/ECOLE/Memoire/dataset/clean_testset_wav/"
                    path_noise = "/Users/leopoldh/DOCUMENTS_LOCAUX/ECOLE/Memoire/dataset/noisy_testset_wav/"
                    valentini_clean = [os.path.join(path_clean, filepath) for filepath in os.listdir(path_clean) if filepath.endswith(".wav")][:num_eval_samples]
                    valentini_noise = [os.path.join(path_noise, filepath) for filepath in os.listdir(path_noise) if filepath.endswith(".wav")][:num_eval_samples]

            for i in tqdm.tqdm(range(num_eval_samples)):
                # if(i!=5):
                #     continue
                clean_key = speech_keys[i]
                noise_key = noise_keys[i]

                
                
                clean_signal = load_hdf5_signal(speech_grp, clean_key, speech_h5.attrs.get("codec", "pcm"))
                noise_signal = load_hdf5_signal(noise_grp, noise_key, noise_h5.attrs.get("codec", "pcm"))

                # print(f"size {clean_signal[0,:20]}")
                # vad_transform = torchaudio.transforms.Vad(
                #     sample_rate=sample_rate,
                #     trigger_level=7.0,
                #     trigger_time=0.25,
                #     search_time=1.0
                # )
                # clean_signal = vad_transform(clean_signal)
                # Remove every white noises 
                # clean_signal[(clean_signal >= -0.01) & (clean_signal <= 0.01)] = 0.0
                # clean_signal = nr.reduce_noise(y=clean_signal, sr=sample_rate, prop_decrease=0.8)
                # clean_signal = torch.tensor(clean_signal)

                
                
                for snr in target_snrs:
                    if args.valentini:
                        clean_signal, sr = torchaudio.load(valentini_clean[i])
                        noisy_signal, sr = torchaudio.load(valentini_noise[i])
                    else:
                        noisy_signal = mix_at_snr(clean_signal, noise_signal, snr)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    start_time = time.perf_counter()
                    denoised_signal = enhance(model, df_state, noisy_signal, pad=True)
                    if hasattr(model, "get_gate_stats"):
                        stats['SNR'].append(snr)
                        for key, value in model.get_gate_stats().items():
                            if key not in stats.keys():
                                stats[key] = []
                            stats[key].append(value)
                        # gate_erb = model.get_gate_stats()["gate_erb"]
                        # gate_df = model.get_gate_stats()["gate_df"]
                        # gate_vad = model.get_gate_stats()["gate_vad"]
                        # nb_non_processed = (gate_vad == 0.0) | ((gate_df == 0.0) & (gate_erb == 0.0))
                        # nb_erb_only = (gate_vad == 1.0) & (gate_df == 0.0)
                        # nb_full_processed = (gate_vad == 1.0) & (gate_df == 1.0)
                        # stats["bypass_rate"].append(nb_non_processed.sum().item() / len(gate_erb))
                        # stats["erb_only_rate"].append(nb_erb_only.sum().sum().item() / len(gate_erb))
                        # stats["full_rate"].append(nb_full_processed.sum().item() / len(gate_erb))
                        # print("-----")
                        # print(stats["bypass_rate"])
                        # print(stats["gate_erb_mean"])
                        # for key, val in stats.items():
                        #     print(f"{key}: {len(val)}")
                        
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    if args.output_audio_path:
                        os.makedirs(args.output_audio_path, exist_ok=True)
                        base_name = "sample_%04d_snr_%+05.1f" % (i, snr)
                        
                        def save_wav(signal, filename):
                            # Signal shape: [1, T] or [T]
                            if signal.dim() == 1:
                                signal = signal.unsqueeze(0)
                            torchaudio.save(os.path.join(args.output_audio_path, filename), signal.cpu(), sample_rate)

                        # Save Clean (Downsampled to 16k for consistency)
                        save_wav(clean_signal, base_name + "_clean.wav")
                        
                        # Save Noisy (Downsampled to 16k)
                        save_wav(noisy_signal, base_name + "_noisy.wav")
                        
                        # Save Enhanced (Downsampled to 16k)
                        save_wav(denoised_signal, base_name + "_enhanced.wav")
                        
                        # OPTIONAL: Save Gates/Masks (Time-aligned, original sample rate)
                        # This helps verify the model's logic
                        # if hasattr(model, 'get_gate_stats'):
                        #     stats_buf = model.get_gate_stats()
                        #     if 'gate_erb_est' in stats_buf and 'gate_df_est' in stats_buf:
                        #        erb_gate = stats_buf['gate_erb_est'][0] # Take first sample
                        #        df_gate = stats_buf['gate_df_est'][0]
                               
                        #        # Reshape for saving if necessary
                        #        if erb_gate.dim() == 2:
                        #            erb_gate = erb_gate.mean(dim=0, keepdim=True) # Mono
                        #        if df_gate.dim() == 2:
                        #            df_gate = df_gate.mean(dim=0, keepdim=True)
                                   
                            #    torchaudio.save(os.path.join(args.output_audio_path, base_name + "_gate_erb.wav"), erb_gate.cpu(), sample_rate)
                            #    torchaudio.save(os.path.join(args.output_audio_path, base_name + "_gate_df.wav"), df_gate.cpu(), sample_rate)
                        
                        # OPTIONAL: Save Log-Spectrograms (Visual)
                        # Useful for debug/viz if needed, uncomment if you want 100s of files!
                        # clean_mag = torch.log10(as_complex(clean_signal).abs().mean(dim=0, keepdim=True) + 1e-6)
                        # noisy_mag = torch.log10(as_complex(noisy_signal).abs().mean(dim=0, keepdim=True) + 1e-6)
                        # enh_mag = torch.log10(as_complex(denoised_signal).abs().mean(dim=0, keepdim=True) + 1e-6)
                        # torchaudio.save(os.path.join(args.output_audio_path, base_name + "_spec_clean.wav"), clean_mag.cpu(), sample_rate)
                        # torchaudio.save(os.path.join(args.output_audio_path, base_name + "_spec_noisy.wav"), noisy_mag.cpu(), sample_rate)
                        # torchaudio.save(os.path.join(args.output_audio_path, base_name + "_spec_enh.wav"), enh_mag.cpu(), sample_rate)
                        
                    elapsed_time = time.perf_counter() - start_time
                    audio_duration = denoised_signal.shape[-1] / sample_rate
                    sample_rtf = elapsed_time / audio_duration
                    
                    # Prepare for metric calculations
                    clean_tensor = clean_signal.float().view(1, -1)
                    denoised_tensor = denoised_signal.float().view(1, -1)
                    
                    # Downsample for PESQ
                    clean_16k = resample_16k(clean_tensor).squeeze(0).numpy()
                    denoised_16k = resample_16k(denoised_tensor).squeeze(0).numpy()
                    
                    # Downsample for STOI
                    clean_10k = resample_10k(clean_tensor).squeeze(0).numpy()
                    denoised_10k = resample_10k(denoised_tensor).squeeze(0).numpy()
                    
                    # Calculate PESQ
                    score_pesq = None
                    try:
                        score_pesq = pesq(16000, clean_16k, denoised_16k, 'wb')
                    except Exception:
                        score_pesq = 0
                        pass
                        
                    # Calculate STOI
                    score_stoi = None
                    try:
                        score_stoi = pystoi.stoi(clean_10k, denoised_10k, 10000, extended=False)
                    except Exception:
                        pass

                    
                    # Calculate Si-SDR
                    score_sisdr = None
                    try:
                        score_sisdr = sisdr(denoised_tensor, clean_tensor)
                    except Exception:
                        pass

                        
                    if score_pesq is not None or score_stoi is not None:
                        boxplot_data.append({
                            'SNR': snr,
                            'PESQ': score_pesq,
                            'STOI': score_stoi,
                            'RTF': sample_rtf,
                            'Si-SDR': score_sisdr.item(),
                        })
                        
            # Generate boxplots
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.DataFrame(boxplot_data)
            
            # Save evaluation data to CSV
            df.to_csv(args.csv_output, index=False)
            print("Saved evaluation data to CSV: %s" % args.csv_output)
            
            # Setup aesthetic styling for a premium feel
            sns.set_theme(style="whitegrid")
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Helvetica Neue', 'Arial']
            
            fig, axes = plt.subplots(1, 3, figsize=(14, 6))
            pesq_color = "#3498db" # Sleek Blue
            stoi_color = "#2ecc71" # Sleek Green
            
            # Left plot: PESQ
            sns.boxplot(x='SNR', y='PESQ', data=df, ax=axes[0], width=0.4, color=pesq_color,
                        boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))
            axes[0].set_title('PESQ Score distribution', fontsize=14, fontweight='bold', pad=15)
            axes[0].set_xlabel('SNR (dB)', fontsize=12, labelpad=10)
            axes[0].set_ylabel('PESQ Score (Wideband)', fontsize=12, labelpad=10)
            axes[0].tick_params(labelsize=10)
            axes[0].set_ylim(0, 5)
            
            # Right plot: STOI
            sns.boxplot(x='SNR', y='STOI', data=df, ax=axes[1], width=0.4, color=stoi_color,
                        boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))
            axes[1].set_title('STOI Score distribution', fontsize=14, fontweight='bold', pad=15)
            axes[1].set_xlabel('SNR (dB)', fontsize=12, labelpad=10)
            axes[1].set_ylabel('STOI Score', fontsize=12, labelpad=10)
            axes[1].tick_params(labelsize=10)
            axes[1].set_ylim(0.75,1)

             # Right plot: Si-SDR
            sns.boxplot(x='SNR', y='Si-SDR', data=df, ax=axes[2], width=0.4, color=stoi_color,
                        boxprops=dict(alpha=0.8), flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, linestyle='none'))
            axes[2].set_title('Si-SDR Score distribution', fontsize=14, fontweight='bold', pad=15)
            axes[2].set_xlabel('SNR (dB)', fontsize=12, labelpad=10)
            axes[2].set_ylabel('Si-SDR Score', fontsize=12, labelpad=10)
            axes[2].tick_params(labelsize=10)
            axes[2].set_ylim(0.75,)
            
            plt.tight_layout()
            plt.savefig(args.plot_output, dpi=300)
            print("\nSaved boxplots to %s" % args.plot_output)
            plt.close()

            

            # Plot stats boxplots
            #if hasattr(model, "get_gate_stats"):
            try:
                df_stats = pd.DataFrame(stats)
                print(df_stats.mean())
                df_stats.to_csv("stats_gate.csv", index=False)
                print("Saved stas data to CSV: stats_gate.csv")

                df_stats = df_stats.drop(columns=["temperature"], errors='ignore')
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_stats, palette="Set2")
                plt.title("Distribution of Rates and Means (Values between 0 and 1)", fontsize=14)
                plt.ylabel("Value", fontsize=12)
                plt.ylim(-0.05, 1.05)  # Enforce the 0-1 boundary visually with a small padding
                plt.xticks(rotation=15)  # Slightly rotate labels if they crowd each other
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # 5. Save or show
                plt.tight_layout()
                plt.savefig("stats_gate.pdf")
                #plt.show()
            except:
                df_stats = pd.DataFrame()
            
            # Print summary per SNR
            print("\n--- Evaluation Results Summary by SNR ---")
            for snr in sorted(target_snrs):
                snr_df = df[df['SNR'] == snr]
                avg_pesq = snr_df['PESQ'].dropna().mean()
                avg_stoi = snr_df['STOI'].dropna().mean()
                avg_sisdr = snr_df['Si-SDR'].dropna().mean()
                avg_rtf = snr_df['RTF'].dropna().mean() if 'RTF' in df.columns else None
                rtf_str = (" | Avg RTF: %.4f" % avg_rtf) if avg_rtf is not None and not pd.isna(avg_rtf) else ""
                print("SNR: %2d dB | Avg PESQ: %.4f | Avg STOI: %.4f | Avg Si-SDR: %.4f%s" % 
                      (snr, avg_pesq if not pd.isna(avg_pesq) else float('nan'), avg_stoi if not pd.isna(avg_stoi) else float('nan'), avg_sisdr if not pd.isna(avg_sisdr) else float('nan'), rtf_str))
            
            return df, df_stats
                      
        else:
            print("Evaluating PESQ & STOI mixed at SNR %.1f dB on %d test samples..." % (args.default_snr, num_eval_samples))
            
            pesq_scores = []
            stoi_scores = []
            sisdr_scores = []
            rtf_scores = []
            for i in tqdm.tqdm(range(num_eval_samples)):
                clean_key = speech_keys[i]
                noise_key = noise_keys[i]
                
                clean_signal = load_hdf5_signal(speech_grp, clean_key, speech_h5.attrs.get("codec", "pcm"))
                noise_signal = load_hdf5_signal(noise_grp, noise_key, noise_h5.attrs.get("codec", "pcm"))
                
                noisy_signal = mix_at_snr(clean_signal, noise_signal, args.default_snr)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                start_time = time.perf_counter()
                denoised_signal = enhance(model, df_state, noisy_signal, pad=True)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                elapsed_time = time.perf_counter() - start_time
                audio_duration = denoised_signal.shape[-1] / sample_rate
                sample_rtf = elapsed_time / audio_duration
                rtf_scores.append(sample_rtf)
                
                # Prepare for metric calculations
                clean_tensor = clean_signal.float().view(1, -1)
                denoised_tensor = denoised_signal.float().view(1, -1)
                
                # Downsample for PESQ
                clean_16k = resample_16k(clean_tensor).squeeze(0).numpy()
                denoised_16k = resample_16k(denoised_tensor).squeeze(0).numpy()
                
                # Downsample for STOI
                clean_10k = resample_10k(clean_tensor).squeeze(0).numpy()
                denoised_10k = resample_10k(denoised_tensor).squeeze(0).numpy()
                
                # Calculate PESQ
                try:
                    score_pesq = pesq(16000, clean_16k, denoised_16k, 'wb')
                    pesq_scores.append(score_pesq)
                except Exception:
                    pass
                    
                # Calculate STOI
                try:
                    score_stoi = pystoi.stoi(clean_10k, denoised_10k, 10000, extended=False)
                    stoi_scores.append(score_stoi)
                except Exception:
                    pass

                # Calculate Si-SDR
                try:
                    score_sisdr = sisdr(16000, clean_16k, denoised_16k)
                    sisdr_scores.append(score_sisdr)
                except Exception:
                    pass
                    
            # Print results
            print("\n--- Evaluation Results Summary ---")
            if pesq_scores:
                print("Average PESQ : %.4f" % np.mean(pesq_scores))
            else:
                print("Average PESQ : N/A (Failed to calculate)")
                
            if stoi_scores:
                print("Average STOI: %.4f" % np.mean(stoi_scores))
            else:
                print("Average STOI: N/A (Failed to calculate)")
                
            if rtf_scores:
                print("Average RTF : %.4f" % np.mean(rtf_scores))
            
            if sisdr_scores:
                print("Average Si-SDR: %.4f" % np.mean(sisdr_scores))
            else:
                print("Average Si-SDR: N/A (Failed to calculate)")

if __name__ == "__main__":
    main()
