#!/usr/bin/env python3
"""
datasetgenerator.py

Automates the preparation of Voicebank-DEMAND datasets for DeepFilterNet training.
1. Computes noise signals by subtracting clean speech from noisy mixtures: noise = noisy - clean.
2. Saves temporary noise wav files.
3. Splits the training dataset into train (90%) and validation (10%) sets.
4. Generates HDF5 databases for training, validation, and testing using DeepFilterNet's built-in `prepare_data.py`.
5. Creates the dataset.cfg configuration file.
6. Cleans up intermediate temporary files to conserve disk space.
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from typing import List, Tuple
import torch
import torchaudio


def match_audio_files(clean_dir: str, noisy_dir: str) -> List[Tuple[str, str]]:
    """Match clean and noisy wav files based on filenames."""
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    matched_pairs = []
    
    for clean_path in clean_files:
        basename = os.path.basename(clean_path)
        noisy_path = os.path.join(noisy_dir, basename)
        if os.path.exists(noisy_path):
            matched_pairs.append((clean_path, noisy_path))
        else:
            print(f"Warning: No matching noisy file found for {clean_path}")
            
    return matched_pairs


def extract_noise(clean_path: str, noisy_path: str, output_path: str):
    """Load clean and noisy audio, compute noise = noisy - clean, and save to output_path."""
    clean, sr_c = torchaudio.load(clean_path)
    noisy, sr_n = torchaudio.load(noisy_path)
    
    if sr_c != sr_n:
        raise ValueError(f"Sample rates do not match: {sr_c} vs {sr_n} for {clean_path}")
        
    # Match lengths in case of minor frame mismatches
    min_len = min(clean.shape[1], noisy.shape[1])
    noise = noisy[:, :min_len] - clean[:, :min_len]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, noise, sr_c)


def write_txt_list(base_dir: str, file_paths: List[str], output_txt: str):
    """Write list of relative file paths to a text file."""
    with open(output_txt, "w") as f:
        for path in file_paths:
            rel_path = os.path.relpath(path, base_dir)
            f.write(rel_path + "\n")


def run_prepare_data(
    prepare_script: str,
    ds_type: str,
    txt_list: str,
    output_hdf5: str,
    sr: int,
    codec: str,
    num_workers: int
):
    """Run prepare_data.py script via subprocess with python path configured."""
    print(f"Running prepare_data.py for type '{ds_type}' -> {output_hdf5}...")
    cmd = [
        sys.executable,
        prepare_script,
        "--sr", str(sr),
        "--codec", codec,
        "--num_workers", str(num_workers),
        ds_type,
        txt_list,
        output_hdf5
    ]
    env = os.environ.copy()
    # Add DeepFilterNet dir to python path
    df_root = os.path.dirname(os.path.dirname(os.path.abspath(prepare_script)))
    env["PYTHONPATH"] = df_root + os.path.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing prepare_data.py:")
        print(result.stderr)
        result.check_returncode()
    else:
        print(f"Successfully generated {output_hdf5}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Voicebank-DEMAND dataset for DeepFilterNet training.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing clean/noisy train/test folders. Default: 'dataset'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/hdf5",
        help="Directory to save generated HDF5 files and dataset.cfg. Default: 'dataset/hdf5'"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training samples to set aside for validation. Default: 0.1"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Sample rate to resample/store in HDF5. Default: 48000"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm",
        help="Codec for HDF5 storage ('pcm', 'vorbis', or 'flac'). Default: 'pcm'"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for HDF5 generation. Default: 4"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not clean up temporary noise wav files and list txt files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of processed files for dry-run/testing."
    )
    args = parser.parse_args()

    # Find prepare_data.py script relative to this file
    repo_root = os.path.dirname(os.path.abspath(__file__))
    prepare_script = os.path.join(repo_root, "DeepFilterNet", "df", "scripts", "prepare_data.py")
    if not os.path.exists(prepare_script):
        print(f"Error: Could not find prepare_data.py at {prepare_script}")
        sys.exit(1)

    # Check input directories
    clean_train_dir = os.path.join(args.dataset_dir, "clean_trainset_28spk_wav")
    noisy_train_dir = os.path.join(args.dataset_dir, "noisy_trainset_28spk_wav")
    clean_test_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    noisy_test_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")

    for d in [clean_train_dir, noisy_train_dir, clean_test_dir, noisy_test_dir]:
        if not os.path.isdir(d):
            print(f"Error: Required dataset directory not found: {d}")
            sys.exit(1)

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    temp_noise_dir = os.path.join(args.dataset_dir, "temp_noise")
    os.makedirs(temp_noise_dir, exist_ok=True)

    # 1. Match files
    print("Matching train audio files...")
    train_pairs = match_audio_files(clean_train_dir, noisy_train_dir)
    print(f"Found {len(train_pairs)} matching training audio pairs.")

    print("Matching test audio files...")
    test_pairs = match_audio_files(clean_test_dir, noisy_test_dir)
    print(f"Found {len(test_pairs)} matching test audio pairs.")

    if args.limit is not None:
        print(f"Limiting datasets to {args.limit} samples for dry run...")
        train_pairs = train_pairs[:args.limit]
        test_pairs = test_pairs[:args.limit]

    # 2. Extract noise
    print("Extracting noise waveforms from mixtures (noise = noisy - clean)...")
    
    train_noise_paths = []
    for idx, (c_path, n_path) in enumerate(train_pairs):
        out_noise_path = os.path.join(temp_noise_dir, "train", os.path.basename(c_path))
        if idx % 100 == 0:
            print(f"Extracting train noise: {idx}/{len(train_pairs)}...")
        extract_noise(c_path, n_path, out_noise_path)
        train_noise_paths.append(out_noise_path)

    test_noise_paths = []
    for idx, (c_path, n_path) in enumerate(test_pairs):
        out_noise_path = os.path.join(temp_noise_dir, "test", os.path.basename(c_path))
        if idx % 100 == 0:
            print(f"Extracting test noise: {idx}/{len(test_pairs)}...")
        extract_noise(c_path, n_path, out_noise_path)
        test_noise_paths.append(out_noise_path)

    # 3. Create splits
    print("Splitting training data into train/validation splits...")
    # Deterministic shuffle with seed for reproducibility
    random.seed(42)
    indices = list(range(len(train_pairs)))
    random.shuffle(indices)
    
    val_size = int(len(train_pairs) * args.val_ratio)
    val_indices = set(indices[:val_size])
    
    train_clean_split = []
    train_noise_split = []
    val_clean_split = []
    val_noise_split = []
    
    for idx, (c_path, _) in enumerate(train_pairs):
        n_path = train_noise_paths[idx]
        if idx in val_indices:
            val_clean_split.append(c_path)
            val_noise_split.append(n_path)
        else:
            train_clean_split.append(c_path)
            train_noise_split.append(n_path)

    test_clean = [c for c, _ in test_pairs]
    test_noise = test_noise_paths

    print(f"Train split size: {len(train_clean_split)} samples.")
    print(f"Validation split size: {len(val_clean_split)} samples.")
    print(f"Test split size: {len(test_clean)} samples.")

    # 4. Write relative path text lists
    print("Writing temporary text lists for prepare_data.py...")
    # Text list paths
    txt_train_speech = os.path.join(args.dataset_dir, "train_speech.txt")
    txt_train_noise = os.path.join(args.dataset_dir, "train_noise.txt")
    txt_val_speech = os.path.join(args.dataset_dir, "val_speech.txt")
    txt_val_noise = os.path.join(args.dataset_dir, "val_noise.txt")
    txt_test_speech = os.path.join(args.dataset_dir, "test_speech.txt")
    txt_test_noise = os.path.join(args.dataset_dir, "test_noise.txt")

    write_txt_list(args.dataset_dir, train_clean_split, txt_train_speech)
    write_txt_list(args.dataset_dir, train_noise_split, txt_train_noise)
    write_txt_list(args.dataset_dir, val_clean_split, txt_val_speech)
    write_txt_list(args.dataset_dir, val_noise_split, txt_val_noise)
    write_txt_list(args.dataset_dir, test_clean, txt_test_speech)
    write_txt_list(args.dataset_dir, test_noise, txt_test_noise)

    # Output HDF5 paths
    h5_train_speech = os.path.join(args.output_dir, "TRAIN_SET_SPEECH.hdf5")
    h5_train_noise = os.path.join(args.output_dir, "TRAIN_SET_NOISE.hdf5")
    h5_val_speech = os.path.join(args.output_dir, "VALID_SET_SPEECH.hdf5")
    h5_val_noise = os.path.join(args.output_dir, "VALID_SET_NOISE.hdf5")
    h5_test_speech = os.path.join(args.output_dir, "TEST_SET_SPEECH.hdf5")
    h5_test_noise = os.path.join(args.output_dir, "TEST_SET_NOISE.hdf5")

    # 5. Run prepare_data.py commands
    run_prepare_data(prepare_script, "speech", txt_train_speech, h5_train_speech, args.sr, args.codec, args.num_workers)
    run_prepare_data(prepare_script, "noise", txt_train_noise, h5_train_noise, args.sr, args.codec, args.num_workers)
    run_prepare_data(prepare_script, "speech", txt_val_speech, h5_val_speech, args.sr, args.codec, args.num_workers)
    run_prepare_data(prepare_script, "noise", txt_val_noise, h5_val_noise, args.sr, args.codec, args.num_workers)
    run_prepare_data(prepare_script, "speech", txt_test_speech, h5_test_speech, args.sr, args.codec, args.num_workers)
    run_prepare_data(prepare_script, "noise", txt_test_noise, h5_test_noise, args.sr, args.codec, args.num_workers)

    # 6. Generate dataset.cfg configuration file
    cfg_data = {
        "train": [
            ["TRAIN_SET_SPEECH.hdf5", 1.0],
            ["TRAIN_SET_NOISE.hdf5", 1.0]
        ],
        "valid": [
            ["VALID_SET_SPEECH.hdf5", 1.0],
            ["VALID_SET_NOISE.hdf5", 1.0]
        ],
        "test": [
            ["TEST_SET_SPEECH.hdf5", 1.0],
            ["TEST_SET_NOISE.hdf5", 1.0]
        ]
    }
    
    cfg_path = os.path.join(args.output_dir, "dataset.cfg")
    print(f"Writing dataset config to {cfg_path}...")
    with open(cfg_path, "w") as f:
        json.dump(cfg_data, f, indent=4)

    # 7. Cleanup
    if not args.no_cleanup:
        print("Cleaning up temporary list files...")
        for f in [txt_train_speech, txt_train_noise, txt_val_speech, txt_val_noise, txt_test_speech, txt_test_noise]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleaning up temporary noise waveforms...")
        shutil.rmtree(temp_noise_dir, ignore_errors=True)

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
