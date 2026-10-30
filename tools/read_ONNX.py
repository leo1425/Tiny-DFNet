import os
import sys
import argparse
import numpy as np
import torch
import onnxruntime as ort
from loguru import logger

# Add project root and DeepFilterNet directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
deepfilternet_dir = os.path.join(project_root, "DeepFilterNet")
sys.path.insert(0, project_root)
sys.path.insert(0, deepfilternet_dir)

from df.config import config
from df.enhance import ModelParams, df_features
from df.io import load_audio, save_audio
from df.utils import as_complex
from libdf import DF


def main():
    parser = argparse.ArgumentParser(description="Run DeepFilterNet ONNX model on a noisy audio file.")
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("noisy_path", type=str, help="Path to the noisy input audio file")
    parser.add_argument("output_path", type=str, help="Path to save the enhanced output audio file")
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config.ini (optional)"
    )
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx_path)
    noisy_path = os.path.abspath(args.noisy_path)
    output_path = os.path.abspath(args.output_path)

    # 1. Load config.ini if available
    config_path = args.config
    if config_path is None:
        onnx_dir = os.path.dirname(onnx_path)
        potential_config = os.path.join(onnx_dir, "config.ini")
        if os.path.isfile(potential_config):
            config_path = potential_config

    if config_path and os.path.isfile(config_path):
        logger.info(f"Loading config from {config_path}")
        config.load(config_path, config_must_exist=True, allow_reload=True)
    else:
        logger.warning("No config.ini found or specified. Using default config parameters.")
        config.use_defaults(allow_reload=True)

    # 2. Initialize ModelParams and DFState
    p = ModelParams()
    logger.info(f"Model parameters: SR={p.sr}, FFT_SIZE={p.fft_size}, HOP_SIZE={p.hop_size}, NB_ERB={p.nb_erb}, NB_DF={p.nb_df}")
    
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )

    # 3. Load noisy audio (resampling if necessary)
    logger.info(f"Loading audio from {noisy_path}")
    audio, info = load_audio(noisy_path, sr=p.sr)
    orig_len = audio.shape[-1]
    
    # Pad audio to compensate for algorithmic delay due to STFT/ISTFT
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()
    assert n_fft % hop == 0, f"n_fft ({n_fft}) must be divisible by hop ({hop})"
    
    audio_padded = torch.nn.functional.pad(audio, (0, n_fft))
    
    # 4. Generate features
    logger.info("Computing features (spec, erb_feat, spec_feat)...")
    nb_df = getattr(p, "nb_df", 96)
    spec, erb_feat, spec_feat = df_features(audio_padded, df_state, nb_df, device="cpu")
    
    logger.info(f"Features generated:")
    logger.info(f"  spec shape:      {list(spec.shape)}")
    logger.info(f"  erb_feat shape:  {list(erb_feat.shape)}")
    logger.info(f"  spec_feat shape: {list(spec_feat.shape)}")

    # 5. Load ONNX model and setup session
    logger.info(f"Loading ONNX model from {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Get details of expected inputs and outputs
    onnx_inputs = sess.get_inputs()
    
    # Determine if there is a fixed time dimension in the inputs
    fixed_time_dim = None
    for inp in onnx_inputs:
        shape = inp.shape
        if len(shape) >= 3:
            t_dim = shape[2]
            if isinstance(t_dim, int) and t_dim > 0:
                fixed_time_dim = t_dim
                break
                
    if fixed_time_dim is not None:
        logger.info(f"ONNX model expects a fixed time dimension of {fixed_time_dim} frames. Chunking features accordingly.")
    else:
        logger.info("ONNX model has dynamic time dimension.")

    # Convert torch features to numpy arrays
    spec_np = spec.numpy()
    erb_feat_np = erb_feat.numpy()
    spec_feat_np = spec_feat.numpy()
    
    C = spec_np.shape[0]
    T_feat = spec_np.shape[2]
    
    # Helper to check if a dimension matches or is dynamic
    def dim_matches(onnx_dim, actual_dim):
        return onnx_dim is None or isinstance(onnx_dim, str) or onnx_dim == actual_dim

    # Main inference execution
    if fixed_time_dim is not None:
        chunk_size = fixed_time_dim
        pad_len = (chunk_size - (T_feat % chunk_size)) % chunk_size
        
        if pad_len > 0:
            logger.info(f"Padding time dimension by {pad_len} frames to be a multiple of chunk size {chunk_size}")
            spec_np = np.pad(spec_np, ((0, 0), (0, 0), (0, pad_len), (0, 0), (0, 0)), mode='constant')
            erb_feat_np = np.pad(erb_feat_np, ((0, 0), (0, 0), (0, pad_len), (0, 0)), mode='constant')
            spec_feat_np = np.pad(spec_feat_np, ((0, 0), (0, 0), (0, pad_len), (0, 0), (0, 0)), mode='constant')
            
        T_padded = T_feat + pad_len
        num_chunks = T_padded // chunk_size
        
        channels_enhanced = []
        
        for c in range(C):
            channel_enhanced_chunks = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                
                # Extract chunk for this channel
                spec_chunk = spec_np[c:c+1, :, start:end, ...]
                erb_chunk = erb_feat_np[c:c+1, :, start:end, ...]
                spec_feat_chunk = spec_feat_np[c:c+1, :, start:end, ...]
                
                # Build inputs dictionary for this chunk
                inputs = {}
                for inp in onnx_inputs:
                    name = inp.name
                    shape = inp.shape
                    ndim = len(shape)
                    
                    if name == "spec":
                        inputs[name] = spec_chunk
                        continue
                    elif name in ("feat_erb", "erb_feat"):
                        inputs[name] = erb_chunk
                        continue
                    elif name in ("feat_spec", "spec_feat"):
                        inputs[name] = spec_feat_chunk
                        continue
                        
                    # Dynamic/shape-based fallback
                    if ndim == 4:
                        if dim_matches(shape[3], erb_chunk.shape[3]):
                            inputs[name] = erb_chunk
                    elif ndim == 5:
                        if dim_matches(shape[4], 2):
                            if dim_matches(shape[3], spec_chunk.shape[3]) and not dim_matches(shape[3], spec_feat_chunk.shape[3]):
                                inputs[name] = spec_chunk
                            elif dim_matches(shape[3], spec_feat_chunk.shape[3]) and not dim_matches(shape[3], spec_chunk.shape[3]):
                                inputs[name] = spec_feat_chunk
                            else:
                                if "feat_spec" in name or "spec_feat" in name or "squeeze" in name or "Squeeze" in name:
                                    inputs[name] = spec_feat_chunk
                                else:
                                    inputs[name] = spec_chunk
                                    
                # Execute chunk
                outputs = sess.run(None, inputs)
                
                # Extract enhanced spec chunk
                enhanced_spec_chunk = None
                for out in outputs:
                    if len(out.shape) == 5 and out.shape[4] == 2 and out.shape[3] == spec_chunk.shape[3]:
                        enhanced_spec_chunk = out
                        break
                        
                if enhanced_spec_chunk is None:
                    enhanced_spec_chunk = outputs[0]
                    
                channel_enhanced_chunks.append(enhanced_spec_chunk)
                
            # Reconstruct full spectrogram for this channel
            channel_enhanced = np.concatenate(channel_enhanced_chunks, axis=2)
            # Remove padding
            channel_enhanced = channel_enhanced[:, :, :T_feat, ...]
            channels_enhanced.append(channel_enhanced)
            
        enhanced_spec = np.concatenate(channels_enhanced, axis=0)
        
    else:
        # Full dynamic sequence execution
        inputs = {}
        for inp in onnx_inputs:
            name = inp.name
            shape = inp.shape
            ndim = len(shape)
            
            if name == "spec":
                inputs[name] = spec_np
                continue
            elif name in ("feat_erb", "erb_feat"):
                inputs[name] = erb_feat_np
                continue
            elif name in ("feat_spec", "spec_feat"):
                inputs[name] = spec_feat_np
                continue
                
            if ndim == 4:
                if dim_matches(shape[3], erb_feat_np.shape[3]):
                    inputs[name] = erb_feat_np
            elif ndim == 5:
                if dim_matches(shape[4], 2):
                    if dim_matches(shape[3], spec_np.shape[3]) and not dim_matches(shape[3], spec_feat_np.shape[3]):
                        inputs[name] = spec_np
                    elif dim_matches(shape[3], spec_feat_np.shape[3]) and not dim_matches(shape[3], spec_np.shape[3]):
                        inputs[name] = spec_feat_np
                    else:
                        if "feat_spec" in name or "spec_feat" in name or "squeeze" in name or "Squeeze" in name:
                            inputs[name] = spec_feat_np
                        else:
                            inputs[name] = spec_np

        logger.info("Executing ONNX model on the entire sequence...")
        outputs = sess.run(None, inputs)
        
        enhanced_spec = None
        for out in outputs:
            if len(out.shape) == 5 and out.shape[4] == 2 and out.shape[3] == spec_np.shape[3]:
                enhanced_spec = out
                break
                
        if enhanced_spec is None:
            enhanced_spec = outputs[0]

    logger.info(f"Enhanced spectrogram shape: {enhanced_spec.shape}")
    
    # 8. Synthesize enhanced audio
    enhanced_spec_torch = torch.from_numpy(enhanced_spec)
    enhanced_complex = as_complex(enhanced_spec_torch.squeeze(1))
    
    logger.info("Reconstructing audio via synthesis...")
    enhanced_audio = torch.as_tensor(df_state.synthesis(enhanced_complex.numpy()))
    
    # Slice the padding to compensate for delay
    d = n_fft - hop
    enhanced_audio = enhanced_audio[:, d : orig_len + d]
    
    # 9. Save enhanced audio
    logger.info(f"Saving enhanced audio to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_audio(output_path, enhanced_audio, p.sr, log=True)
    logger.info("Done!")


if __name__ == "__main__":
    main()
