#!/usr/bin/env python3

import argparse
import os
import sys
import time
from loguru import logger
import onnxruntime.quantization as oq
from df.enhance import init_df, enhance, df_features
import torchaudio
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


class MultiInputCalibrationDataReader(oq.CalibrationDataReader):
    def __init__(self, calibration_samples):
        # calibration_samples is a list of dicts: 
        # [{"input_1": array, "input_2": array, ...}, ...]
        self.data = iter(calibration_samples)

    def get_next(self):
        # Simply returns the next complete multi-input dictionary
        return next(self.data, None)

def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    quant_type: str = "int8",
    per_channel: bool = False,
    reduce_range: bool = False,
    optimize: bool = True,
    model_dir: str = None,
    audio_path: str = None,
):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading ONNX model from: {input_path}")
    orig_size = get_file_size_mb(input_path)
    logger.info(f"Original model size: {orig_size:.2f} MB")

    # Load and check the model
    import onnx
    try:
        model = onnx.load(input_path)
        onnx.checker.check_model(model)
        logger.info("Original ONNX model validation passed.")
    except Exception as e:
        logger.warning(f"Original ONNX model check failed or skipped: {e}")

    start_time = time.time()

    if quant_type.lower() == "int8":
        try:
            # pyrefly: ignore [missing-import]
            model, df_state, suffix, epoch_num = init_df(
                model_base_dir=model_dir,
                epoch="best",
                config_allow_defaults=True,
                log_file=None
            )

            calibration_samples = []

            # Generate 100-200 representative samples using your existing df_features function
            all_audios = [os.path.join(audio_path, file) for file in os.listdir(audio_path) if file.endswith(".wav")]

            for i in range(100): 
                # Use real or representative audio instead of raw zeros for better accuracy
                dummy_audio, sr = torchaudio.load(all_audios[i])
                dummy_audio = dummy_audio[:,:sr]
                
                spec, erb_feat, spec_feat = df_features(dummy_audio, df_state, getattr(model, "nb_df", 96))
                
                # ONNX expects CPU NumPy arrays, not PyTorch CUDA tensors
                calibration_samples.append({
                    "spec": spec.cpu().numpy(),
                    "feat_erb": erb_feat.cpu().numpy(),
                    "feat_spec": spec_feat.cpu().numpy()  # Include df_state if your ONNX model takes it as input
                })
            # data_reader = MultiInputCalibrationDataReader(calibration_samples)
            # logger.info("Starting INT8 Static Quantization...")
            # oq.quantize_static(
            #     model_input=input_path,
            #     model_output=output_path,
            #     calibration_data_reader=data_reader,
            #     quant_format=oq.QuantFormat.QOperator,  # Or oq.QuantFormat.QDQ depending on target engine
            #     per_channel=per_channel,
            #     reduce_range=reduce_range,
            #     weight_type=oq.QuantType.QInt8,
            #     activation_type=oq.QuantType.QUInt8,
            # )
            
            logger.info("Starting INT8 Dynamic Quantization...")
            oq.quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                per_channel=per_channel,
                reduce_range=reduce_range,
                weight_type=oq.QuantType.QInt8,
            )
            
        except ImportError as e:
            logger.error("Failed to import onnxruntime.quantization. Make sure onnxruntime is installed correctly.")
            logger.error(e)
            sys.exit(1)
        except Exception as e:
            logger.error("Quantization process failed:")
            logger.exception(e)
            sys.exit(1)
            
    elif quant_type.lower() == "fp16":
        try:
            import onnxconverter_common as oqc
            logger.info("Starting Float16 Precision Reduction...")
            model_fp16 = oqc.convert_float_to_float16(
                model, 
                keep_io_types=True, 
                op_block_list=["Cast", "Greater", "Less"],
                check_fp16_ready=False
            )
            onnx.save(model_fp16, output_path)
        except ImportError:
            logger.error(
                "onnxconverter_common is not installed. To use fp16 conversion, "
                "please run: pip install onnxconverter_common"
            )
            sys.exit(1)
        except Exception as e:
            logger.error("FP16 conversion failed:")
            logger.exception(e)
            sys.exit(1)
    else:
        logger.error(f"Unknown quantization type: {quant_type}")
        sys.exit(1)

    duration = time.time() - start_time
    logger.info(f"Quantization finished in {duration:.2f} seconds.")

    if not os.path.isfile(output_path):
        logger.error(f"Output model was not written to: {output_path}")
        sys.exit(1)

    quant_size = get_file_size_mb(output_path)
    reduction = (1.0 - quant_size / orig_size) * 100
    logger.info(f"Quantized model saved to: {output_path}")
    logger.info(f"Quantized model size: {quant_size:.2f} MB")
    logger.info(f"Size reduction: {reduction:.2f}% (Compression ratio: {orig_size / quant_size:.2f}x)")

    # Verify the quantized model can be loaded by ONNX Runtime
    logger.info("Verifying quantized model by loading with ONNX Runtime...")
    try:
        import onnxruntime as ort
        # Use CPUExecutionProvider for standard verification
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        logger.info("Successfully loaded quantized model in ONNX Runtime session!")
        
        # Print basic model input/output info
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        logger.info("Model Inputs:")
        for inp in inputs:
            logger.info(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        logger.info("Model Outputs:")
        for out in outputs:
            logger.info(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
            
    except Exception as e:
        logger.error(f"Failed to load or verify the quantized model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models for DeepFilterNet or other applications."
    )
    parser.add_argument("model_path", type=str, help="Path to the input ONNX model file")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to save the quantized model (optional)"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["int8", "fp16"],
        default="int8",
        help="Quantization type: 'int8' (dynamic int8 quantization, default) or 'fp16' (precision reduction)",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Quantize weights per channel (useful for conv layers)",
    )
    parser.add_argument(
        "--reduce_range",
        action="store_true",
        help="Quantize weights with 7-bits (reduces overflow issues on some x86 platforms)",
    )
    parser.add_argument(
        "--no_optimize",
        action="store_true",
        help="Disable onnxruntime pre-quantization optimization",
    )
    parser.add_argument('--model_dir', type=str, required=True, help="Base directory of the DeepFilterNet model containing config.ini and checkpoint files")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the calibration audio file")
    args = parser.parse_args()

    # Determine default output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.model_path)
        output_path = f"{base}_{args.type}{ext}"
    else:
        output_path = args.output

    quantize_onnx_model(
        input_path=args.model_path,
        output_path=output_path,
        quant_type=args.type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        optimize=not args.no_optimize,
        model_dir=args.model_dir,
        audio_path=args.audio_path,
    )

    print(f"File saved to {output_path}")


if __name__ == "__main__":
    main()
