from numpy import exceptions
import argparse
import json
import os
import random
import signal
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn, optim
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.autograd.grad_mode import set_grad_enabled
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.types import Number

from df.checkpoint import check_patience, load_model, read_cp, write_cp
from df.config import Csv, config
from df.logger import init_logger, log_metrics, log_model_summary
from df.loss import Istft, Loss
from df.lr import cosine_scheduler
from df.model import ModelParams
from df.modules import get_device
from df.utils import (
    as_complex,
    as_real,
    check_finite_module,
    check_manual_seed,
    detach_hidden,
    get_host,
    get_norm_alpha,
    make_np,
)
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader

should_stop = False
debug = False
log_timings = False
state: Optional[DF] = None
istft: Optional[nn.Module]
MAX_NANS = 50


# =============================================================================
# Concept-Aware Training Helpers (for DeepFilterNet8)
# =============================================================================


def load_concept_dataloaders(
    base_dir: str,
    p,
    bs: int,
    bs_eval: int,
    seed: int,
    log_level: str,
    overfit: bool = False,
) -> Optional[Dict[str, DataLoader]]:
    """Load concept_datasets.json and create one DataLoader per noise concept.

    Args:
        base_dir: Model base directory containing concept_datasets.json
        p: ModelParams instance
        bs: Training batch size
        bs_eval: Eval batch size
        seed: Random seed
        log_level: Logging level string
        overfit: Whether to overfit

    Returns:
        Dict mapping concept_name -> DataLoader, or None if no config found.
    """
    concept_file = os.path.join(base_dir, "concept_datasets.json")
    if not os.path.isfile(concept_file):
        return None

    with open(concept_file, "r", encoding="utf-8") as f:
        concept_map = json.load(f)

    if not concept_map:
        return None

    logger.info(f"Loading {len(concept_map)} concept dataloaders from {concept_file}")

    num_workers_total = config("NUM_WORKERS", 4, int, section="train")
    workers_per_loader = max(1, num_workers_total // len(concept_map))

    concept_loaders: Dict[str, DataLoader] = {}
    for concept_name, data_dir in concept_map.items():
        ds_config = os.path.join(data_dir, "dataset.cfg")
        if not os.path.isfile(ds_config):
            logger.warning(f"Skipping concept '{concept_name}': no dataset.cfg in {data_dir}")
            continue
        if not os.path.isdir(data_dir):
            logger.warning(f"Skipping concept '{concept_name}': directory not found: {data_dir}")
            continue

        logger.info(f"  Loading concept '{concept_name}' from {data_dir}")
        loader = DataLoader(
            ds_dir=data_dir,
            ds_config=ds_config,
            sr=p.sr,
            batch_size=bs,
            batch_size_eval=bs_eval,
            num_workers=workers_per_loader,
            pin_memory=get_device().type == "cuda",
            max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
            fft_size=p.fft_size,
            hop_size=p.hop_size,
            nb_erb=p.nb_erb,
            nb_spec=p.nb_df,
            norm_alpha=get_norm_alpha(log=False),
            p_reverb=config("p_reverb", 0.2, float, section="distortion"),
            p_bw_ext=config("p_bandwidth_ext", 0.0, float, section="distortion"),
            p_clipping=config("p_clipping", 0.0, float, section="distortion"),
            p_zeroing=config("p_zeroing", 0.0, float, section="distortion"),
            p_air_absorption=config("p_air_absorption", 0.0, float, section="distortion"),
            p_interfer_sp=config("p_interfer_sp", 0.0, float, section="distortion"),
            prefetch=config("NUM_PREFETCH_BATCHES", 32, int, section="train"),
            overfit=overfit,
            seed=seed,
            min_nb_erb_freqs=p.min_nb_freqs,
            log_timings=False,
            global_sampling_factor=config("GLOBAL_DS_SAMPLING_F", 1.0, float, section="train"),
            snrs=config("DATALOADER_SNRS", [-5, 0, 5, 10, 20, 40], Csv(int), section="train"),  # type: ignore
            gains=config("DATALOADER_GAINS", [-6, 0, 6], Csv(int), section="train"),  # type: ignore
            log_level=log_level,
        )
        concept_loaders[concept_name] = loader

    if not concept_loaders:
        logger.warning("No valid concept dataloaders found in concept_datasets.json")
        return None

    logger.info(
        f"Loaded {len(concept_loaders)} concept dataloaders: {list(concept_loaders.keys())}"
    )
    return concept_loaders


def generate_query(concept_name: str) -> str:
    """Generate a random symbolic query for a given noise concept.

    Query distribution:
        60%  NOT(concept)  — suppress this specific noise type
        20%  DENOISE()     — general denoising (backward-compatible)
        20%  KEEP(voice)   — keep voice, suppress everything else
    """
    r = random.random()
    if r < 0.6:
        return f"NOT({concept_name})"
    elif r < 0.8:
        return "DENOISE()"
    else:
        return "KEEP(voice)"


def get_concept_steps_per_epoch(concept_loaders: Dict[str, DataLoader]) -> int:
    """Total training steps per epoch across all concept dataloaders."""
    return sum(cl.len("train") for cl in concept_loaders.values())


@logger.catch
def main():
    global should_stop, debug, state, log_timings

    parser = argparse.ArgumentParser()
    parser.add_argument("data_config_file", type=str, help="Path to a dataset config file.")
    parser.add_argument(
        "data_dir", type=str, help="Path to the dataset directory containing .hdf5 files."
    )
    parser.add_argument(
        "base_dir", type=str, help="Directory to store logs, summaries, checkpoints, etc."
    )
    parser.add_argument(
        "--host-batchsize-config",
        "-b",
        type=str,
        default=None,
        help="Path to a host specific batch size config.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logger verbosity. Can be one of (trace, debug, info, error, none)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-debug", action="store_false", dest="debug")
    args = parser.parse_args()
    if not os.path.isfile(args.data_config_file):
        raise FileNotFoundError("Dataset config not found at {}".format(args.data_config_file))
    if not os.path.isdir(args.data_dir):
        NotADirectoryError("Data directory not found at {}".format(args.data_dir))
    os.makedirs(args.base_dir, exist_ok=True)
    summary_dir = os.path.join(args.base_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    debug = args.debug
    if args.log_level is not None:
        if debug and args.log_level.lower() != "debug":
            raise ValueError("Either specify debug or a manual log level")
        log_level = args.log_level
    else:
        log_level = "DEBUG" if debug else "INFO"
    init_logger(file=os.path.join(args.base_dir, "train.log"), level=log_level, model=args.base_dir)
    config_file = os.path.join(args.base_dir, "config.ini")
    config.load(config_file)
    seed = config("SEED", 42, int, section="train")
    check_manual_seed(seed)
    logger.info("Running on device {}".format(get_device()))

    # Maybe update batch size
    if args.host_batchsize_config is not None:
        try:
            sys.path.append(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            from scripts.set_batch_size import main as set_batch_size  # type: ignore

            key = get_host() + "_" + config.get("model", section="train")
            key += "_" + config.get("fft_size", section="df")
            set_batch_size(config_file, args.host_batchsize_config, host_key=key)
            config.load(config_file, allow_reload=True)  # Load again
        except Exception as e:
            logger.error(f"Could not apply host specific batch size config: {str(e)}")

    signal.signal(signal.SIGUSR1, get_sigusr1_handler(args.base_dir))

    p = ModelParams()
    state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    mask_only: bool = config("MASK_ONLY", False, bool, section="train")
    train_df_only: bool = config("DF_ONLY", False, bool, section="train")
    jit = config("JIT", False, cast=bool, section="train")
    model, epoch = load_model(
        checkpoint_dir if args.resume else None,
        state,
        jit=False,
        mask_only=mask_only,
        train_df_only=train_df_only,
    )

    bs: int = config("BATCH_SIZE", 1, int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", 0, int, section="train")
    bs_eval = bs_eval if bs_eval > 0 else bs
    overfit = config("OVERFIT", False, bool, section="train")
    log_timings = config("LOG_TIMINGS", False, bool, section="train", save=False)
    dataloader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config_file,
        sr=p.sr,
        batch_size=bs,
        batch_size_eval=bs_eval,
        num_workers=config("NUM_WORKERS", 4, int, section="train"),
        pin_memory=get_device().type == "cuda",
        max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(log=False),
        p_reverb=config("p_reverb", 0.2, float, section="distortion"),
        p_bw_ext=config("p_bandwidth_ext", 0.0, float, section="distortion"),
        p_clipping=config("p_clipping", 0.0, float, section="distortion"),
        p_zeroing=config("p_zeroing", 0.0, float, section="distortion"),
        p_air_absorption=config("p_air_absorption", 0.0, float, section="distortion"),
        p_interfer_sp=config("p_interfer_sp", 0.0, float, section="distortion"),
        prefetch=config("NUM_PREFETCH_BATCHES", 32, int, section="train"),
        overfit=overfit,
        seed=seed,
        min_nb_erb_freqs=p.min_nb_freqs,
        log_timings=log_timings,
        global_sampling_factor=config("GLOBAL_DS_SAMPLING_F", 1.0, float, section="train"),
        snrs=config("DATALOADER_SNRS", [-5, 0, 5, 10, 20, 40], Csv(int), section="train"),  # type: ignore
        gains=config("DATALOADER_GAINS", [-6, 0, 6], Csv(int), section="train"),  # type: ignore
        log_level=log_level,
    )

    # --- Concept-Aware Training (DeepFilterNet8) ---
    concept_loaders = load_concept_dataloaders(
        base_dir=args.base_dir,
        p=p,
        bs=bs,
        bs_eval=bs_eval,
        seed=seed,
        log_level=log_level,
        overfit=overfit,
    )
    if concept_loaders is not None:
        logger.info(
            f"Concept-aware training ENABLED with {len(concept_loaders)} concepts. "
            f"Total training steps/epoch: {get_concept_steps_per_epoch(concept_loaders)}"
        )

    # Batch size scheduling limits the batch size for the first epochs. It will increase the batch
    # size during training as specified. Used format is a comma separated list containing
    # epoch/batch size tuples where each tuple is separated via '/':
    # '<epoch>/<batch_size>,<epoch>/<batch_size>,<epoch>/<batch_size>'
    # The first epoch has to be 0, later epoch may modify the batch size as specified.
    # This only applies to training batch size.
    batch_size_scheduling: List[str] = config("BATCH_SIZE_SCHEDULING", [], Csv(str), section="train")  # type: ignore
    scheduling_bs = bs
    prev_scheduling_bs = bs
    if len(batch_size_scheduling) > 0:
        batch_size_scheduling = [
            (int(bs[0]), int(bs[1])) for bs in (bs.split("/") for bs in batch_size_scheduling)
        ]
        assert batch_size_scheduling[0][0] == 0  # First epoch must be 0
        logger.info("Running with learning rate scheduling")
        
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: %d" % total_params)
    logger.info("Trainable Parameters: %d" % trainable_params)

    max_epochs = config("MAX_EPOCHS", 10, int, section="train")
    assert epoch >= 0
    opt = load_opt(
        checkpoint_dir if args.resume else None,
        model,
        mask_only,
        train_df_only,
    )
    if concept_loaders is not None:
        _steps_per_epoch = get_concept_steps_per_epoch(concept_loaders)
    else:
        _steps_per_epoch = len(dataloader)
    lrs = setup_lrs(_steps_per_epoch)
    wds = setup_wds(_steps_per_epoch)
    if not args.resume and os.path.isfile(os.path.join(checkpoint_dir, ".patience")):
        os.remove(os.path.join(checkpoint_dir, ".patience"))
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")
    if jit:
        # Load as jit after log_model_summary
        model = torch.jit.script(model)

    # Validation optimization target. Used for early stopping and selecting best checkpoint
    val_criteria = []
    val_criteria_type = config("VALIDATION_CRITERIA", "loss", section="train")  # must be in metrics
    val_criteria_rule = config("VALIDATION_CRITERIA_RULE", "min", section="train")
    val_criteria_rule = val_criteria_rule.replace("less", "min").replace("more", "max")
    patience = config("EARLY_STOPPING_PATIENCE", 5, int, section="train")

    losses = setup_losses()

    if config("START_EVAL", False, cast=bool, section="train"):
        val_loss = run_epoch(
            model=model,
            epoch=epoch - 1,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: sum(vals) / len(vals) if len(vals) > 0 else 0.0 for n, vals in losses.get_summaries()}
        )
        log_metrics(f"[{epoch - 1}] [valid]", metrics)
    losses.reset_summaries()
    # Save default values to disk
    config.save(os.path.join(args.base_dir, "config.ini"))
    for epoch in range(epoch, max_epochs):
        if len(batch_size_scheduling) > 0:
            # Get current batch size
            for e, b in batch_size_scheduling:
                if e <= epoch:
                    # Update bs, but don't go higher than the batch size specified in the config
                    scheduling_bs = min(b, bs)
            if prev_scheduling_bs != scheduling_bs:
                logger.info(f"Batch scheduling | Setting batch size to {scheduling_bs}")
                dataloader.set_batch_size(scheduling_bs, "train")
                if concept_loaders is not None:
                    for cl in concept_loaders.values():
                        cl.set_batch_size(scheduling_bs, "train")
                    _steps_per_epoch = get_concept_steps_per_epoch(concept_loaders)
                else:
                    _steps_per_epoch = len(dataloader)
                # Update lr/wd scheduling since dataloader len changed
                lrs = setup_lrs(_steps_per_epoch)
                wds = setup_wds(_steps_per_epoch)
                prev_scheduling_bs = scheduling_bs
        train_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="train",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
            lr_scheduler_values=lrs,
            wd_scheduler_values=wds,
            concept_loaders=concept_loaders,
        )
        metrics = {"loss": train_loss}
        try:
            metrics["lr"] = opt.param_groups[0]["lr"]
        except AttributeError:
            pass
        if debug:
            metrics.update(
                {n: sum(vals) / len(vals) if len(vals) > 0 else 0.0 for n, vals in losses.get_summaries()}
            )
        log_metrics(f"[{epoch}] [train]", metrics)
        write_cp(model, "model", checkpoint_dir, epoch + 1)
        write_cp(opt, "opt", checkpoint_dir, epoch + 1)
        losses.reset_summaries()
        val_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
            concept_loaders=concept_loaders,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: sum(vals) / len(vals) if len(vals) > 0 else 0.0 for n, vals in losses.get_summaries()}
        )
        val_criteria = metrics[val_criteria_type]
        write_cp(
            model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule
        )
        log_metrics(f"[{epoch}] [valid]", metrics)
        if not check_patience(
            checkpoint_dir,
            max_patience=patience,
            new_metric=val_criteria,
            cmp=val_criteria_rule,
            raise_=False,
        ):
            break
        if should_stop:
            logger.info("Stopping training due to timeout")
            exit(0)
        losses.reset_summaries()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    model, epoch = load_model(
        checkpoint_dir,
        state,
        jit=jit,
        epoch="best",
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    test_loss = run_epoch(
        model=model,
        epoch=epoch,
        loader=dataloader,
        split="test",
        opt=opt,
        losses=losses,
        summary_dir=summary_dir,
        concept_loaders=concept_loaders,
    )
    metrics: Dict[str, Number] = {"loss": test_loss}
    metrics.update({n: sum(vals) / len(vals) if len(vals) > 0 else 0.0 for n, vals in losses.get_summaries()})
    log_metrics(f"[{epoch}] [test]", metrics)
    logger.info("Finished training")


def run_epoch(
    model: nn.Module,
    epoch: int,
    loader: DataLoader,
    split: str,
    opt: optim.Optimizer,
    losses: Loss,
    summary_dir: str,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
    concept_loaders: Optional[Dict[str, DataLoader]] = None,
) -> float:
    global debug

    log_freq = config("LOG_FREQ", cast=int, default=100, section="train")
    bs = loader.get_batch_size(split)
    logger.info("Start {} epoch {} with batch size {}".format(split, epoch, bs))

    detect_anomaly: bool = config("DETECT_ANOMALY", False, bool, section="train")
    if detect_anomaly:
        logger.info("Running with autograd profiling")
    dev = get_device()
    l_mem = []
    is_train = split == "train"
    model.train(mode=is_train)
    losses.store_losses = debug or not is_train
    seed = epoch if is_train else 42
    n_nans = 0

    # --- Build batch iterator: concept-aware or standard ---
    use_concepts = concept_loaders is not None and len(concept_loaders) > 0
    if use_concepts:
        concept_names = list(concept_loaders.keys())
        concept_iters: Dict[str, object] = {}
        for name in concept_names:
            concept_iters[name] = iter(concept_loaders[name].iter_epoch(split, seed))
        max_steps = sum(concept_loaders[n].len(split) for n in concept_names) - 1
        start_steps = epoch * (max_steps + 1)
        logger.info(
            f"  Concept-aware mode: {len(concept_names)} concepts, "
            f"{max_steps + 1} total steps"
        )
    else:
        max_steps = loader.len(split) - 1
        start_steps = epoch * loader.len(split)

    def _make_batch_iter():
        """Yields (step_index, batch, query_or_none) tuples."""
        if use_concepts:
            step = 0
            for _round in range(max_steps + 1):
                cname = random.choice(concept_names)
                try:
                    b = next(concept_iters[cname])
                except StopIteration:
                    concept_iters[cname] = iter(
                        concept_loaders[cname].iter_epoch(split, seed + _round)
                    )
                    try:
                        b = next(concept_iters[cname])
                    except StopIteration:
                        continue
                q = generate_query(cname) if is_train else f"NOT({cname})"
                yield step, b, q, cname
                step += 1
        else:
            for idx, b in enumerate(loader.iter_epoch(split, seed)):
                yield idx, b, None, None

    for i, batch, cur_query, cur_concept in _make_batch_iter():
        opt.zero_grad()
        it = start_steps + i  # global training iteration
        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None:
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler_values[it]
        assert batch.feat_spec is not None
        assert batch.feat_erb is not None
        feat_erb = batch.feat_erb.to(dev, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
        noisy = batch.noisy.to(dev, non_blocking=True)
        clean = batch.speech.to(dev, non_blocking=True)
        snrs = batch.snr.to(dev, non_blocking=True)
        with set_detect_anomaly(detect_anomaly and is_train), set_grad_enabled(is_train):
            if not is_train:
                input = as_real(noisy).clone()
            else:
                input = as_real(noisy)
            # Build forward kwargs (pass query only when concept-aware)
            forward_kwargs = dict(spec=input, feat_erb=feat_erb, feat_spec=feat_spec)
            if cur_query is not None:
                forward_kwargs["query"] = cur_query
            enh, m, lsnr, *rest = model.forward(**forward_kwargs)
            # DFNet8 returns (spec, mask, lsnr, df_coefs, concept_loss)
            # DFNet7 returns (spec, mask, lsnr, df_coefs, vq_loss)
            # DFNet4 returns (spec, mask, lsnr, df_coefs, vad)
            # DFNet1/2/3 returns (spec, mask, lsnr, df_coefs)
            vad = None
            vq_loss = None
            if len(rest) > 1:
                # Check if the second extra output is a scalar (vq_loss/concept_loss)
                # or a per-frame tensor (vad)
                extra = rest[1]
                if extra.dim() == 0:
                    # Scalar → VQ/concept loss (DeepFilterNet7/8)
                    vq_loss = extra
                else:
                    # Per-frame tensor → VAD (DeepFilterNet4)
                    vad = extra
            try:
                err = losses.forward(
                    clean, noisy, enh, m, lsnr, snrs=snrs, vad=vad, vq_loss=vq_loss,
                )
            except Exception as e:
                if "nan" in str(e).lower() or "finite" in str(e).lower():
                    logger.warning("NaN in loss computation: {}. Skipping backward.".format(str(e)))
                    check_finite_module(model)
                    n_nans += 1
                    if n_nans > MAX_NANS:
                        raise e
                    continue
                raise e
            if is_train:
                try:
                    err.backward()
                    clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except RuntimeError as e:
                    e_str = str(e)
                    if "nan" in e_str.lower() or "non-finite" in e_str:
                        check_finite_module(model)
                        logger.error(e_str)
                        os.makedirs(os.path.join(summary_dir, "nan"), exist_ok=True)
                        for batch_idx in range(clean.shape[0]):
                            clean_idx = batch.ids[batch_idx].item()
                            summary_write(
                                clean.detach(),
                                noisy.detach(),
                                enh.detach(),
                                batch.snr.detach(),
                                lsnr.detach().float(),
                                os.path.join(summary_dir, "nan"),
                                prefix=split + f"_e{epoch}_i{i}_b{batch_idx}_ds{clean_idx}",
                                idx=batch_idx,
                            )
                        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
                        n_nans += 1
                        if n_nans > MAX_NANS:
                            raise e
                        continue
                    else:
                        raise e
                opt.step()
            detach_hidden(model)
        l_mem.append(err.detach().item())
        if i % log_freq == 0:
            l_mean = sum(l_mem[-100:]) / len(l_mem[-100:]) if len(l_mem[-100:]) > 0 else 0.0
            if np.isnan(l_mean):
                check_finite_module(model)
            l_dict = {"loss": l_mean}
            if lr_scheduler_values is not None:
                l_dict["lr"] = opt.param_groups[0]["lr"]
            if wd_scheduler_values is not None:
                l_dict["wd"] = opt.param_groups[0]["weight_decay"]
            if log_timings:
                l_dict["t_sample"] = batch.timings[:-1].sum()
                l_dict["t_batch"] = batch.timings[-1].mean()  # last is for whole batch
            if cur_concept is not None:
                l_dict["concept"] = cur_concept
                l_dict["query"] = cur_query
            if debug:
                l_dict.update(
                    {
                        n: sum(vals[-bs:]) / len(vals[-bs:]) if len(vals[-bs:]) > 0 else 0.0
                        for n, vals in losses.get_summaries()
                    }
                )
            step = str(i).rjust(len(str(max_steps)))
            log_metrics(f"[{epoch}] [{step}/{max_steps}]", l_dict)
            # logger.info(f"gate_stats: {model.get_gate_stats()}")
            summary_write(
                clean.detach(),
                noisy.detach(),
                enh.detach(),
                batch.snr.detach(),
                lsnr.detach().float(),
                summary_dir,
                prefix=split,
            )
    try:
        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
    except UnboundLocalError as err:
        logger.error(str(err))
    return sum(l_mem) / len(l_mem) if len(l_mem) > 0 else 0.0


def setup_losses() -> Loss:
    global state, istft
    assert state is not None

    p = ModelParams()

    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(
        get_device()
    )
    loss = Loss(state, istft).to(get_device())
    # loss = torch.jit.script(loss)
    return loss


def load_opt(
    cp_dir: Optional[str], model: nn.Module, mask_only: bool = False, df_only: bool = False
) -> optim.Optimizer:
    lr = config("LR", 5e-4, float, section="optim")
    momentum = config("momentum", 0, float, section="optim")  # For sgd, rmsprop
    decay = config("weight_decay", 0.05, float, section="optim")
    optimizer = config("optimizer", "adamw", str, section="optim").lower()
    betas: Tuple[int, int] = config(
        "opt_betas", [0.9, 0.999], Csv(float), section="optim", save=False  # type: ignore
    )
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    supported = {
        "adam": lambda p: optim.Adam(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "adamw": lambda p: optim.AdamW(p, lr=lr, weight_decay=decay, betas=betas, amsgrad=True),
        "sgd": lambda p: optim.SGD(p, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=lr, momentum=momentum, weight_decay=decay),
    }
    if optimizer not in supported:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Must be one of {list(supported.keys())}"
        )
    opt = supported[optimizer](params)
    logger.debug(f"Training with optimizer {opt}")
    if cp_dir is not None:
        try:
            read_cp(opt, "opt", cp_dir, log=False)
        except ValueError as e:
            logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt


def setup_lrs(steps_per_epoch: int) -> np.ndarray:
    lr = config.get("lr", float, "optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("lr_min", 1e-6, float, section="optim")
    lr_warmup = config("lr_warmup", 1e-4, float, section="optim")
    assert lr_warmup < lr
    warmup_epochs = config("warmup_epochs", 3, int, section="optim")
    lr_cycle_mul = config("lr_cycle_mul", 1.0, float, section="optim")
    lr_cycle_decay = config("lr_cycle_decay", 0.5, float, section="optim")
    lr_cycle_epochs = config("lr_cycle_epochs", -1, int, section="optim")
    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_warmup,
        initial_ep_per_cycle=lr_cycle_epochs,
        cycle_decay=lr_cycle_decay,
        cycle_mul=lr_cycle_mul,
    )
    return lr_values


def setup_wds(steps_per_epoch: int) -> Optional[np.ndarray]:
    decay = config("weight_decay", 0.05, float, section="optim")
    decay_end = config("weight_decay_end", -1, float, section="optim")
    if decay_end == -1:
        return None
    if decay == 0.0:
        decay = 1e-12
        logger.warning("Got 'weight_decay_end' value > 0, but weight_decay is disabled.")
        logger.warning(f"Setting initial weight decay to {decay}.")
        config.overwrite("optim", "weight_decay", decay)
    num_epochs = config.get("max_epochs", int, "train")
    decay_values = cosine_scheduler(
        decay, decay_end, niter_per_ep=steps_per_epoch, epochs=num_epochs
    )
    return decay_values


@torch.no_grad()
def summary_write(
    clean: Tensor,
    noisy: Tensor,
    enh: Tensor,
    snrs: Tensor,
    lsnr: Tensor,
    summary_dir: str,
    prefix="train",
    idx: Optional[int] = None,
):
    global state
    assert state is not None

    p = ModelParams()
    bs = snrs.shape[0]
    if idx is None:
        idx = random.randrange(bs)
    snr = snrs[idx].detach().cpu().item()

    def synthesis(x: Tensor) -> Tensor:
        return torch.as_tensor(state.synthesis(make_np(as_complex(x.detach()))))

    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_clean_snr{snr}.wav"), synthesis(clean[idx]), p.sr
    )
    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_noisy_snr{snr}.wav"), synthesis(noisy[idx]), p.sr
    )
    torchaudio.save(
        os.path.join(summary_dir, f"{prefix}_enh_snr{snr}.wav"), synthesis(enh[idx]), p.sr
    )
    np.savetxt(
        os.path.join(summary_dir, f"{prefix}_lsnr_snr{snr}.txt"),
        lsnr[idx].detach().cpu().numpy(),
        fmt="%.3f",
    )


def summary_noop(*__args, **__kwargs):  # type: ignore
    pass


def get_sigusr1_handler(base_dir):
    def h(*__args):  # type: ignore
        global should_stop
        logger.warning("Received timeout signal. Stopping after current epoch")
        should_stop = True
        continue_file = os.path.join(base_dir, "continue")
        logger.warning(f"Writing {continue_file}")
        open(continue_file, "w").close()

    return h


def cleanup(*args):
    import gc

    for arg in args:
        del arg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    from icecream import ic, install

    ic.includeContext = True
    install()
    main()
