from email import contentmanager
import scallopy
from df.loss import Loss
from os import stat
from matplotlib import figure
from email import contentmanager
from functools import partial
from typing import Dict, Final, List, Optional, Tuple
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function
import matplotlib.pyplot as plt
import df.multiframe as MF
from df.config import Csv, DfParams, config
from df.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    SqueezedGRU_S,
    erb_fb,
    get_device,
)
from df.utils import as_complex
from libdf import DF
PI = 3.1415926535897932384626433
class ModelParams(DfParams):
    section = "deepfilternet"
    def __init__(self):
        super().__init__()
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.convt_kernel: List[int] = config(
            "CONVT_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.emb_gru_skip_enc: str = config(
            "EMB_GRU_SKIP_ENC", default="none", section=self.section
        )
        self.emb_gru_skip: str = config("EMB_GRU_SKIP", default="none", section=self.section)
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.enc_lin_groups: int = config(
            "ENC_LINEAR_GROUPS", cast=int, default=16, section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)
        self.pf_beta: float = config("PF_BETA", cast=float, default=0.02, section=self.section)
        self.lsnr_dropout: bool = config(
            "LSNR_DROPOUT", cast=bool, default=False, section=self.section
        )
        # ----- Symbolic Gating parameters (new in DeepFilterNet4) -----
        self.symbolic_gate: bool = config(
            "SYMBOLIC_GATE", cast=bool, default=True, section=self.section
        )
        self.symbolic_gate_backend: str = config(
            "SYMBOLIC_GATE_BACKEND", cast=str, default="pytorch", section=self.section
        )
        self.gate_snr_threshold: float = config(
            "GATE_SNR_THRESHOLD", cast=float, default=30.0, section=self.section
        )
        self.gate_vad_threshold: float = config(
            "GATE_VAD_THRESHOLD", cast=float, default=1.0, section=self.section
        )
        self.gate_stationarity_window: int = config(
            "GATE_STATIONARITY_WINDOW", cast=int, default=5, section=self.section
        )
        self.gate_stationarity_threshold: float = config(
            "GATE_STATIONARITY_THRESHOLD", cast=float, default=0.3, section=self.section
        )
        self.gate_temperature_start: float = config(
            "GATE_TEMPERATURE_START", cast=float, default=1.0, section=self.section
        )
        self.gate_temperature_end: float = config(
            "GATE_TEMPERATURE_END", cast=float, default=10.0, section=self.section
        )
        self.gate_temperature_anneal_steps: int = config(
            "GATE_TEMPERATURE_ANNEAL_STEPS", cast=int, default=10000, section=self.section
        )
        self.gate_hard_inference: bool = config(
            "GATE_HARD_INFERENCE", cast=bool, default=True, section=self.section
        )
        # self.vad_gate_threshold: float = config(
        #     "VAD_GATE_THRESHOLD", cast=float, default=0.03, section=self.section
        # )

        # ----- Smoothing parameters (new in DeepFilterNet4) -----
        self.smoothing_method: str = config(
            "SMOOTHING_METHOD", cast=str, default="attack_release", section=self.section
        )
        self.smoothing_window_size: int = config(
            "SMOOTHING_WINDOW_SIZE", cast=int, default=5, section=self.section
        )
        self.smoothing_alpha_attack: float = config(
            "SMOOTHING_ALPHA_ATTACK", cast=float, default=0.9, section=self.section
        )
        self.smoothing_alpha_release: float = config(
            "SMOOTHING_ALPHA_RELEASE", cast=float, default=0.2, section=self.section
        )

        # ----- DurationFilter parameters -----
        self.vad = config("vad", cast=Csv(int), default=(15,15), section="DurationFilter")
        self.noise = config("noise", cast=Csv(int), default=(10,25), section="DurationFilter")
        self.stationarity = config("stationarity", cast=Csv(int), default=(10,25), section="DurationFilter")
        
def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DfNet(erb, erb_inverse, run_df, train_mask)
    return model.to(device=get_device())
class Add(nn.Module):
    def forward(self, a, b):
        return a + b
class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"
        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(
            2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = p.nb_erb
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(
            p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.enc_lin_groups
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = p.emb_num_layers
        if p.emb_gru_skip_enc == "none":
            skip_op = None
        elif p.emb_gru_skip_enc == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif p.emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_out_dim,
                hidden_size=self.emb_out_dim,
                groups=p.lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min
        # Learned VAD head: predicts per-frame voice activity probability from embedding
        self.vad_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        # b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc/2]
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F]
        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)  # [B, T, -1]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        vad = self.vad_fc(emb)  # [B, T, 1] — learned VAD probability
        return e0, e1, e2, e3, emb, c0, lsnr, vad
class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        if p.emb_gru_skip == "none":
            skip_op = None
        elif p.emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif p.emb_gru_skip == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                groups=p.lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=p.emb_num_layers - 1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=p.convt_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid
        )
    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor) -> Tensor:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb, _ = self.emb_gru(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m
class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule
    Requires input of shape B, C, T, F, 2.
    """
    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs
class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim
        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2
        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))
        self.df_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            batch_first=True,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        p.df_gru_skip = p.df_gru_skip.lower()
        assert p.df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: Optional[nn.Module]
        if p.df_gru_skip == "none":
            self.df_skip = None
        elif p.df_gru_skip == "identity":
            assert p.emb_hidden_dim == p.df_hidden_dim, "Dimensions do not match"
            self.df_skip = nn.Identity()
        elif p.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(self.emb_in_dim, self.emb_dim, groups=p.lin_groups)
        else:
            raise NotImplementedError()
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        df_out = GroupedLinearEinsum(self.df_n_hidden, out_dim, groups=p.lin_groups)
        self.df_out = nn.Sequential(df_out, nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())
    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        return c
# =============================================================================
# Neuro-Symbolic Gating Module (new in DeepFilterNet4)
# =============================================================================
class SymbolicGatingModule(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        # Thresholds for symbolic rules
        self.snr_threshold: float = p.gate_snr_threshold
        self.vad_threshold: float = p.gate_vad_threshold
        self.stationarity_window: int = p.gate_stationarity_window
        self.stationarity_threshold: float = p.gate_stationarity_threshold

        # Temperature annealing schedule
        self.temperature_start: float = p.gate_temperature_start
        self.temperature_end: float = p.gate_temperature_end
        self.temperature_anneal_steps: int = p.gate_temperature_anneal_steps

        # Inference behavior
        self.hard_inference: bool = p.gate_hard_inference

        # Step counter for temperature annealing (not a trainable parameter)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        #Smoothing
        self.smoothing_method: str = p.smoothing_method
        self.smoothing_window_size: int = p.smoothing_window_size
        self.smoothing_alpha_attack: float = p.smoothing_alpha_attack
        self.smoothing_alpha_release: float = p.smoothing_alpha_release
        self.vad = p.vad
        self.noise = p.noise
        self.stationarity = p.stationarity
    @property
    def temperature(self) -> float:
        """Current temperature based on linear annealing schedule.
        Linearly interpolates from temperature_start to temperature_end
        over temperature_anneal_steps training steps.
        """
        if self.temperature_anneal_steps <= 0:
            return self.temperature_end
        progress = min(self._step.item() / self.temperature_anneal_steps, 1.0)
        return self.temperature_start + progress * (self.temperature_end - self.temperature_start)
    def estimate_vad(self, feat_erb: Tensor) -> Tuple[Tensor, Tensor]:
        """Estimate Voice Activity Detection from ERB features.
        Computes frame-level energy and compares against an adaptive noise floor
        estimated as the 10th percentile of frame energies per utterance.
        Args:
            feat_erb: [B, 1, T, E] ERB features (dB scale, normalized)
        Returns:
            energy:    [B, T, 1] mean frame energy across ERB bands
            vad_score: [B, T, 1] energy above noise floor (higher = speech likely)
        """
        x = feat_erb.squeeze(1)  # [B, T, E]
        energy = x.mean(dim=-1, keepdim=True)  # [B, T, 1]
        # Noise floor estimate: 10th percentile of frame energies per utterance
        noise_floor = torch.quantile(
            energy, 0.1, dim=1, keepdim=True
        )  # [B, 1, 1]
        vad_score = energy - noise_floor  # [B, T, 1]
        return energy, vad_score

    def estimate_stationarity(self, feat_spec: Tensor) -> Tensor:
        """
        Autocorrélation normalisée sur la magnitude du spectre
        Args:
            feat_spec: [B, 1, T, F', 2] spectre complexe
        Returns:
            periodicity_score: [B, T, 1] entre 0 (bruit) et 1 (stationnaire/voix)
        """
        self.min_lag = 2
        self.max_lag = 16

        x = feat_spec.squeeze(1)
        mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)

        mag_mean = mag.mean(dim=-1, keepdim=True)
        mag_centered = mag - mag_mean


        energy = (mag_centered**2).sum(dim=-1, keepdim=True) + 1e-6

        # Autocorrélation en fct du temps
        corrs = []
        for lag in range(self.min_lag, self.max_lag + 1):
            # Décalage temporel
            mag_lag = F.pad(mag_centered[:, :-lag], (0, 0, lag, 0))

            # Corrélation croisée normalisée
            r = (mag_centered * mag_lag).sum(dim=-1, keepdim=True) / energy
            corrs.append(r)

        # [B, T, L]
        corrs = torch.cat(corrs, dim=-1)

        # Calcul de l'autocorrélation maximale pour chaque frame
        periodicity_score, _ = corrs.max(dim=-1, keepdim=True)

        return torch.clamp(periodicity_score, 0.0, 1.0)

    def smooth_gating_signal(
        self,
        gate: Tensor,
    ) -> Tensor:

        orig_shape = gate.shape

        if len(orig_shape) == 4:
            # [B, 1, T, 1] -> [B, 1, T]
            x = gate.squeeze(3)
        elif len(orig_shape) == 3:
            # [B, T, 1] -> [B, 1, T]
            x = gate.transpose(1, 2)
        else:
            x = gate
            
        B, C, T = x.shape

    
        if self.smoothing_method == "attack_release":

            smoothed = torch.zeros_like(x)
            prev_state = x[:, :, 0] # Initialize with first frame
            smoothed[:, :, 0] = prev_state
            
            for t in range(1, T):
                current = x[:, :, t]

                is_attack = current > prev_state
                alpha = torch.where(is_attack, self.smoothing_alpha_attack, self.smoothing_alpha_release)

                prev_state = alpha * current + (1.0 - alpha) * prev_state
                smoothed[:, :, t] = prev_state
                
            x_smooth = smoothed

        elif self.smoothing_method == "hann":

            if window_size < 3:
                return gate

            if window_size % 2 == 0:
                window_size += 1

            window = torch.hann_window(window_size, periodic=False, device=gate.device, dtype=gate.dtype)
            kernel = (window / window.sum()).view(1, 1, -1) # [1, 1, W]

            pad_amt = window_size // 2
            x_padded = F.pad(x, (pad_amt, pad_amt), mode="replicate")

            x_smooth = F.conv1d(x_padded, kernel)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

        # Restore original tensor shape
        if len(orig_shape) == 4:
            return x_smooth.unsqueeze(3)
        elif len(orig_shape) == 3:
            return x_smooth.transpose(1, 2)
        return x_smooth

    # Remove short silences/extend one duration
    def apply_dual_duration_filter(self,
    gate: Tensor, 
    min_zero_gap: int, 
    min_one_duration: int
    ) -> Tensor:

        # [B, T, 1] -> [B, 1, T]
        x = gate.transpose(1, 2)

        if min_zero_gap > 1:
            p_max = (min_zero_gap // 2, min_zero_gap // 2 - (1 if min_zero_gap % 2 == 0 else 0))
            x = F.pad(x, p_max, mode="constant", value=0.0)
            x = F.max_pool1d(x, kernel_size=min_zero_gap, stride=1)
            p_min = p_max
            x = F.pad(x, p_min, mode="constant", value=1.0)
            x = -F.max_pool1d(-x, kernel_size=min_zero_gap, stride=1)

        if min_one_duration > 1:
            p_right = min_one_duration - 1
            p_left = 0
            x = F.pad(x, (p_left, p_right), mode="constant", value=0.0)
            x = F.max_pool1d(x, kernel_size=min_one_duration, stride=1)

        return x.transpose(1, 2)


    def forward(
        self, lsnr: Tensor, feat_erb: Tensor, feat_spec: Tensor,  vad: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:


        # Get features
        energy = feat_erb.squeeze(1).mean(dim=-1, keepdim=True) 
        stationarity_score = self.estimate_stationarity(feat_spec) # higher = more non-stationary

        # Apply rules + filters
        speech_gate = (vad > self.vad_threshold).float() #0 pas de traitement (pas de parole), 1 traitement possible (car parole)
        speech_gate = self.apply_dual_duration_filter(speech_gate, self.vad[0], self.vad[1])
        noise_gate = (lsnr < self.snr_threshold).float()  # 0 mean high SNR (pas de bruit), 1 mean low SNR (bcp de bruit)
        noise_gate = self.apply_dual_duration_filter(noise_gate, self.noise[0], self.noise[1])*speech_gate
        stationarity_gate = (stationarity_score > self.stationarity_threshold).float()*speech_gate  # 0 mean high SNR (pas de bruit), 1 mean low SNR (bcp de bruit)
        stationarity_gate = self.apply_dual_duration_filter(stationarity_gate, self.stationarity[0], self.stationarity[1])

        return noise_gate, stationarity_gate, speech_gate


# Same as SymbolicGatingModule but with Scallop
class SymbolicGatingModuleScallop(nn.Module):

    def __init__(self):
        super().__init__()
        p = ModelParams()

        self.lookahead: int = p.conv_lookahead

        # ---- Thresholds ----
        self.snr_threshold: float = p.gate_snr_threshold
        self.vad_threshold: float = p.gate_vad_threshold
        self.stationarity_threshold: float = p.gate_stationarity_threshold
        self.stationarity_window: int = p.gate_stationarity_window

        # ---- Smoothing ----
        self.smoothing_method: str = p.smoothing_method
        self.smoothing_window_size: int = p.smoothing_window_size
        self.smoothing_alpha_attack: float = p.smoothing_alpha_attack
        self.smoothing_alpha_release: float = p.smoothing_alpha_release

        # ---- Duration filter parameters ----
        self.vad_dur = p.vad          # (min_zero_gap, min_one_duration)
        self.noise_dur = p.noise
        self.stationarity_dur = p.stationarity

        # ---- Build Scallop context ----
        self._build_scallop_context()


    def _build_scallop_context(self):
        ctx = scallopy.ScallopContext(provenance="topkproofs", k=1)

        # relations
        ctx.add_relation("vad", (int, int, float))        # (B, T, value)
        ctx.add_relation("lsnr", (int, int, float))       # (B, T, value)
        ctx.add_relation("stat", (int, int, float))       # (B, T, value)
        ctx.add_relation("time_step", (int, int))          # (B, T)

        # rules
        ctx.add_rule(
            f"raw_speech(b, t) :- vad(b, t, v), v > {self.vad_threshold}"
        )
        ctx.add_rule(
            f"raw_noise(b, t) :- lsnr(b, t, s), s < {self.snr_threshold}"
        )
        ctx.add_rule(
            f"raw_stat(b, t) :- stat(b, t, st), st > {self.stationarity_threshold}"
        )

        ctx.add_rule(
            f"speech_gate(b, t) :- raw_speech(b,t)"
        )
        

        ctx.add_rule("erb_gate(b, t) :- speech_gate(b, t), raw_noise(b, t)")
        ctx.add_rule("df_gate(b, t) :- erb_gate(b, t), raw_stat(b, t)")

        self._ctx = ctx

    # same as SymbolicGatingModule
    def estimate_stationarity(self, feat_spec: Tensor) -> Tensor:
        """
        Autocorrélation normalisée sur la magnitude du spectre
        Args:
            feat_spec: [B, 1, T, F', 2] spectre complexe
        Returns:
            periodicity_score: [B, T, 1] entre 0 (bruit) et 1 (stationnaire/voix)
        """
        self.min_lag = 2
        self.max_lag = 16

        x = feat_spec.squeeze(1)
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-8)

        mag_mean = mag.mean(dim=-1, keepdim=True)
        mag_centered = mag - mag_mean

        energy = (mag_centered ** 2).sum(dim=-1, keepdim=True) + 1e-6

        # Autocorrélation en fct du temps
        corrs = []
        for lag in range(self.min_lag, self.max_lag + 1):
            # Décalage temporel
            mag_lag = F.pad(mag_centered[:, :-lag], (0, 0, lag, 0))

            # Corrélation croisée normalisée
            r = (mag_centered * mag_lag).sum(dim=-1, keepdim=True) / energy
            corrs.append(r)

        # [B, T, L]
        corrs = torch.cat(corrs, dim=-1)

        # Calcul de l'autocorrélation maximale pour chaque frame
        periodicity_score, _ = corrs.max(dim=-1, keepdim=True)

        return torch.clamp(periodicity_score, 0.0, 1.0)

    # same as SymbolicGatingModule
    def smooth_gating_signal(
            self,
            gate: Tensor,
    ) -> Tensor:

        orig_shape = gate.shape

        if len(orig_shape) == 4:
            # [B, 1, T, 1] -> [B, 1, T]
            x = gate.squeeze(3)
        elif len(orig_shape) == 3:
            # [B, T, 1] -> [B, 1, T]
            x = gate.transpose(1, 2)
        else:
            x = gate

        B, C, T = x.shape

        if self.smoothing_method == "attack_release":

            smoothed = torch.zeros_like(x)
            prev_state = x[:, :, 0]  # Initialize with first frame
            smoothed[:, :, 0] = prev_state

            for t in range(1, T):
                current = x[:, :, t]

                is_attack = current > prev_state
                alpha = torch.where(is_attack, self.smoothing_alpha_attack, self.smoothing_alpha_release)

                prev_state = alpha * current + (1.0 - alpha) * prev_state
                smoothed[:, :, t] = prev_state

            x_smooth = smoothed

        elif self.smoothing_method == "hann":

            if window_size < 3:
                return gate

            if window_size % 2 == 0:
                window_size += 1

            window = torch.hann_window(window_size, periodic=False, device=gate.device, dtype=gate.dtype)
            kernel = (window / window.sum()).view(1, 1, -1)  # [1, 1, W]

            pad_amt = window_size // 2
            x_padded = F.pad(x, (pad_amt, pad_amt), mode="replicate")

            x_smooth = F.conv1d(x_padded, kernel)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

        # Restore original tensor shape
        if len(orig_shape) == 4:
            return x_smooth.unsqueeze(3)
        elif len(orig_shape) == 3:
            return x_smooth.transpose(1, 2)
        return x_smooth

    def apply_dual_duration_filter(
    self,
    gate: Tensor, 
    min_zero_gap: int, 
    min_one_duration: int
    ) -> Tensor:
        #  [B, T, 1] -> [B, T]
        x = gate.squeeze(-1)  # [B, T]
        B, T = x.shape
        if T == 0:
            return gate

        p_left = min_one_duration - 1
        p_right = 0
        
        padded = F.pad(x, (p_left, p_right), mode="constant", value=0.0)
        gate_out = F.max_pool1d(padded, kernel_size=min_one_duration, stride=1)  # [B, 1, T]

        # 3. Reshape final [B, T, 1]
        return gate_out.squeeze(1).unsqueeze(-1)

    def forward(
        self, lsnr: Tensor, feat_erb: Tensor, feat_spec: Tensor, vad: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:

        energy = feat_erb.squeeze(1).mean(dim=-1, keepdim=True)  # [B, T, 1]
        stationarity_score = self.estimate_stationarity(feat_spec)  # [B, T, 1]

        # Pre-process VAD signal
        p_speech = vad * energy

        B, T, _ = lsnr.shape
        device = lsnr.device

        vad_squeezed = vad.squeeze(-1)            # [B, T]
        lsnr_squeezed = lsnr.squeeze(-1)          # [B, T]
        stat_squeezed = stationarity_score.squeeze(-1)  # [B, T]

        time_step_facts = [
            (None, (int(b), int(t))) for b in range(B) for t in range(T)
        ]
        vad_facts = [
            (None, (int(b), int(t), float(vad_squeezed[b, t].item())))
            for b in range(B) for t in range(T)
        ]
        lsnr_facts = [
            (None, (int(b), int(t), float(lsnr_squeezed[b, t].item())))
            for b in range(B) for t in range(T)
        ]
        stat_facts = [
            (None, (int(b), int(t), float(stat_squeezed[b, t].item())))
            for b in range(B) for t in range(T)
        ]

        # --- Scallop reasoning ---
        ctx = self._ctx.clone()
        ctx.add_facts("time_step", time_step_facts)
        ctx.add_facts("vad", vad_facts)
        ctx.add_facts("lsnr", lsnr_facts)
        ctx.add_facts("stat", stat_facts)
        ctx.run()

        # --- Reconstruct gate tensors from Scallop output ---
        speech_gate = torch.zeros((B, T, 1), device=device, dtype=lsnr.dtype)
        erb_gate = torch.zeros((B, T, 1), device=device, dtype=lsnr.dtype)
        df_gate = torch.zeros((B, T, 1), device=device, dtype=lsnr.dtype)

        for _tag, (b, t) in ctx.relation("speech_gate"):
            speech_gate[int(b), int(t), 0] = 1.0
        for _tag, (b, t) in ctx.relation("erb_gate"):
            erb_gate[int(b), int(t), 0] = 1.0
        for _tag, (b, t) in ctx.relation("df_gate"):
            df_gate[int(b), int(t), 0] = 1.0

        # --- Apply duration_filtre ---
        speech_gate = self.apply_dual_duration_filter(speech_gate, self.vad_dur[0], self.vad_dur[1])
        erb_gate = self.apply_dual_duration_filter(erb_gate, self.noise_dur[0], self.noise_dur[1])
        df_gate = self.apply_dual_duration_filter(df_gate, self.stationarity_dur[0], self.stationarity_dur[1])

        # --- Export to csv ---
        # with open("df4_gated_small_gates.csv", "w") as f:
        #     f.write("batch,time,vad,lsnr,stat,speech_gate,erb_gate,df_gate\n")
        #     for b in range(B):
        #         for t in range(T):
        #             f.write("{},{},{},{},{},{},{},{}\n".format(
        #                 b, t, vad_squeezed[b, t].item(), lsnr_squeezed[b, t].item(),
        #                 stat_squeezed[b, t].item(), speech_gate[b, t, 0].item(),
        #                 erb_gate[b, t, 0].item(), df_gate[b, t, 0].item()
        #             ))
            



        return erb_gate, df_gate, speech_gate

# =============================================================================
# Main Network
# =============================================================================
class DfNet(nn.Module):
    run_df: Final[bool]
    run_erb: Final[bool]
    lsnr_droput: Final[bool]
    post_filter: Final[bool]
    post_filter_beta: Final[float]
    symbolic_gate_enabled: Final[bool]
    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.df_lookahead = p.df_lookahead
        self.nb_df = p.nb_df
        self.freq_bins: int = p.fft_size // 2 + 1
        self.emb_dim: int = layer_width * p.nb_erb
        self.erb_bins: int = p.nb_erb
        if p.conv_lookahead > 0:
            assert p.conv_lookahead >= p.df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        if p.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        # NOTE: Unlike DeepFilterNet3, no parameter freezing in v4.
        # All encoder and decoder parameters are trainable.
        self.mask = Mask(erb_inv_fb)
        self.erb_inv_fb = erb_inv_fb
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta
        self.df_order = p.df_order
        self.df_op = MF.DF(num_freqs=p.nb_df, frame_size=p.df_order, lookahead=self.df_lookahead)
        self.df_dec = DfDecoder()
        self.df_out_transform = DfOutputReshapeMF(self.df_order, p.nb_df)
        self.run_erb = p.nb_df + 1 < self.freq_bins
        if not self.run_erb:
            logger.warning("Running without ERB stage")
        self.run_df = run_df
        if not run_df:
            logger.warning("Running without DF stage")
        self.train_mask = train_mask
        self.lsnr_droput = p.lsnr_dropout
        assert p.df_n_iter == 1
        # --- Symbolic Gating (new in DeepFilterNet4) ---
        self.symbolic_gate_enabled = p.symbolic_gate
        if self.symbolic_gate_enabled:
            backend = p.symbolic_gate_backend.lower()
            if backend == "scallop":
                self.symbolic_gate_module = SymbolicGatingModuleScallop()
                logger.info("Neuro-Symbolic Gating Module enabled (DeepFilterNet4) — Scallop backend")
            else:
                self.symbolic_gate_module = SymbolicGatingModule()
                logger.info("Neuro-Symbolic Gating Module enabled (DeepFilterNet4) — PyTorch backend")
            logger.info(
                f"  SNR threshold: {p.gate_snr_threshold:.1f} dB | "
                f"VAD threshold: {p.gate_vad_threshold:.1f} | "
                f"Stationarity window: {p.gate_stationarity_window} frames"
            )
            logger.info(
                f"  Temperature annealing: {p.gate_temperature_start:.1f} → "
                f"{p.gate_temperature_end:.1f} over {p.gate_temperature_anneal_steps} steps"
            )
            logger.info(f"  Hard inference gating: {p.gate_hard_inference}")
        else:
            self.symbolic_gate_module = None
        # Gate statistics storage (updated each forward pass, accessible via get_gate_stats())
        self._gate_stats: Dict[str, float] = {}
    def get_gate_stats(self) -> Dict[str, float]:
        """Return the symbolic gate statistics from the last forward pass.
        Useful for logging during training. Returns an empty dict if symbolic
        gating is disabled or no forward pass has been executed yet.
        Returns:
            Dict with keys:
                - gate_erb_mean:  mean ERB gate activation value
                - gate_df_mean:   mean DF gate activation value
                - bypass_rate:    fraction of frames in full bypass mode
                - erb_only_rate:  fraction of frames using ERB stage only
                - full_rate:      fraction of frames using full processing
                - temperature:    current gating temperature
        """
        return self._gate_stats
    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # Not used, take spec modified by mask instead
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward method of DeepFilterNet4 (Neuro-Symbolic).
        Args:
            spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
            feat_erb (Tensor): ERB features of shape [B, 1, T, E]
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]
        Returns:
            spec_e (Tensor): Enhanced spectrum of shape [B, 1, T, F, 2]
            m (Tensor): Raw ERB mask estimate of shape [B, 1, T, E] (before gating)
            lsnr (Tensor): Local SNR estimate of shape [B, T, 1]
            df_coefs (Tensor): DF coefficients (before gating)
            vad (Tensor): Learned VAD probability of shape [B, T, 1]
        """
        with record_function("PreProc"):
            feat_spec_orig = feat_spec  # Preserve original for gating [B, 1, T, F', 2]
            feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)
            # Preserve original feat_erb for symbolic gating (before causal padding)
            feat_erb_orig = feat_erb
            feat_erb = self.pad_feat(feat_erb)
            feat_spec = self.pad_feat(feat_spec)
        with record_function("Encoder"):
            e0, e1, e2, e3, emb, c0, lsnr, vad = self.enc(feat_erb, feat_spec)
        # --- Symbolic Gating Decision ---
        with record_function("Gate process"):
            if self.symbolic_gate_enabled and self.symbolic_gate_module is not None:
                gate_erb, gate_df, gate_vad = self.symbolic_gate_module(lsnr, feat_erb_orig, feat_spec_orig, vad=vad)
                # Compute and store gate statistics for logging
                with torch.no_grad():
                    ge = gate_erb.detach()
                    gd = gate_df.detach()
                    gv = gate_vad.detach()
                    self._gate_stats = {
                        "gate_erb_mean": ge.mean().item(),
                        "gate_df_mean": gd.mean().item(),
                        "gate_vad_mean": gv.mean().item(),
                    }
            else:
                gate_erb = None
                gate_df = None
                gate_vad = None
        with record_function("LSRN Dropout"):
            if self.lsnr_droput:
                idcs = lsnr.squeeze() > -10.0
                b, t = (spec.shape[0], spec.shape[2])
                m = torch.zeros((b, 1, t, self.erb_bins), device=spec.device)
                df_coefs = torch.zeros((b, t, self.nb_df, self.df_order * 2))
                spec_m = spec.clone()
                emb = emb[:, idcs]
                e0 = e0[:, :, idcs]
                e1 = e1[:, :, idcs]
                e2 = e2[:, :, idcs]
                e3 = e3[:, :, idcs]
                c0 = c0[:, :, idcs]
        # --- Stage 1: ERB Masking ---
        if self.run_erb:
            with record_function("ERB decoder"):
                if self.lsnr_droput:
                    m[:, :, idcs] = self.erb_dec(emb, e3, e2, e1, e0)
                else:
                    m = self.erb_dec(emb, e3, e2, e1, e0)
            # Apply symbolic gate to ERB mask:
            #   gate_erb=1 → use predicted mask m (normal denoising)
            #   gate_erb=0 → use identity mask (ones, bypass)
            with record_function("ERB decoder gate"):
                if gate_erb is not None and not self.lsnr_droput:
                    gate_erb_4d = gate_erb.unsqueeze(1)  # [B, 1, T, 1]
                    m_gated = m * gate_erb_4d + (1.0 - gate_erb_4d)  # interpolate to identity
                    spec_m = self.mask(spec, m_gated) #* gate_vad.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 481, 2)
                    #print(f"spec_m shape: {spec_m.shape}, gate_vad shape: {gate_vad.unsqueeze(1).shape}")
                else:
                    spec_m = self.mask(spec, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)
        # --- Stage 2: Deep Filtering ---
        if self.run_df:
            with record_function("DF Decoder"):
                if self.lsnr_droput:
                    df_coefs[:, idcs] = self.df_dec(emb, c0)
                else:
                    df_coefs = self.df_dec(emb, c0)
            with record_function("DF decoder gate"):
                df_coefs = self.df_out_transform(df_coefs)
                spec_df = self.df_op(spec.clone(), df_coefs)
                # Apply symbolic gate to DF output:
                #   gate_df=1 → use DF-filtered spectrum for low-freq bins
                #   gate_df=0 → use ERB-masked spectrum for low-freq bins (ERB only)
                if gate_df is not None and not self.lsnr_droput:
                    gate_df_5d = gate_df.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1, 1]
                    spec_e = spec_m.clone()
                    spec_e[..., : self.nb_df, :] = (
                        gate_df_5d * spec_df[..., : self.nb_df, :]
                        + (1.0 - gate_df_5d) * spec_m[..., : self.nb_df, :]
                    )
                    # High-freq bins already come from spec_m via the clone
                else:
                    spec_e = spec_df
                    spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m
        # ---Gated Depp Filter ---
        if gate_erb is not None and not self.lsnr_droput:
            spec_e = spec_e * gate_vad.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 481, 2)
        # --- Post-filter (optional, from DeepFilterNet3) ---
        with record_function("Post-filter"):
            if self.post_filter:
                beta = self.post_filter_beta
                eps = 1e-12
                spec_e_abs = torch.norm(spec_e, dim=-1)
                spec_abs = torch.norm(spec, dim=-1)
                mask = (spec_e_abs / spec_abs.add(eps)).clamp(eps, 1)
                mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
                pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
                spec_e = spec_e * pf.unsqueeze(-1)
        return spec_e, m, lsnr, df_coefs, vad