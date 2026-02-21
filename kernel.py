import torch
import numpy as np
import torchaudio.transforms as T
import torch.nn.functional as F

from torch import nn
from typing import Any, Dict, List
from itertools import product


class WaveformAugment(nn.Module):
    """Waveform-level data augmentation for keyword spotting.

    Applies simple, on-the-fly perturbations directly to raw audio waveforms (shape: [B, T]).
    The goal is to improve robustness to small temporal misalignments, recording gain changes,
    and background noise.

    Pipeline (when enabled):
    1) Random time shift up to ±max_shift_ms (zero-padded).
    2) Random gain scaling in [gain_min, gain_max] with clamping to [-1, 1].
    3) Optional additive white noise with SNR sampled uniformly in [snr_db_min, snr_db_max]
       (applied with probability 0.6), then clamped to [-1, 1].

    The whole augmentation block is applied with probability `p`. If not applied, the waveform
    is returned unchanged.

    Args:
        sr: Sampling rate in Hz.
        p: Probability of applying the augmentation pipeline.
        max_shift_ms: Maximum absolute time shift in milliseconds.
        snr_db_min: Minimum SNR (dB) for noise injection.
        snr_db_max: Maximum SNR (dB) for noise injection.
        gain_min: Minimum multiplicative gain.
        gain_max: Maximum multiplicative gain.

    Notes:
        - Assumes waveform amplitude roughly normalized in [-1, 1].
        - Time shift uses zero padding rather than wrap-around.
    """
    def __init__(self, sr=16000, p=0.5, max_shift_ms=80, snr_db_min=15, snr_db_max=30, gain_min=0.8, gain_max=1.2):
        super().__init__()
        self.sr = sr
        self.p = float(p)
        self.max_shift = int(sr * (max_shift_ms / 1000.0))
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

    @staticmethod
    def _clamp(x): 
        return torch.clamp(x, -1.0, 1.0)

    def _time_shift(self, w):
        if self.max_shift <= 0: 
            return w
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        if shift == 0: 
            return w
        if shift > 0:
            pad = torch.zeros((w.shape[0], shift), dtype=w.dtype, device=w.device)
            return torch.cat([w[:, shift:], pad], dim=1)
        else:
            shift = -shift
            pad = torch.zeros((w.shape[0], shift), dtype=w.dtype, device=w.device)
            return torch.cat([pad, w[:, :-shift]], dim=1)

    def _gain(self, w):
        g = float(torch.empty(1).uniform_(self.gain_min, self.gain_max).item())
        return self._clamp(w * g)

    def _noise_snr(self, w):
        snr_db = float(torch.empty(1).uniform_(self.snr_db_min, self.snr_db_max).item())
        rms = torch.sqrt(torch.mean(w**2) + 1e-8)
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        return self._clamp(w + torch.randn_like(w) * noise_rms)

    def forward(self, w):
        if float(torch.rand(1).item()) > self.p:
            return w
        # shift -> gain -> noise
        w = self._time_shift(w)
        w = self._gain(w)
        if float(torch.rand(1).item()) < 0.6:
            w = self._noise_snr(w)
        return w
    
    
class Functions(object):
    """Collection of small utility functions used across training/evaluation.

    This class groups together stateless helpers used by the keyword-spotting pipeline:

    - Feature extraction: `build_mfcc(cfg)` builds a torchaudio MFCC transform from a config dict.
    - Evaluation: `eval_acc(model, loader)` and `eval_loss(model, loader, loss_fn)` compute
      dataset-level accuracy and average loss.
    - Hyperparameter search: `build_grid(search_space)` expands a dict of parameter lists into
      a Cartesian-product grid (list of dicts).
    - Confidence-based metric: `pass_rate(model, loader, thr)` computes the fraction of samples
      that are both correctly classified and have top-1 probability > `thr`.
    - Numpy softmax: `softmax_np(x)`.
    - ONNX evaluation: `pass_rate_onnx(test_ds, ort_frontend, ort_model, thr)` computes the same
      pass@thr metric using ONNX Runtime sessions for frontend + model.

    Notes:
        - Methods are intended to be called as `Functions.method(...)`.
        - No internal state is stored.
    """
    def __init__():
        pass

    def build_mfcc(cfg):
        transform = T.MFCC(
            sample_rate=cfg['sampling_rate'],
            n_mfcc=cfg['n_mfcc'],
            log_mels=True,
            melkwargs=dict(
                n_fft=int(cfg['frame_length_in_s'] * cfg['sampling_rate']),
                win_length=int(cfg['frame_length_in_s'] * cfg['sampling_rate']),
                hop_length=int(cfg['frame_step_in_s'] * cfg['sampling_rate']),
                center=False,
                f_min=cfg['f_min'],
                f_max=cfg['f_max'],
                n_mels=cfg['n_mels'],
            )
        )

        return transform
    
    def eval_acc(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for b in loader:
                x = b["x"]
                y = b["label"]
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
                
        return 100.0 * correct / total
    
    def eval_loss(model, loader, loss_fn):
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for b in loader:
                x = b["x"]; y = b["label"]
                logits = model(x)
                loss = loss_fn(logits, y)
                tot += loss.item() * y.numel()
                n += y.numel()
        return tot / n
    
    def build_grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        search_space: dict {key: [v1, v2, ...], ...}
        return: list of dicts con tutte le combinazioni (cartesiano)
        """
        if not search_space:
            return []

        keys = list(search_space.keys())
        values_lists = [search_space[k] for k in keys]

        for k, vs in zip(keys, values_lists):
            if not isinstance(vs, (list, tuple)) or len(vs) == 0:
                raise ValueError(f"Valori per '{k}' devono essere una lista/tuple non vuota. Trovato: {vs!r}")

        return [dict(zip(keys, combo)) for combo in product(*values_lists)]

    def pass_rate(model, loader, thr=0.999):
        model.eval()
        tot = corr = pass_thr_corr = 0
        with torch.no_grad():
            for b in loader:
                x = b["x"]; y = b["label"]
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                top_p, top_c = probs.max(dim=-1)
                tot += y.numel()
                corr += (top_c == y).sum().item()
                pass_thr_corr += ((top_c == y) & (top_p > thr)).sum().item()
                
        return pass_thr_corr/tot
    
    def softmax_np(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def pass_rate_onnx(test_ds, ort_frontend, ort_model, thr=0.999):
        tot = 0
        pass_thr_corr = 0

        for sample in test_ds:
            x = sample["x"].numpy().astype(np.float32)  # (1,16000)
            y = int(sample["label"])

            x = np.expand_dims(x, 0)                    # (1,1,16000)
            feats = ort_frontend.run(None, {"input": x})[0]
            logits = ort_model.run(None, {"input": feats})[0]  # (1,2)

            probs = Functions.softmax_np(logits)[0]
            pred = int(np.argmax(probs))
            top_p = float(np.max(probs))

            tot += 1
            if (pred == y) and (top_p > thr):
                pass_thr_corr += 1

        return pass_thr_corr / tot