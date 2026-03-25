"""
extractor.py  —  Rich multi-feature SDR processor for TEXBAT .bin files
========================================================================
Extracts 6 node-level observables per satellite per snapshot:
  1. CN0      (dB-Hz)       — absolute signal strength
  2. Doppler  (Hz)          — carrier frequency offset
  3. CodePhase (chips)      — fractional peak position from parabolic interp.
  4. CN0_rate  (dB/snap)    — first-order temporal derivative of CN0
  5. Dop_rate  (Hz/snap)    — first-order temporal derivative of Doppler
  6. PeakRatio (linear)     — peak-to-mean-sidelobe ratio (replica of SQM)

Why these 6 features?
----------------------
* CN0 + PeakRatio  → signal quality / power anomaly detection (ds1/ds2)
* Doppler          → carrier frequency consistency (required for all attacks)
* CodePhase        → code-domain distortion (sensitive to traction attacks)
* CN0_rate         → sudden power jumps reveal switching attacks (ds1/ds5)
* Dop_rate         → acceleration residuals expose fake dynamic (ds5/ds6)

The extractor writes one CSV per input .bin file into DATA_PROC_DIR.
Column layout:
  Time, PRN, CN0_dBHz, Doppler_Hz, CodePhase_chips, CN0_rate, Dop_rate, PeakRatio
"""

import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import os
import gc

# Use absolute import when running as a package (python -m src.extractor)
# and relative import when imported from another src module.
try:
    from src import config
except ImportError:
    from .. import config   # relative import within the package


# ── C/A Code Generator ────────────────────────────────────────────────────────

# G2 tap pairs for all 32 GPS PRNs (standard ICD-GPS-200 Table 3-I).
_G2_TAPS = {
     1:(2,6),  2:(3,7),  3:(4,8),  4:(5,9),  5:(1,9),  6:(2,10), 7:(1,8),  8:(2,9),
     9:(3,10),10:(2,3), 11:(3,4), 12:(5,6), 13:(6,7), 14:(7,8), 15:(8,9), 16:(9,10),
    17:(1,4), 18:(2,5), 19:(3,6), 20:(4,7), 21:(5,8), 22:(6,9), 23:(1,3), 24:(4,6),
    25:(5,7), 26:(6,8), 27:(7,9), 28:(8,10),29:(1,6), 30:(2,7), 31:(3,8), 32:(4,9),
}


def _generate_ca_code(prn: int) -> np.ndarray:
    """Return the 1023-chip C/A code for PRN `prn` as {-1, +1} numpy array."""
    tap_a, tap_b = _G2_TAPS[prn]
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    code = np.empty(1023, dtype=np.float32)
    for k in range(1023):
        chip = (g1[9] ^ (g2[tap_a - 1] ^ g2[tap_b - 1])) & 1
        code[k] = 1.0 - 2.0 * chip          # 0 → +1, 1 → -1
        fb_g1 = (g1[2] ^ g1[9]) & 1
        g1 = np.roll(g1, 1); g1[0] = fb_g1
        fb_g2 = (g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]) & 1
        g2 = np.roll(g2, 1); g2[0] = fb_g2
    return code


def _resample_code(code_1023: np.ndarray, total_samples: int) -> np.ndarray:
    """
    Resample the 1023-chip C/A code to `total_samples` at the work sampling rate.
    Uses ceiling-index mapping — the standard GPS receiver approach.
    """
    samples_per_ms = total_samples // config.INT_TIME_MS
    tc = 1.0 / 1.023e6          # chip duration in seconds
    ts = 1.0 / config.FS_WORK   # sample duration in seconds
    # One-ms index template
    idx_1ms = np.ceil(ts * np.arange(1, samples_per_ms + 1) / tc).astype(int) - 1
    idx_1ms = np.clip(idx_1ms, 0, 1022)
    # Repeat for INT_TIME_MS milliseconds
    full = np.tile(code_1023[idx_1ms], config.INT_TIME_MS)
    # Force exact length (rounding may cause ±1 sample error)
    if len(full) > total_samples:
        full = full[:total_samples]
    elif len(full) < total_samples:
        full = np.pad(full, (0, total_samples - len(full)))
    return full


# ── Parabolic Sub-Sample Peak Interpolation ───────────────────────────────────

def _parabolic_peak(corr_mag_sq: np.ndarray) -> tuple[float, float]:
    """
    Given a 1-D correlation magnitude-squared vector, return:
      (peak_index_fractional, peak_value_linear)
    using a 3-point parabola fit around the discrete maximum.

    This gives sub-chip code-phase resolution without DLL hardware.
    Returns (peak_idx, peak_val).
    """
    n = len(corr_mag_sq)
    i_max = int(np.argmax(corr_mag_sq))
    peak_val = corr_mag_sq[i_max]

    # Wrap-around neighbours for circular correlation
    i_prev = (i_max - 1) % n
    i_next = (i_max + 1) % n
    y0, y1, y2 = corr_mag_sq[i_prev], peak_val, corr_mag_sq[i_next]

    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        frac_offset = 0.0   # flat top — no sub-sample refinement possible
    else:
        frac_offset = 0.5 * (y0 - y2) / denom  # ∈ (-0.5, +0.5) samples

    peak_idx = (i_max + frac_offset) / n  # normalised to [0, 1) chip fractions
    return float(peak_idx), float(peak_val)


# ── Feature Extractor Class ───────────────────────────────────────────────────

class FeatureExtractor:
    """
    Processes a single TEXBAT .bin file and writes a CSV with 8 columns:
      Time, PRN, CN0_dBHz, Doppler_Hz, CodePhase_chips,
      CN0_rate, Dop_rate, PeakRatio
    """

    def __init__(self):
        # Pre-compute conjugate FFTs of all 32 C/A codes once at startup.
        # Reusing these across all snapshots avoids 32 × N FFTs per snapshot.
        print("⚡ [Init] Pre-computing C/A code FFT library …")
        self._samples_per_snap = int(
            (config.FS_WORK / 1000) * config.INT_TIME_MS
        )
        self._fft_codes = self._precompute_fft_codes()
        # Time axis for Doppler wipeoff (computed once, reused per snapshot)
        self._ts_vec = np.arange(self._samples_per_snap) / config.FS_WORK
        # CN0 compensation: 10·log10(1 / INT_TIME_s)
        self._cn0_gain = 10.0 * np.log10(1.0 / (config.INT_TIME_MS / 1000.0))
        print(f"   ✓ {self._samples_per_snap} samples/snap, "
              f"CN0 gain = {self._cn0_gain:.1f} dB")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _precompute_fft_codes(self) -> np.ndarray:
        """
        Returns shape (32, N) complex array:
          row k → conjugate FFT of C/A code for PRN (k+1),
          at the work sampling rate and with INT_TIME_MS repetitions.
        Used for batch parallel correlation via FFT convolution.
        """
        N = self._samples_per_snap
        fft_matrix = np.zeros((32, N), dtype=np.complex64)
        for prn in range(1, 33):
            code = _generate_ca_code(prn)
            resampled = _resample_code(code, N)
            fft_matrix[prn - 1] = fft(resampled).conj()
        return fft_matrix

    def _read_snapshot(self, f, snap_index: int) -> np.ndarray | None:
        """
        Read one raw snapshot from an open .bin file handle.
        Returns a complex64 array of length `_samples_per_snap`, or None on EOF.

        File format: interleaved int16 I/Q at FS_RAW, so each sample is 4 bytes.
        Stride between snapshots: SNAP_STRIDE_SEC × FS_RAW × 4 bytes.
        """
        # Byte position of this snapshot in the raw file
        raw_samples_per_snap = int(config.FS_RAW / 1000 * config.INT_TIME_MS)
        stride_bytes = int(config.SNAP_STRIDE_SEC * config.FS_RAW * 4)
        seek_pos = snap_index * stride_bytes
        f.seek(seek_pos)

        # Read int16 pairs (I, Q)
        n_int16 = raw_samples_per_snap * 2
        raw = np.fromfile(f, dtype=np.int16, count=n_int16)
        if len(raw) < n_int16:
            return None   # EOF

        # Complex baseband + decimation
        iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        iq_dec = iq[:: config.DECIMATION]

        # Force exact length after decimation
        N = self._samples_per_snap
        if len(iq_dec) > N:
            iq_dec = iq_dec[:N]
        elif len(iq_dec) < N:
            iq_dec = np.pad(iq_dec, (0, N - len(iq_dec)))
        return iq_dec

    def _correlate_all_prns(self, sig: np.ndarray, doppler: int) -> np.ndarray:
        """
        Strip the carrier at `doppler` Hz, then compute parallel circular
        correlation with all 32 C/A codes via FFT.

        Returns corr_sq shape (32, N): magnitude-squared correlation surface.
        """
        wipeoff = sig * np.exp(-1j * 2.0 * np.pi * doppler * self._ts_vec)
        sig_fft = fft(wipeoff)
        # Batch multiply + IFFT
        corr_sq = np.abs(ifft(sig_fft * self._fft_codes, axis=1)) ** 2
        return corr_sq  # shape (32, N)

    def _estimate_features(
        self, sig: np.ndarray
    ) -> dict[int, dict]:
        """
        For a single snapshot, search across the Doppler grid and return a
        dict keyed by PRN (1–32) with fields:
          cn0, doppler, code_phase, peak_ratio
        Only PRNs exceeding CN0_THRESHOLD are included.
        """
        N = self._samples_per_snap
        doppler_range = range(
            -config.DOPPLER_MAX_HZ,
            config.DOPPLER_MAX_HZ + config.DOPPLER_STEP_HZ,
            config.DOPPLER_STEP_HZ,
        )

        # Accumulators: best values found so far for each PRN
        best_peak    = np.zeros(32, dtype=np.float64)   # peak |corr|²
        best_doppler = np.zeros(32, dtype=np.float64)   # Hz
        best_corr_sq = [None] * 32                      # full 1-D corr profile

        for dop in doppler_range:
            corr_sq = self._correlate_all_prns(sig, dop)  # (32, N)
            peaks = corr_sq.max(axis=1)                    # (32,)
            update_mask = peaks > best_peak
            for k in np.where(update_mask)[0]:
                best_peak[k]    = peaks[k]
                best_doppler[k] = dop
                best_corr_sq[k] = corr_sq[k].copy()

        # Derive per-PRN observables
        results = {}
        for k in range(32):
            prn = k + 1
            if best_corr_sq[k] is None:
                continue

            corr = best_corr_sq[k]  # shape (N,)

            # Parabolic peak interpolation → fractional code phase + peak power
            frac_phase, peak_val = _parabolic_peak(corr)

            # Noise floor = mean of samples excluding a ±2-chip window around peak
            i_peak_disc = int(np.argmax(corr))
            guard = max(1, int(2 * config.FS_WORK / 1.023e6))  # ~2 chips in samples
            mask = np.ones(N, dtype=bool)
            for di in range(-guard, guard + 1):
                mask[(i_peak_disc + di) % N] = False
            noise_floor = corr[mask].mean() if mask.any() else corr.mean()

            if noise_floor < 1e-12:
                continue

            snr_linear = (peak_val - noise_floor) / noise_floor
            if snr_linear <= 0:
                continue

            cn0 = 10.0 * np.log10(snr_linear) + self._cn0_gain
            if cn0 < config.CN0_THRESHOLD:
                continue

            # Peak-to-sidelobe ratio (peak / noise floor, linear)
            peak_ratio = peak_val / noise_floor

            results[prn] = {
                'cn0':        cn0,
                'doppler':    float(best_doppler[k]),
                'code_phase': frac_phase,   # ∈ [0, 1) chips
                'peak_ratio': peak_ratio,
            }

        return results

    # ── Public API ───────────────────────────────────────────────────────────

    def process_single_file(self, filename: str) -> str | None:
        """
        Extract features from `filename` and save results to a CSV.
        Returns the output CSV path on success, or None if the raw file is absent.
        """
        raw_path = os.path.join(config.DATA_RAW_DIR, filename)
        out_csv  = os.path.join(
            config.DATA_PROC_DIR, filename.replace('.bin', '_features.csv')
        )

        if not os.path.exists(raw_path):
            print(f"  ⚠ Raw file not found: {raw_path}")
            return None

        if os.path.exists(out_csv):
            print(f"  ✓ Already extracted: {out_csv}  (skipping)")
            return out_csv

        print(f"\n🚀 Extracting: {filename}")

        file_size  = os.path.getsize(raw_path)
        stride_bytes = int(config.SNAP_STRIDE_SEC * config.FS_RAW * 4)
        total_snaps  = file_size // stride_bytes

        rows = []
        # Buffer to compute temporal derivatives (cn0_rate, dop_rate)
        prev_snap: dict[int, dict] = {}  # PRN → last snapshot's raw values

        with open(raw_path, 'rb') as f:
            for snap_idx in tqdm(range(total_snaps), desc=filename, unit='snap'):
                sig = self._read_snapshot(f, snap_idx)
                if sig is None:
                    break

                cur_snap = self._estimate_features(sig)
                time_s   = round(snap_idx * config.SNAP_STRIDE_SEC, 2)

                for prn, vals in cur_snap.items():
                    # Temporal derivatives require at least two consecutive snaps
                    if prn in prev_snap:
                        cn0_rate = vals['cn0']     - prev_snap[prn]['cn0']
                        dop_rate = vals['doppler'] - prev_snap[prn]['doppler']
                    else:
                        # First occurrence of this PRN: rate = 0 (unknown)
                        cn0_rate = 0.0
                        dop_rate = 0.0

                    rows.append({
                        'Time':             time_s,
                        'PRN':              prn,
                        'CN0_dBHz':         round(vals['cn0'],        2),
                        'Doppler_Hz':       int(vals['doppler']),
                        'CodePhase_chips':  round(vals['code_phase'],  6),
                        'CN0_rate':         round(cn0_rate,            4),
                        'Dop_rate':         round(dop_rate,            2),
                        'PeakRatio':        round(vals['peak_ratio'],  4),
                    })

                prev_snap = cur_snap

                if snap_idx % 200 == 0:
                    gc.collect()

        if not rows:
            print(f"  ✗ No valid signals found in {filename}")
            return None

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"  ✓ Saved {len(df)} rows → {out_csv}")
        return out_csv
