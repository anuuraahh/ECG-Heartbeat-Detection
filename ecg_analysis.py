"""
ECG heartbeat detection and rhythm analysis
===========================================

What this script does, step by step:
  1. Load an ECG recording (real data from the MIT-BIH Arrhythmia Database,
     or a synthetic ECG for a quick offline demo).
  2. Clean the signal with band-pass filters (removes baseline wander and noise).
  3. Detect every heartbeat (the R-peak of each QRS complex) with a simplified
     Pan-Tompkins algorithm.
  4. Compute RR intervals and heart rate.
  5. Flag irregular beats: premature beats (come too early) and pauses (too late).
  6. Check the detector against the cardiologists' beat annotations
     (sensitivity and positive predictive value).
  7. Save plots and a text summary to the output folder.

Usage:
  python ecg_analysis.py --record 100 --minutes 5      # real MIT-BIH record
  python ecg_analysis.py --record 119 --minutes 10     # record with many PVCs
  python ecg_analysis.py --synthetic                   # offline demo, no download
"""

import argparse
import os

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt, find_peaks

# MIT-BIH annotation symbols that mark an actual heartbeat
# (other symbols mark rhythm changes, noise, etc., and are not beats).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Beat types a cardiologist labelled as abnormal ectopic beats.
#   V = premature ventricular contraction (PVC)
#   A = atrial premature beat, a = aberrated atrial premature beat
#   S = supraventricular premature beat, J = junctional premature beat
PREMATURE_SYMBOLS = set("VAaSJ")


# ----------------------------------------------------------------------------
# 1. Loading data
# ----------------------------------------------------------------------------
def load_mitbih(record="100", minutes=5.0, channel=0):
    """Download one MIT-BIH record from PhysioNet.

    Returns the ECG signal (in mV), the sampling rate, and the cardiologists'
    beat annotations (sample positions and beat-type symbols).
    """
    import wfdb  # imported here so the --synthetic demo works without wfdb

    fs = 360  # every MIT-BIH record is sampled at 360 Hz
    sampto = int(minutes * 60 * fs)
    rec = wfdb.rdrecord(record, pn_dir="mitdb", sampto=sampto)
    ann = wfdb.rdann(record, "atr", pn_dir="mitdb", sampto=sampto)

    signal = rec.p_signal[:, channel]
    lead = rec.sig_name[channel]

    is_beat = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ref_peaks = np.asarray(ann.sample)[is_beat]
    ref_symbols = np.asarray(ann.symbol)[is_beat]
    return signal, rec.fs, ref_peaks, ref_symbols, f"MIT-BIH record {record} (lead {lead})"


def make_synthetic_ecg(seconds=60, fs=360, seed=7):
    """Build a realistic-looking ECG from simple wave shapes.

    Each beat is the sum of five bumps: P, Q, R, S and T waves. We add
    a few premature beats, baseline wander (from breathing) and noise,
    so the detector has something realistic to work on.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    ecg = np.zeros_like(t)

    # (offset from R-peak in seconds, amplitude in mV, width in seconds)
    waves = {
        "P": (-0.20, 0.15, 0.025),
        "Q": (-0.03, -0.12, 0.010),
        "R": (0.00, 1.10, 0.012),
        "S": (0.03, -0.25, 0.012),
        "T": (0.25, 0.30, 0.050),
    }

    peaks, symbols = [], []
    beat_time = 0.5
    base_rr = 0.85  # about 70 bpm
    premature = False
    while beat_time < seconds - 0.5:
        scale = 1.4 if premature else 1.0  # ectopic beats often look bigger
        for name, (offset, amp, width) in waves.items():
            if premature and name == "P":
                continue  # premature ventricular beats have no P-wave
            w = width * (2.2 if premature and name in "QRS" else 1.0)
            ecg += scale * amp * np.exp(-((t - (beat_time + offset)) ** 2) / (2 * w**2))
        peaks.append(int(round(beat_time * fs)))
        symbols.append("V" if premature else "N")

        rr = base_rr + rng.normal(0, 0.03)  # normal heart-rate variability
        if premature:
            rr *= 1.35  # compensatory pause after a premature beat
            premature = False
        elif rng.random() < 0.06:
            rr *= 0.65  # the next beat arrives early: a premature beat
            premature = True
        beat_time += rr

    # breathing-related baseline wander + muscle/electrical noise
    ecg += 0.25 * np.sin(2 * np.pi * 0.25 * t)
    ecg += 0.03 * rng.normal(size=t.size)
    return ecg, fs, np.array(peaks), np.array(symbols), "Synthetic ECG (demo)"


# ----------------------------------------------------------------------------
# 2. Filtering
# ----------------------------------------------------------------------------
def bandpass(signal, fs, low, high, order=3):
    """Zero-phase band-pass filter (filtfilt does not shift the peaks in time)."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


# ----------------------------------------------------------------------------
# 3. R-peak detection (simplified Pan-Tompkins)
# ----------------------------------------------------------------------------
def detect_r_peaks(signal, fs):
    """Find the R-peak of every heartbeat.

    Pan-Tompkins idea in four steps:
      a) band-pass 5-15 Hz keeps the sharp QRS energy, drops P/T waves and noise
      b) derivative highlights the steep slopes of the QRS complex
      c) squaring makes everything positive and boosts large slopes
      d) a 150 ms moving average merges each QRS into one smooth bump
    Then we pick bumps above an adaptive threshold, at least 200 ms apart
    (the heart can't physiologically beat again that fast), and move each
    detection to the true R-peak in the cleaned signal.
    """
    qrs_band = bandpass(signal, fs, 5, 15)
    slope = np.gradient(qrs_band)
    energy = slope**2
    window = int(0.150 * fs)
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    # Adaptive threshold: 30% of the largest bump in the surrounding 5 seconds,
    # so the detector copes when the signal gets bigger or smaller over time.
    local_max = maximum_filter1d(integrated, size=int(5 * fs))
    candidates, _ = find_peaks(integrated, distance=int(0.2 * fs))
    candidates = candidates[integrated[candidates] > 0.3 * local_max[candidates]]

    # Refine: the true R-peak is the largest deflection near each bump.
    clean = bandpass(signal, fs, 0.5, 40)
    search = int(0.10 * fs)
    r_peaks = []
    for c in candidates:
        lo, hi = max(c - search, 0), min(c + search, len(clean))
        r_peaks.append(lo + np.argmax(np.abs(clean[lo:hi])))
    r_peaks = np.unique(np.array(r_peaks, dtype=int))

    # Two candidates can snap to the same beat; keep peaks >= 200 ms apart.
    if r_peaks.size:
        keep = np.concatenate([[True], np.diff(r_peaks) >= int(0.2 * fs)])
        r_peaks = r_peaks[keep]
    return r_peaks, clean


# ----------------------------------------------------------------------------
# 4 & 5. Heart rate and irregular beats
# ----------------------------------------------------------------------------
def analyse_rhythm(r_peaks, fs, premature_ratio=0.80, pause_ratio=1.5):
    """Compute RR intervals and heart rate, and flag irregular beats.

    A beat is 'premature' if it arrives in less than 80% of the typical
    (local median) RR interval, and a 'pause' if the gap before it is more
    than 1.5x the typical interval.
    """
    rr = np.diff(r_peaks) / fs  # seconds between consecutive beats
    hr = 60.0 / rr  # beats per minute

    premature, pauses = [], []
    for i in range(len(rr)):
        lo, hi = max(0, i - 4), min(len(rr), i + 5)
        typical = np.median(np.delete(rr[lo:hi], i - lo)) if hi - lo > 1 else rr[i]
        beat_index = i + 1  # rr[i] ends at beat i+1
        if rr[i] < premature_ratio * typical:
            premature.append(beat_index)
        elif rr[i] > pause_ratio * typical:
            pauses.append(beat_index)

    return {
        "rr": rr,
        "hr": hr,
        "mean_hr": float(np.mean(hr)) if hr.size else float("nan"),
        "min_hr": float(np.min(hr)) if hr.size else float("nan"),
        "max_hr": float(np.max(hr)) if hr.size else float("nan"),
        # SDNN-style variability: standard deviation of RR intervals (ms)
        "rr_sd_ms": float(np.std(rr) * 1000) if rr.size else float("nan"),
        "premature": np.array(premature, dtype=int),
        "pauses": np.array(pauses, dtype=int),
    }


# ----------------------------------------------------------------------------
# 6. Evaluation against cardiologists' annotations
# ----------------------------------------------------------------------------
def evaluate(detected, reference, fs, tolerance_s=0.15):
    """Match detected beats to annotated beats within +/-150 ms.

    Sensitivity = how many real beats we found (TP / (TP + FN)).
    PPV (positive predictive value) = how many detections were real beats
    (TP / (TP + FP)).
    """
    tol = int(tolerance_s * fs)
    matched_ref = np.zeros(len(reference), dtype=bool)
    tp = 0
    for d in detected:
        idx = np.searchsorted(reference, d)
        best, best_dist = None, tol + 1
        for j in (idx - 1, idx):
            if 0 <= j < len(reference) and not matched_ref[j]:
                dist = abs(int(reference[j]) - int(d))
                if dist < best_dist:
                    best, best_dist = j, dist
        if best is not None and best_dist <= tol:
            matched_ref[best] = True
            tp += 1
    fp = len(detected) - tp
    fn = len(reference) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "matched_ref": matched_ref,
    }


def premature_agreement(r_peaks, flagged, ref_peaks, ref_symbols, fs, tolerance_s=0.15):
    """Of the beats a cardiologist labelled premature, how many did we flag?"""
    tol = int(tolerance_s * fs)
    ref_premature = ref_peaks[np.isin(ref_symbols, list(PREMATURE_SYMBOLS))]
    flagged_samples = r_peaks[flagged] if len(flagged) else np.array([], dtype=int)
    caught = sum(np.any(np.abs(flagged_samples - p) <= tol) for p in ref_premature)
    return int(caught), int(len(ref_premature))


# ----------------------------------------------------------------------------
# 7. Plots and report
# ----------------------------------------------------------------------------
# Colours: one blue for the ECG, one orange for flagged beats (flagged beats
# also use a different marker shape, so they don't rely on colour alone).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_signal(raw, clean, r_peaks, flagged, fs, title, path, seconds=10):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show a 10-second window; if any beat was flagged, centre the window on
    # the first flagged beat so the plot shows what an irregular beat looks like.
    start = 0
    if len(flagged):
        start = max(0, int(r_peaks[np.min(flagged)] - 3 * fs))
    end = min(len(raw), start + int(seconds * fs))
    start = max(0, end - int(seconds * fs))
    t = np.arange(start, end) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, facecolor=SURFACE)

    axes[0].plot(t, raw[start:end], color=INK_2, linewidth=1)
    axes[0].set_title("Raw ECG: note the drifting baseline", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("mV", color=INK_2)

    axes[1].plot(t, clean[start:end], color=BLUE, linewidth=1.2, label="Filtered ECG (0.5-40 Hz)")
    shown = r_peaks[(r_peaks >= start) & (r_peaks < end)]
    normal = np.setdiff1d(shown, r_peaks[flagged]) if len(flagged) else shown
    odd = np.intersect1d(shown, r_peaks[flagged]) if len(flagged) else np.array([], dtype=int)
    axes[1].plot(normal / fs, clean[normal], "o", color=INK, markersize=5, label="Detected beat")
    if odd.size:
        axes[1].plot(odd / fs, clean[odd] + 0.08, "v", color=ORANGE, markersize=9,
                     markeredgecolor=SURFACE, label="Flagged irregular beat")
    axes[1].set_title("Cleaned ECG with detected heartbeats", loc="left", color=INK, fontsize=11)
    axes[1].set_ylabel("mV", color=INK_2)
    axes[1].set_xlabel("Time (s)", color=INK_2)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, frameon=False,
                   fontsize=9, labelcolor=INK_2)

    for ax in axes:
        _style(ax)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_heart_rate(r_peaks, rhythm, fs, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_beats = r_peaks[1:] / fs / 60  # minutes; HR value belongs to the beat ending each RR
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURFACE)
    ax.plot(t_beats, rhythm["hr"], color=BLUE, linewidth=1.2, label="Heart rate, beat to beat")
    flagged = np.concatenate([rhythm["premature"], rhythm["pauses"]]).astype(int)
    if flagged.size:
        ax.plot(r_peaks[flagged] / fs / 60, rhythm["hr"][flagged - 1], "^", color=ORANGE,
                markersize=8, markeredgecolor=SURFACE, label="Flagged irregular beat")
    ax.axhspan(60, 100, color=GRID, alpha=0.45, linewidth=0,
               label="Normal resting range (60-100 bpm)")
    ax.set_xlabel("Time (min)", color=INK_2)
    ax.set_ylabel("bpm", color=INK_2)
    ax.set_title(f"{title}: heart rate over time", loc="left", color=INK, fontsize=11)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK_2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(path, title, fs, n_samples, r_peaks, rhythm, scores, prem):
    minutes = n_samples / fs / 60
    lines = [
        f"ECG analysis report: {title}",
        "=" * 60,
        f"Duration analysed:      {minutes:.1f} min at {fs} Hz",
        f"Beats detected:         {len(r_peaks)}",
        f"Mean heart rate:        {rhythm['mean_hr']:.0f} bpm "
        f"(min {rhythm['min_hr']:.0f}, max {rhythm['max_hr']:.0f}, beat to beat)",
        f"RR variability (SD):    {rhythm['rr_sd_ms']:.0f} ms",
        f"Premature beats flagged: {len(rhythm['premature'])}",
        f"Pauses flagged:          {len(rhythm['pauses'])}",
    ]
    mean_hr = rhythm["mean_hr"]
    if mean_hr < 60:
        lines.append("Average rate is below 60 bpm (bradycardia range).")
    elif mean_hr > 100:
        lines.append("Average rate is above 100 bpm (tachycardia range).")
    else:
        lines.append("Average rate is in the normal 60-100 bpm range.")

    if scores is not None:
        lines += [
            "",
            "Detector accuracy vs. reference beat annotations (+/-150 ms):",
            f"  True positives:  {scores['tp']}",
            f"  False positives: {scores['fp']}",
            f"  Missed beats:    {scores['fn']}",
            f"  Sensitivity:     {scores['sensitivity'] * 100:.2f}%",
            f"  PPV:             {scores['ppv'] * 100:.2f}%",
        ]
    if prem is not None and prem[1] > 0:
        lines.append(
            f"  Annotated premature beats caught by the RR rule: {prem[0]} of {prem[1]}"
        )
    lines += ["", "Note: educational project only, not a medical device or diagnosis."]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ECG heartbeat detection and rhythm analysis")
    parser.add_argument("--record", default="100", help="MIT-BIH record number, e.g. 100, 119, 208")
    parser.add_argument("--minutes", type=float, default=5.0, help="how many minutes to analyse")
    parser.add_argument("--synthetic", action="store_true", help="use a synthetic ECG (no download)")
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    if args.synthetic:
        raw, fs, ref_peaks, ref_symbols, title = make_synthetic_ecg(seconds=args.minutes * 60)
        tag = "synthetic"
    else:
        raw, fs, ref_peaks, ref_symbols, title = load_mitbih(args.record, args.minutes)
        tag = f"record_{args.record}"

    r_peaks, clean = detect_r_peaks(raw, fs)
    rhythm = analyse_rhythm(r_peaks, fs)
    flagged = np.concatenate([rhythm["premature"], rhythm["pauses"]]).astype(int)
    scores = evaluate(r_peaks, ref_peaks, fs) if len(ref_peaks) else None
    prem = premature_agreement(r_peaks, rhythm["premature"], ref_peaks, ref_symbols, fs)

    os.makedirs(args.out, exist_ok=True)
    plot_signal(raw, clean, r_peaks, flagged, fs, title, os.path.join(args.out, f"{tag}_signal.png"))
    plot_heart_rate(r_peaks, rhythm, fs, title, os.path.join(args.out, f"{tag}_heart_rate.png"))
    report = write_report(os.path.join(args.out, f"{tag}_report.txt"), title, fs, len(raw),
                          r_peaks, rhythm, scores, prem)
    print(report)
    print(f"\nPlots and report saved in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
