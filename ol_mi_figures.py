#!/usr/bin/env python3
"""
OL_MI report figures — Subject B6, session 20260708 (256-ch ECoG @ 1 kHz).

Seven figures, five code paths (the three per-task "own-ROI" panels share one function):

    fig_single_channel()      single-trial single-channel HG/beta, 3 tasks   [needs ol_mi_ch5.npz]
    fig_channel_grid()        HG execute-rest selection map on ChMapB2        [ol_mi_env.npz]
    fig_own_roi(task)         per-task 3x3 channel-level panel, own ROIs      [needs ol_mi_bothroi.npz]
    fig_roi_grid()            HG-ROI vs beta-ROI layout on ChMapB2            [ol_mi_env.npz]
    fig_pace_detection()      per-trial individual-pace / arrhythmia test     [ol_mi_env.npz]

DATA
----
All inputs are the frozen analysis products under DATA_DIR (default ./figdata):
  ol_mi_env.npz, ol_mi_labels.json  -- the envelope checkpoint (saved artifacts, always available)
  ol_mi_ch5.npz, ol_mi_ch5.json     -- 5-channel single-trial 1 kHz traces  (re-extract if absent)
  ol_mi_bothroi.npz                 -- per-trial ROI-mean traces, both bands (re-extract if absent)
The two "re-extract if absent" files are produced on the recording server from the raw .mat;
the three env-based figures run anywhere from the checkpoint alone.

Envelope convention for every figure: 70-150 Hz (HG) / 13-30 Hz (beta) broadband -> Hilbert
amplitude -> common-median reference, z-score to that trial's rest -> 100 Hz grid -> 200 ms
smoothing for display.
"""
import os
import json
import numpy as np

DATA_DIR = 'data'
OUT_DIR  = '.'

# ---- palette / labels ----------------------------------------------------------
C_HG, C_BE, C_CUE, C_SW = '#c0392b', '#2471a3', '#16a085', '#8e44ad'
TASK_ORDER = ['ColorPaced', 'SelfPaced', 'RandSwitch']
TASK_LABEL = {'ColorPaced': 'ColorPaced (external cue)',
              'SelfPaced':  'SelfPaced (internal)',
              'RandSwitch': 'RandSwitch (2 s switch)'}
TASK_COL   = {'ColorPaced': '#16a085', 'SelfPaced': '#e67e22', 'RandSwitch': '#8e44ad'}
ACTION = {0: 'Home/rest', 1: 'Right thumb', 2: 'Both legs', 3: 'Left thumb',
          4: 'Head', 5: 'Lips', 6: 'Tongue', 7: 'Both middle fingers'}

# ChMapB2 electrode layout: grid[row, col] = 1-indexed channel id (11 x 23 = 253 on-grid)
CHMAP_B2 = np.array([
    [137,143,148,152,155, 23, 19, 14,  8,  2,130,136,142,147,151, 18, 13,  7,  1,129,135,141,146],
    [159,160, 30, 28, 25, 21, 16, 10,  4,132,138,144,149,153,156,158, 27, 24, 20, 15,  9,  3,131],
    [134,140,170,174,178,182,186,189,192, 32, 31, 29, 26, 22, 17, 11,  5,133,139,145,150,154,157],
    [ 49, 45, 41, 38, 35,163,166,169,173,177,181,185,188,191, 64, 61, 58, 54, 50, 46, 42, 12,  6],
    [221, 62, 59, 56, 52, 48, 44, 40, 37, 34,162,165,168,172,176,180,184,187,190, 63, 60, 57, 53],
    [205,208,211,214,218,222, 93, 89, 55, 51, 47, 43, 39, 36, 33,161,164,167,171,175,179,183,217],
    [ 70, 66,194,198,202,206,209,212,215,219,223, 94, 90, 86, 83, 80, 77, 73, 69, 65,193,197,201],
    [ 88, 85, 82, 79, 75, 71, 67,195,199,203,207,210,213,216,220,224, 95, 91, 87, 84, 81, 78, 74],
    [248,252,255,128,126,123,119,114,109, 76, 72, 68,196,200,204,237,242,247,251,254,256, 96, 92],
    [106,102, 98,226,230,234,239,244,249,253,127,124,120,115,110,105,101, 97,225,229,233,238,243],
    [104,100,228,232,236,241,246,122,117,112,107,103, 99,227,231,235,240,245,250,125,121,116,111],
], dtype=int)
GRID_ROWS, GRID_COLS = CHMAP_B2.shape
OFFGRID = [107, 112, 117]                        # 0-indexed reference/ground (ids 108/113/118)
CH2RC = {CHMAP_B2[r, c] - 1: (r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)}

def to_grid(vec256):
    """Map a length-256 per-channel vector onto the ChMapB2 grid; off-grid channels -> NaN."""
    g = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for ch, (r, c) in CH2RC.items():
        g[r, c] = vec256[ch]
    return g


# ---- style / smoothing ---------------------------------------------------------
def apply_style():
    """Compact stand-in for the figure-style skill: the rcParams these figures rely on."""
    _rc = '/usr/share/matplotlib/mpl-data/matplotlibrc'      # sandbox-only workaround;
    if os.path.exists(_rc):                                  # skipped where the path is absent
        os.environ.setdefault('MATPLOTLIBRC', _rc)           # (e.g. a normal conda matplotlib)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8,
        'axes.titlelocation': 'left', 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

def smooth(x, w=20, axis=-1):
    """200 ms moving average on the 100 Hz analysis grid."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.asarray(x, float), w, axis=axis, mode='nearest')


# ---- data loaders --------------------------------------------------------------
def load_env():
    """The envelope checkpoint: (npz dict, labels dict, META, file-records)."""
    npz = dict(np.load(os.path.join(DATA_DIR, 'ol_mi_env.npz')))
    labels = json.load(open(os.path.join(DATA_DIR, 'ol_mi_labels.json')))
    return npz, labels, labels['meta'], labels['files']

def _need(fname):
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{fname} not in {DATA_DIR}/ — this figure needs the server-side extraction "
            f"(see module docstring). The env-checkpoint figures run without it.")
    return p


# ================================================================================
# FIGURE 1 — single trial, single channel: same action ext/int vs action sequence
# ================================================================================
def fig_single_channel(out='ol_mi_single_channel_ext_vs_int.png'):
    """Needs ol_mi_ch5.npz / .json (5 most HG-modulated channels, one example trial per task)."""
    import matplotlib.pyplot as plt
    ch5 = np.load(_need('ol_mi_ch5.npz'))
    ch5meta = json.load(open(_need('ol_mi_ch5.json')))
    _, labels, META, files = load_env()
    Fs = 1000.0
    pick5 = [int(c) for c in META['roi_hg'][:5]]

    rs = next(f for f in files if f['task'] == 'RandSwitch' and f['run'] == '161339'
              and f['file'] == 'Data0001.mat')
    sw, tg = rs['switch_rel'], rs['switch_target']
    seg_cols = plt.cm.tab10(np.linspace(0, 1, 10))
    cols = [('ColorPaced', 'External cue — ColorPaced (Tongue)', C_CUE),
            ('SelfPaced',  'Internal pace — SelfPaced (Tongue)', '#e67e22'),
            ('RandSwitch', 'Action sequence — RandSwitch (7 actions x 2 s)', C_SW)]

    fig, axes = plt.subplots(5, 3, figsize=(19, 13.5), sharey=True)
    for col, (task, ctitle, ccol) in enumerate(cols):
        t, HG, BE = ch5[f'{task}_t'], ch5[f'{task}_hg'], ch5[f'{task}_be']
        ee = ch5meta[task]['exec_end_rel']
        sel = (t >= -1.0) & (t <= ee + 0.3)
        for row, chan in enumerate(pick5):
            ax = axes[row, col]
            if task == 'ColorPaced':
                for tt in range(0, int(ee) + 1, 2):                    # 2 s cue-cycle stripes
                    ax.axvspan(tt, min(tt + 2, ee), color='0.95' if (tt // 2) % 2 == 0 else 'white', zorder=0)
                for cr in ch5meta[task].get('cue_rel', []):
                    ax.axvline(cr, color=C_CUE, ls='--', lw=1.0, alpha=0.75, zorder=2)
            elif task == 'RandSwitch':
                for k in range(len(sw) - 1):                           # colored action segments
                    ax.axvspan(sw[k], sw[k + 1], color=seg_cols[tg[k]], alpha=0.14, zorder=0)
                    ax.axvline(sw[k], color=C_SW, ls='--', lw=0.9, alpha=0.7, zorder=2)
                    if row == 0:
                        ax.text((sw[k] + sw[k + 1]) / 2, 3.45, ACTION[tg[k]].replace(' ', '\n'),
                                ha='center', va='top', fontsize=5.6, color='0.25', zorder=6)
            ax.axhline(0, color='0.6', lw=0.6, zorder=1)
            ax.plot(t[sel], smooth(HG[row], int(0.2 * Fs))[sel], color=C_HG, lw=1.3, zorder=5, label='HG 70-150 Hz')
            ax.plot(t[sel], smooth(BE[row], int(0.2 * Fs))[sel], color=C_BE, lw=1.3, zorder=5, label='\u03b2 13-30 Hz')
            ax.set_ylim(-2.2, 3.6); ax.set_xlim(-1.0, ee + 0.3); ax.margins(x=0)
            if col == 0: ax.set_ylabel(f"channel {chan}\nSD vs rest", fontsize=9)
            if row == 0: ax.set_title(ctitle, fontsize=9.5, fontweight='bold', color=ccol)
            if row == 4: ax.set_xlabel("time from execute onset (s)", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='upper right', ncol=2)
    fig.suptitle("Single trial, single channel — same action (external / internal) vs action "
                 "sequence (B6, 1 kHz, 200 ms smoothed)", fontsize=11.5, y=0.997)
    fig.tight_layout(rect=[0, 0.016, 1, 0.975])
    fig.savefig(os.path.join(OUT_DIR, out), dpi=170, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# FIGURE 2 — channel-selection grid: HG execute-rest map, top-5 marked
# ================================================================================
def fig_channel_grid(out='ol_mi_channel_selection_grid.png'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    npz, labels, META, _ = load_env()
    dHG = npz['acc_dHG']
    roi_hg = np.array(META['roi_hg'])
    top5 = [85, 71, 198, 75, 58]

    G = to_grid(dHG); vmax = np.nanpercentile(np.abs(G), 98)
    fig, ax = plt.subplots(figsize=(11.5, 6))
    im = ax.imshow(G, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    for r in range(GRID_ROWS):                                         # off-grid hatch
        for c in range(GRID_COLS):
            if np.isnan(G[r, c]):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc='0.85', ec='0.7', lw=0.3, hatch='///'))
    for ch in roi_hg:                                                  # 40-ch ROI outline
        r, c = CH2RC[int(ch)]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec='black', lw=1.4))
    for i, ch in enumerate(top5, 1):                                   # top-5 stars
        r, c = CH2RC[ch]
        ax.plot(c, r, marker='*', ms=22, mfc='yellow', mec='black', mew=1.1, zorder=6)
        ax.text(c, r - 0.02, str(i), ha='center', va='center', fontsize=7.5, fontweight='bold', zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("23 columns (anterior <-> posterior)", fontsize=8); ax.set_ylabel("11 rows", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("\u0394HG = execute - rest (SD)", fontsize=8); cb.ax.tick_params(labelsize=7)
    leg = "\n".join(f"* {i}   ch {ch:>3}   \u0394HG {dHG[ch]:+.2f}" for i, ch in enumerate(top5, 1))
    ax.text(1.14, 0.5, "Top-5 HG channels\n(single-trial figure)\n\n" + leg, transform=ax.transAxes,
            fontsize=8.2, va='center', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#fffbe6', ec='0.6', lw=0.6))
    fig.suptitle("Channel-selection grid — HG execute-rest map, top-5 channels marked (B6, ChMapB2)",
                 fontsize=10.5, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, out), dpi=195, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# FIGURE 3-5 — per-task channel-level panel, HG and beta each on its own ROI
# ================================================================================
def fig_own_roi(task, out=None):
    """Needs ol_mi_bothroi.npz. 3 rows (all / top-40 / top-5) x 3 cols (trace / HG raster / beta raster)."""
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    out = out or f'ol_mi_{task.lower()}_channel_levels_ownROI.png'
    br = np.load(_need('ol_mi_bothroi.npz'))
    npz, labels, META, files = load_env()
    roi_hg, roi_b = np.array(META['roi_hg']), np.array(META['roi_b'])
    ongrid = set(c for c in range(256) if c not in OFFGRID)

    def task_delta(key):
        idx = [f['idx'] for f in files if f['task'] == task]
        return np.nanmean(np.vstack([npz[f'{i}_{key}'] for i in idx]), 0)
    dHG, dB = task_delta('dHG'), task_delta('dB')
    hg_all = [c for c in np.argsort(dHG)[::-1] if c in ongrid]
    be_all = [c for c in np.argsort(dB) if c in ongrid]
    levels = [('all channels (n=253)', hg_all, 'all', be_all, 'all'),
              ('top-40 (own ROI)', sorted(roi_hg, key=lambda c: -dHG[c]), 'top40',
               sorted(roi_b, key=lambda c: dB[c]), 'top40b'),
              ('top-5 (own ROI)', sorted(roi_hg[:5], key=lambda c: -dHG[c]), 'top5',
               sorted(roi_b[:5], key=lambda c: dB[c]), 'top5b')]
    evs = {'ColorPaced': [0, 2, 4, 6, 8], 'SelfPaced': [],
           'RandSwitch': [0, 1.8, 3.8, 5.8, 7.8, 9.8, 11.8, 14.0]}[task]
    ecol = {'ColorPaced': C_CUE, 'SelfPaced': None, 'RandSwitch': C_SW}[task]
    tg = br[f'{task}_tg']
    cavgH, cavgB = smooth(br[f'{task}_chanavg_HG']), smooth(br[f'{task}_chanavg_B'])

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 11), gridspec_kw={'width_ratios': [1.05, 1, 1]})
    for row, (lab, hgr, hgk, ber, bek) in enumerate(levels):
        axA, axH, axB = axes[row]
        trH, trB = smooth(br[f'{task}_tr_{hgk}_HG']), smooth(br[f'{task}_tr_{bek}_B'])
        for tr, m_color, name in [(trH, C_HG, 'HG (HG ROI)'), (trB, C_BE, '\u03b2 (\u03b2 ROI)')]:
            n = tr.shape[0]; mu = np.nanmean(tr, 0)
            ci = sps.t.ppf(0.975, n - 1) * np.nanstd(tr, 0, ddof=1) / np.sqrt(n)
            axA.fill_between(tg, mu - ci, mu + ci, color=m_color, alpha=0.2, lw=0)
            axA.plot(tg, mu, color=m_color, lw=2, label=name)
        for cx in evs: axA.axvline(cx, color=ecol, ls='--', lw=1.0, alpha=0.7, zorder=1)
        axA.axhline(0, color='0.7', lw=0.6); axA.set_xlim(0, tg[-1])
        axA.set_ylabel(f"{lab}\nenvelope (SD)", fontsize=8.5, fontweight='bold')
        if row == 0:
            axA.set_title("Trial-averaged ROI mean (\u00b195% CI)", fontsize=9.5, fontweight='bold')
            axA.legend(fontsize=7.2, loc='upper right')
        if row == 2: axA.set_xlabel("time from execute onset (s)", fontsize=9)
        for ax, M, rows, lb, ttl in [(axH, cavgH, hgr, 'HG (SD)', 'HG raster — HG ROI (by \u0394HG)'),
                                      (axB, cavgB, ber, '\u03b2 (SD)', '\u03b2 raster — \u03b2 ROI (by \u0394\u03b2)')]:
            im = ax.imshow(M[rows], aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6,
                           extent=[tg[0], tg[-1], len(rows), 0], interpolation='nearest')
            for cx in evs: ax.axvline(cx, color='k', ls='--', lw=0.8, alpha=0.55)
            if len(rows) <= 5:
                ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([f"ch {c}" for c in rows], fontsize=7)
            else:
                ax.set_ylabel(f"channel (n={len(rows)})", fontsize=7.5)
            if row == 0: ax.set_title(ttl, fontsize=9.0, fontweight='bold')
            if row == 2: ax.set_xlabel("time from execute onset (s)", fontsize=9)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label(lb, fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"{TASK_LABEL[task]} — HG and \u03b2 each shown on its OWN ROI (B6)", fontsize=11.5, y=0.998)
    fig.tight_layout(rect=[0, 0.015, 1, 0.975])
    fig.savefig(os.path.join(OUT_DIR, out), dpi=165, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# FIGURE 6 — ROI grid layout: HG ROI vs beta ROI on ChMapB2
# ================================================================================
def fig_roi_grid(out='ol_mi_roi_grid_hg_beta.png'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    npz, *_ = load_env()
    acc_dHG, acc_dB = npz['acc_dHG'], npz['acc_dB']
    roi_hg = np.argsort(acc_dHG)[::-1][:40]                            # top-40 HG increase
    roi_b = np.argsort(acc_dB)[:40]                                    # top-40 beta decrease

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.2))
    panels = [('HG ROI', acc_dHG, roi_hg, 0.44, C_HG, '\u0394HG (execute - rest, SD)'),
              ('\u03b2 ROI', acc_dB, roi_b, 0.69, C_BE, '\u0394\u03b2 (execute - rest, SD)')]
    for ax, (name, vec, roi, vm, ecol, clab) in zip(axes, panels):
        G = to_grid(vec)
        im = ax.imshow(G, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='equal')
        for r in range(GRID_ROWS):                                    # off-grid (NaN) cells -> hatch
            for c in range(GRID_COLS):
                if np.isnan(G[r, c]):
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, fc='0.85', ec='0.6', hatch='//////', lw=0.4))
        for ch in roi:
            r, c = CH2RC[ch]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec='k', lw=1.6))
        for i, ch in enumerate(roi[:5], 1):
            r, c = CH2RC[ch]
            ax.scatter(c, r, marker='*', s=240, c='yellow', edgecolors='k', lw=0.8, zorder=6)
            ax.text(c, r - 0.02, str(i), fontsize=6.5, ha='center', va='center', zorder=7, fontweight='bold')
        ax.set_title(f"{name} — 40-channel ROI (black outline), top-5 *", fontsize=10.5, fontweight='bold', color=ecol)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("ChMapB2 grid (11x23)", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label(clab, fontsize=8); cb.ax.tick_params(labelsize=7)
        lab = "\n".join(f"*{i} ch{ch} ({vec[ch]:+.2f})" for i, ch in enumerate(roi[:5], 1))
        ax.text(1.16, 0.5, lab, transform=ax.transAxes, fontsize=7.5, va='center', ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='#fffef5', ec='0.6', lw=0.5))
    fig.suptitle("ROI grid layout on the ChMapB2 array — HG and \u03b2 ROIs occupy largely different "
                 "electrodes (B6)", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT_DIR, out), dpi=175, bbox_inches='tight')
    plt.close(fig)


# ================================================================================
# FIGURE 7 — per-trial individual-pace detection (arrhythmia test)
# ================================================================================
def fig_pace_detection(out='ol_mi_selfpaced_pace_detection.png', seed=42, nsurr=500):
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, detrend, get_window, lfilter
    from scipy.fft import rfft, rfftfreq
    from scipy.ndimage import median_filter
    npz, labels, META, files = load_env()
    FS, BAND = 100.0, (0.2, 1.5)
    rng = np.random.default_rng(seed)

    def exec_env(f):
        g, y, sg = npz[f"{f['idx']}_grid"], npz[f"{f['idx']}_hg_roi"], f"{f['idx']}_state_grid"
        m = (npz[sg] > 2.5) & (npz[sg] < 3.5) if sg in npz else (g >= 0) & (g <= f['exec_end_rel'])
        return y[m]

    def whiten(P2d, win):                                             # divide out 1/f background in log space
        logP = np.log(P2d + 1e-12)
        return np.exp(logP - median_filter(logP, size=(1, win)))

    def metrics(y):
        y = smooth(y); y = y[np.isfinite(y)]; n = len(y)
        if n < int(3 * FS): return None
        yd = detrend(y, type='linear'); nfft = int(2 ** np.ceil(np.log2(n * 4)))
        f = rfftfreq(nfft, 1 / FS); fm = (f >= BAND[0]) & (f <= BAND[1]); fb = f[fm]
        w = get_window('hann', n); win = max(5, int(0.5 / (FS / nfft)))
        Pw = whiten((np.abs(rfft(yd * w, nfft)) ** 2)[None, :], win)[0][fm]
        peak_f, peakiness = fb[np.argmax(Pw)], Pw.max()
        a = np.clip(np.corrcoef(yd[:-1], yd[1:])[0, 1], -0.999, 0.999)      # AR(1) surrogate null
        s0 = np.std(yd) * np.sqrt(1 - a ** 2)
        S = lfilter([1.0], [1.0, -a], rng.standard_normal((nsurr, n)) * s0, axis=1)
        nullmax = whiten(np.abs(rfft(S * w, nfft, axis=1)) ** 2, win)[:, fm].max(1)
        p = (np.sum(nullmax >= peakiness) + 1) / (nsurr + 1)
        pk, _ = find_peaks(y, prominence=0.25, distance=int(0.7 * FS))
        ibis = np.diff(pk) / FS if len(pk) >= 2 else np.array([])
        return dict(peak_f=peak_f, peakiness=peakiness, sig=p < 0.05, n_bursts=len(pk),
                    ibi_cv=(ibis.std() / ibis.mean()) if len(ibis) >= 2 else np.nan,
                    t=np.arange(n) / FS, y=y, pk=pk)

    recs = {t: [f for f in files if f['task'] == t] for t in TASK_ORDER}
    RES = {t: [m for f in recs[t] if (m := metrics(exec_env(f)))] for t in TASK_ORDER}
    fg = np.arange(0.2, 1.501, 0.01)

    def whitened_on_grid(y):
        y = smooth(y); y = y[np.isfinite(y)]; n = len(y); yd = detrend(y, type='linear')
        nfft = int(2 ** np.ceil(np.log2(n * 4))); f = rfftfreq(nfft, 1 / FS)
        w = get_window('hann', n); win = max(5, int(0.5 / (FS / nfft)))
        Pw = whiten((np.abs(rfft(yd * w, nfft)) ** 2)[None, :], win)[0]
        return np.interp(fg, f, Pw)
    SPEC = {t: np.vstack([whitened_on_grid(exec_env(f)) for f in recs[t]]) for t in TASK_ORDER}

    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.42, wspace=0.28)
    axS = fig.add_subplot(gs[0, :])                                   # per-trial peak-frequency strip
    for i, t in enumerate(TASK_ORDER):
        o = RES[t]; y0 = len(TASK_ORDER) - 1 - i
        for m, jj in zip(o, (rng.random(len(o)) - 0.5) * 0.5):
            axS.scatter(m['peak_f'], y0 + jj, s=70, c=(TASK_COL[t] if m['sig'] else 'white'),
                        edgecolors=TASK_COL[t], lw=1.4, alpha=0.9, zorder=3)
        pf = np.array([m['peak_f'] for m in o]); frac = 100 * np.mean([m['sig'] for m in o])
        axS.plot([np.median(pf)] * 2, [y0 - 0.32, y0 + 0.32], color='k', lw=2.4, zorder=4)
        axS.text(1.62, y0, f"{TASK_LABEL[t]}\nmed {np.median(pf):.2f} Hz \u00b7 {frac:.0f}% sig",
                 fontsize=9, va='center', color=TASK_COL[t], fontweight='bold')
    axS.axvline(0.5, color='0.5', ls=':', lw=1.5); axS.text(0.5, 2.65, '0.5 Hz (2 s)', fontsize=8, color='0.4', ha='center')
    axS.set_xlim(0.15, 1.55); axS.set_ylim(-0.6, 2.75); axS.set_yticks([])
    axS.set_xlabel("per-trial dominant frequency in 0.2-1.5 Hz (Hz)", fontsize=10)
    axS.set_title("Per-trial pace — each point is one trial's dominant HG-envelope frequency "
                  "(filled = significant vs AR(1) null, p<0.05)", fontsize=10.5, fontweight='bold')
    axS.spines['left'].set_visible(False)
    for j, t in enumerate(TASK_ORDER):                                # whitened spectra heatmaps
        ax = fig.add_subplot(gs[1, j]); M = SPEC[t]
        im = ax.imshow(M, aspect='auto', cmap='magma', vmin=0, vmax=np.percentile(SPEC['RandSwitch'], 99),
                       extent=[fg[0], fg[-1], M.shape[0], 0], interpolation='nearest')
        ax.axvline(0.5, color='cyan', ls=':', lw=1.2)
        ax.set_title(f"{t} — whitened spectra (trials x freq)", fontsize=9, fontweight='bold', color=TASK_COL[t])
        ax.set_xlabel("frequency (Hz)", fontsize=8.5); ax.set_ylabel("trial #", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("peak/bg", fontsize=7)
    for j, t in enumerate(TASK_ORDER):                                # best-case example trial
        ax = fig.add_subplot(gs[2, j]); m = max(RES[t], key=lambda d: d['peakiness'])
        ax.plot(m['t'], m['y'], color=TASK_COL[t], lw=1.4)
        ax.plot(m['t'][m['pk']], m['y'][m['pk']], 'v', color='k', ms=7, zorder=5)
        for x in m['t'][m['pk']]: ax.axvline(x, color='0.7', lw=0.6, zorder=0)
        ax.set_title(f"{t} — best-case trial (peakiness {m['peakiness']:.1f}, {'sig' if m['sig'] else 'n.s.'})",
                     fontsize=8.5, fontweight='bold', color=TASK_COL[t])
        ax.set_xlabel("time from execute onset (s)", fontsize=8.5)
        if j == 0: ax.set_ylabel("HG ROI env (SD)", fontsize=8.5)
        txt = f"{m['n_bursts']} bursts \u00b7 IBI CV {m['ibi_cv']:.2f}" if np.isfinite(m['ibi_cv']) else f"{m['n_bursts']} bursts"
        ax.text(0.98, 0.03, txt, transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fffef5', ec='0.6', lw=0.5))
    fig.suptitle("Per-trial individual-pace detection — is SelfPaced arrhythmic, or rhythmic at a "
                 "variable pace? (B6 HG, motor ROI)", fontsize=12.5, y=0.965)
    fig.savefig(os.path.join(OUT_DIR, out), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    """Render every figure whose input data is present in DATA_DIR; skip the rest with a note."""
    apply_style()
    figures = [
        ('channel-selection grid', fig_channel_grid),
        ('ROI grid',               fig_roi_grid),
        ('pace detection',         fig_pace_detection),
        ('single-channel',         fig_single_channel),
        ('own-ROI (ColorPaced)',   lambda: fig_own_roi('ColorPaced')),
        ('own-ROI (SelfPaced)',    lambda: fig_own_roi('SelfPaced')),
        ('own-ROI (RandSwitch)',   lambda: fig_own_roi('RandSwitch')),
    ]
    for name, fn in figures:
        try:
            fn(); print(f"ok:   {name}")
        except FileNotFoundError as e:
            print(f"skip: {name} — {e}")
        except Exception as e:
            print(f"FAIL: {name} — {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
