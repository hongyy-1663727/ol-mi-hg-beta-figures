#!/usr/bin/env python3
"""
ColorPaced per-action breakdown — all 7 imagined actions, HG and beta each on its own ROI.
Subject B6, session 20260708, task OL_MI_ColorPaced (256-ch ECoG @ 1 kHz).

Three stages, split by where they run and how much data they touch:

    extract_envelopes()   SERVER   raw .mat  ->  per-trial per-channel 100 Hz envelopes
    reduce_per_action()   SERVER   envelopes ->  per-action trial-mean ROI arrays (~2 MB)
    plot_per_action()     LOCAL    reduced npz -> the 7-column x 3-row figure

The raw recordings (~700 MB/run) live on the recording server, so the two extraction
stages run there and only the small reduced npz is copied back for plotting. Re-running
the figure needs `plot_per_action()` alone; the reduced npz is the saved artifact.

ROI_HG / ROI_B are the frozen top-40 channels (by |execute - rest| HG increase / beta
decrease), defined once over the sustained tasks and reused unchanged here.
"""
import os, json
import numpy as np
from scipy.ndimage import uniform_filter1d

# ---- acquisition / preprocessing constants -------------------------------------
Fs        = 1000.0            # broadband sample rate (Hz)
HG_BAND   = (70.0, 150.0)     # high-gamma passband (Hz)
BETA_BAND = (13.0, 30.0)      # beta passband (Hz)
NOTCH_HZ  = (60, 120, 180)    # line noise + harmonics
GRID_FS   = 100.0             # analysis grid after envelope extraction (Hz)
PAD       = 1000             # reflect-pad length (samples) to kill filter/Hilbert edge transients
SMOOTH_W  = 20               # display smoothing window = 200 ms on the 100 Hz grid (samples)

# ---- labels / ROI / style ------------------------------------------------------
ACTIONS = {1: 'Right thumb', 2: 'Both legs', 3: 'Left thumb', 4: 'Head',
           5: 'Lips', 6: 'Tongue', 7: 'Both middle fingers'}

ROI_HG = [85, 71, 198, 75, 58, 194, 40, 207, 37, 108, 206, 55, 65, 248, 243, 252, 210, 116,
          202, 84, 215, 214, 81, 44, 61, 211, 238, 69, 197, 121, 204, 213, 82, 193, 169, 126,
          151, 154, 147, 199]
ROI_B  = [191, 3, 131, 31, 7, 9, 1, 13, 188, 172, 168, 18, 129, 154, 22, 137, 176, 24, 30, 15,
          20, 185, 177, 135, 27, 143, 151, 165, 173, 181, 148, 180, 36, 21, 108, 141, 43, 245,
          39, 113]

C_HG, C_BE, C_CUE = '#c0392b', '#2471a3', '#16a085'   # HG red, beta blue, cue green
CUES  = (2, 4, 6, 8)          # 0.5 Hz color-cue times within the 10 s hold (s)
VMAX  = 0.6                   # raster color scale (SD), matches the aggregate figure


def smooth(y, axis=-1):
    """200 ms moving average on the 100 Hz grid."""
    return uniform_filter1d(np.asarray(y, float), SMOOTH_W, axis=axis, mode='nearest')


# ================================================================================
# STAGE 1 — per-file band-envelope extraction                             (SERVER)
# ================================================================================
def extract_envelopes(root, out_npz, out_meta):
    """Broadband .mat -> HG/beta envelopes, z-scored to rest, on an execute-locked 100 Hz grid.

    Writes, per trial `idx`:  <idx>_hg_roi (40,T), <idx>_b_roi (40,T), <idx>_t (T,)
    plus per-trial {idx, target, file} metadata to `out_meta`.
    """
    import glob
    import h5py
    from scipy.signal import butter, filtfilt, hilbert, iirnotch

    bp_hg = butter(4, [f / (Fs / 2) for f in HG_BAND], btype='band')
    bp_be = butter(4, [f / (Fs / 2) for f in BETA_BAND], btype='band')
    notches = [iirnotch(f0 / (Fs / 2), Q=30) for f0 in NOTCH_HZ]

    def load_file(fp):
        with h5py.File(fp, 'r') as f:
            td = f['TrialData']
            ts   = np.array(td['TaskState']).ravel().astype(int)   # per-bin: 1 rest / 3 execute / 4 ITI
            Time = np.array(td['Time']).ravel()                    # per-bin end timestamp
            bb = td['BroadbandData']
            bins = [np.array(f[bb[i, 0]]) for i in range(bb.shape[0])]  # each (256, n_i)
            ni = np.array([b.shape[1] for b in bins])
            X = np.concatenate(bins, axis=1)                       # (256, total samples)
            target = int(np.array(td['TargetID']).ravel()[0])
        # expand per-bin timestamps and TaskState to per-sample
        tsamp = np.empty(X.shape[1]); smask = np.zeros(X.shape[1], int); pos = 0
        for i, n in enumerate(ni):
            tsamp[pos:pos + n] = Time[i] - (n - 1 - np.arange(n)) / Fs
            smask[pos:pos + n] = ts[i]
            pos += n
        return dict(X=X, tsamp=tsamp, smask=smask, ts=ts, Time=Time, ni=ni, target=target)

    def envelopes(d):
        """Common-median reference -> notch -> band Hilbert amplitude -> z-score to rest."""
        Xc = d['X'] - np.median(d['X'], axis=0, keepdims=True)
        Xp = np.pad(Xc, ((0, 0), (PAD, PAD)), mode='reflect')
        for b, a in notches:
            Xp = filtfilt(b, a, Xp, axis=-1)
        rest = d['smask'] == 1

        def band(coef):
            b, a = coef
            env = np.abs(hilbert(filtfilt(b, a, Xp, axis=-1), axis=-1))[:, PAD:-PAD]
            mu = env[:, rest].mean(1, keepdims=True)
            sd = env[:, rest].std(1, keepdims=True) + 1e-9
            return (env - mu) / sd

        return band(bp_hg), band(bp_be)

    files = [fp
             for run in sorted(p for p in glob.glob(os.path.join(root, '*')) if os.path.isdir(p))
             for fp in sorted(glob.glob(os.path.join(run, '**', 'Data*.mat'), recursive=True))]

    out, meta = {}, []
    for idx, fp in enumerate(files):
        d = load_file(fp)
        hgz, bez = envelopes(d)
        exec_bins = np.where(d['ts'] == 3)[0]
        if exec_bins.size == 0:
            continue
        g0 = d['Time'][exec_bins[0]] - (d['ni'][exec_bins[0]] - 1) / Fs   # execute-window start
        g1 = d['Time'][exec_bins[-1]]                                     # execute-window end
        grid = np.arange(g0, g1, 1.0 / GRID_FS)

        def on_grid(env, channels):                                       # (len(channels), T)
            return np.vstack([np.interp(grid, d['tsamp'], env[c]) for c in channels]).astype('float32')

        out[f'{idx}_hg_roi'] = on_grid(hgz, ROI_HG)
        out[f'{idx}_b_roi']  = on_grid(bez, ROI_B)
        out[f'{idx}_t']      = (grid - g0).astype('float32')
        meta.append(dict(idx=idx, target=d['target'], file=os.path.basename(fp)))

    np.savez_compressed(out_npz, **out)
    json.dump(meta, open(out_meta, 'w'))


# ================================================================================
# STAGE 2 — trial-average per target action                               (SERVER)
# ================================================================================
def reduce_per_action(in_npz, in_meta, out_npz):
    """Group trials by target action, trial-average each ROI array; write the small figure npz.

    Writes:  t (T,)  and per action `tg`:  <tg>_hg_roi (40,T), <tg>_b_roi (40,T), <tg>_ntr (1,)
    """
    d = np.load(in_npz)
    meta = json.load(open(in_meta))

    trials_by_action = {}
    for r in meta:
        trials_by_action.setdefault(r['target'], []).append(r['idx'])

    tmin = min(d[f"{r['idx']}_t"].shape[0] for r in meta)     # crop all trials to the shortest
    out = {'t': d[f"{meta[0]['idx']}_t"][:tmin].astype('float32')}
    for tg, idxs in trials_by_action.items():
        hg = np.stack([d[f'{i}_hg_roi'][:, :tmin] for i in idxs]).mean(0)
        be = np.stack([d[f'{i}_b_roi'][:, :tmin] for i in idxs]).mean(0)
        out[f'{tg}_hg_roi'] = hg.astype('float32')
        out[f'{tg}_b_roi']  = be.astype('float32')
        out[f'{tg}_ntr']    = np.array([len(idxs)])
    np.savez_compressed(out_npz, **out)


# ================================================================================
# STAGE 3 — the figure                                                     (LOCAL)
# ================================================================================
def plot_per_action(fig_npz, out_png):
    """7 actions (columns) x [ROI-mean trace / HG raster / beta raster] (rows)."""
    _rc = '/usr/share/matplotlib/mpl-data/matplotlibrc'      # sandbox-only workaround; skipped
    if os.path.exists(_rc):                                  # where absent (normal conda matplotlib)
        os.environ.setdefault('MATPLOTLIBRC', _rc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fg = np.load(fig_npz)
    t = fg['t']
    actions = sorted(ACTIONS)

    def raster(ax, M):
        """Channel x time heatmap of one ROI, smoothed; returns the image for the colorbar."""
        im = ax.imshow(smooth(M, axis=1), aspect='auto', cmap='RdBu_r', vmin=-VMAX, vmax=VMAX,
                       extent=[0, 10, M.shape[0], 0], interpolation='nearest')
        for cu in CUES:
            ax.axvline(cu, color='k', lw=0.5, ls=':', alpha=0.4)
        ax.set_xlim(0, 10)
        return im

    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(3, 7, hspace=0.22, wspace=0.16, height_ratios=[1.15, 1, 1])
    im = None
    for j, tg in enumerate(actions):
        hg, be, ntr = fg[f'{tg}_hg_roi'], fg[f'{tg}_b_roi'], fg[f'{tg}_ntr'][0]

        # row 0 — top-40 ROI-mean trace: HG on HG-ROI, beta on beta-ROI
        ax = fig.add_subplot(gs[0, j])
        ax.plot(t, smooth(hg.mean(0)), color=C_HG, lw=1.9, label='HG (HG ROI)')
        ax.plot(t, smooth(be.mean(0)), color=C_BE, lw=1.9, label='\u03b2 (\u03b2 ROI)')
        ax.axhline(0, color='0.6', lw=0.6)
        for cu in CUES:
            ax.axvline(cu, color=C_CUE, lw=0.8, ls='--', alpha=0.6)
        ax.set(xlim=(0, 10), ylim=(-1.0, 1.0))
        ax.set_title(f"{tg}. {ACTIONS[tg]}\n(n={ntr} trials)", fontsize=9.5, fontweight='bold')
        ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel("top-40 ROI mean\nenvelope (SD, 200 ms)", fontsize=9)
            ax.legend(fontsize=6.5, frameon=False, loc='lower right')
        else:
            ax.set_yticklabels([])

        # row 1 — HG raster over the 40 HG-ROI channels (sorted by dHG)
        ax = fig.add_subplot(gs[1, j])
        raster(ax, hg)
        ax.set_xticklabels([])
        ax.set_ylabel("HG raster\nHG-ROI ch (by \u0394HG)", fontsize=9) if j == 0 else ax.set_yticklabels([])

        # row 2 — beta raster over the 40 beta-ROI channels (sorted by dbeta)
        ax = fig.add_subplot(gs[2, j])
        im = raster(ax, be)
        ax.set_xlabel("time (s)", fontsize=8.5)
        ax.set_ylabel("\u03b2 raster\n\u03b2-ROI ch (by \u0394\u03b2)", fontsize=9) if j == 0 else ax.set_yticklabels([])

    fig.colorbar(im, cax=fig.add_axes([0.995, 0.12, 0.008, 0.5])).set_label("envelope (SD)", fontsize=8)
    fig.suptitle("ColorPaced (external cue) \u2014 per-action breakdown of all 7 imagined "
                 "actions, HG and \u03b2 each on its own ROI  (B6)", fontsize=13, y=0.99)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    root       = '/mnt/DataDrive/B6Data/20260708/OL_MI_ColorPaced'   # raw .mat (server only)
    trials_npz = 'cp_peraction.npz'
    trials_meta = 'cp_peraction_meta.json'
    fig_npz    = 'data/ol_mi_cp_pa_fig.npz'                        # the saved ~2 MB artifact
    out_png    = 'ol_mi_colorpaced_per_action.png'

    # Stages 1-2 need the raw recordings on the server; uncomment to run there.
    # extract_envelopes(root, trials_npz, trials_meta)
    # reduce_per_action(trials_npz, trials_meta, fig_npz)
    plot_per_action(fig_npz, out_png)


if __name__ == '__main__':
    main()
