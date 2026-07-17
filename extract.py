#!/usr/bin/env python3
"""
extract.py -- raw .mat  ->  analysis checkpoints  (server-side stage of the pipeline)

Reads the raw B6 OL_MI recordings (~700 MB/run, on the acquisition server) and
produces the three checkpoint files that `ol_mi_figures.py` consumes:

    ol_mi_env.npz  + ol_mi_labels.json   -> channel-selection grid, ROI grid, pace detection
    ol_mi_ch5.npz  + ol_mi_ch5.json      -> single-trial single-channel figure
    ol_mi_bothroi.npz                    -> per-task own-ROI channel-level figures

Envelope convention (LOCKED -- the group's standard ECoG pipeline; do not alter):
    common-median reference (median across channels)
    -> 1 s reflect-pad (kills filter/Hilbert edge transients), cropped after
    -> 60/120/180 Hz notch (IIR, Q=30)
    -> 70-150 Hz (HG) / 13-30 Hz (beta) 4th-order Butterworth band-pass
    -> Hilbert analytic-amplitude envelope
    -> z-score to that trial's rest baseline (TaskState == 1)
Downstream, envelopes are resampled to a 100 Hz grid (GRID_FS); the ROI is the
40 channels with the largest execute-rest HG increase / beta decrease, computed
here over the two sustained-action tasks and written into ol_mi_labels.json.

Usage (on the server, needs h5py + the raw .mat under --root):
    python extract.py --root /mnt/DataDrive/B6Data/20260708 --out data

Then `python ol_mi_figures.py` renders six of the seven figures from data/ with
no raw data. The seventh (ColorPaced per-action) has its own self-contained
raw->figure pipeline in ol_mi_colorpaced_per_action.py.
"""
import os
import glob
import json
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch

# ------------------------------------------------------------------ LOCKED core
Fs      = 1000.0
HG      = (70.0, 150.0)
BETA    = (13.0, 30.0)
GRID_FS = 100.0
ROI_K   = 40
PAD     = 1000                     # 1 s reflect padding (samples)
OFFGRID = (107, 112, 117)          # ref/gnd electrodes, not on the ChMapB2 grid (0-indexed)
TASKS   = ['ColorPaced', 'RandSwitch', 'SelfPaced']   # extraction order (fixes trial indexing)
EXEMPLAR_TARGET = 6                # Tongue -- the single-channel figure's example action


def _bp(lo, hi, order=4):
    return butter(order, [lo / (Fs / 2), hi / (Fs / 2)], btype='band')


BHG   = _bp(*HG)
BBE   = _bp(*BETA)
NOTCH = [iirnotch(f0 / (Fs / 2), Q=30) for f0 in (60, 120, 180)]


def load_file(fp):
    """Read one Data*.mat: concatenate the broadband bins into a (256, nsamp) array
    with a per-sample time vector (tsamp) and per-sample TaskState mask (smask)."""
    import h5py
    with h5py.File(fp, 'r') as f:
        td = f['TrialData']
        ts = np.array(td['TaskState']).ravel().astype(int)
        Time = np.array(td['Time']).ravel()
        bb = td['BroadbandData']
        nbins = bb.shape[0]
        bins = [np.array(f[bb[i, 0]]) for i in range(nbins)]
        ni = np.array([b.shape[1] for b in bins])
        X = np.concatenate(bins, axis=1)
        tg = np.array(td['TargetID']).ravel()
        co = np.array(td['color_onsets']).ravel() if 'color_onsets' in td else None
    tot = X.shape[1]
    tsamp = np.empty(tot); pos = 0
    for i in range(nbins):
        n = ni[i]; tsamp[pos:pos + n] = Time[i] - (n - 1 - np.arange(n)) / Fs; pos += n
    smask = np.zeros(tot, dtype=int); pos = 0
    for i in range(nbins):
        smask[pos:pos + ni[i]] = ts[i]; pos += ni[i]
    return dict(X=X, tsamp=tsamp, smask=smask, ts=ts, Time=Time, ni=ni, tg=tg, co=co)


def envelopes(d):
    """LOCKED: CAR -> reflect-pad -> notch -> HG/beta Butterworth -> Hilbert amplitude
    -> crop pad -> z-score to rest.  Returns (hg_z, beta_z, rest_mask), both (256, nsamp)."""
    Xc = d['X'] - np.median(d['X'], axis=0, keepdims=True)          # common-median reference
    Xp = np.pad(Xc, ((0, 0), (PAD, PAD)), mode='reflect')           # reflect-pad BEFORE filtering
    for b, a in NOTCH:
        Xp = filtfilt(b, a, Xp, axis=-1)

    def env(coef):
        b, a = coef
        e = np.abs(hilbert(filtfilt(b, a, Xp, axis=-1), axis=-1))
        return e[:, PAD:-PAD]                                        # crop the padding

    hg = env(BHG); be = env(BBE)
    rm = d['smask'] == 1
    def zsc(E):
        mu = E[:, rm].mean(1, keepdims=True); sd = E[:, rm].std(1, keepdims=True) + 1e-9
        return (E - mu) / sd
    return zsc(hg), zsc(be), rm


def dmod(Ez, smask):
    """Per-channel execute - rest response (SD units)."""
    return Ez[:, smask == 3].mean(1) - Ez[:, smask == 1].mean(1)


def to_grid(sig, t, g0, g1):
    grid = np.arange(g0, g1, 1.0 / GRID_FS)
    return grid, np.interp(grid, t, sig)


def iter_files(root):
    """(task, run, path) for every Data*.mat, in the LOCKED task order."""
    files = []
    for task in TASKS:
        dname = 'OL_MI_' + task
        for run in sorted(g for g in glob.glob(os.path.join(root, dname, '*')) if os.path.isdir(g)):
            for fp in sorted(glob.glob(os.path.join(run, '**', 'Data*.mat'), recursive=True)):
                files.append((task, os.path.basename(run), fp))
    return files


# ------------------------------------------------------ 1) env + labels checkpoint
def build_env_checkpoint(root, out_dir):
    """Two passes over every trial: (1) define the ROI from the execute-rest maps of
    the two sustained tasks; (2) write ROI-mean / grand-mean envelopes on the 100 Hz
    grid, per-channel maps, and per-trial metadata.  Returns (roi_hg, roi_b)."""
    files = iter_files(root)
    print('total files', len(files), flush=True)

    # PASS 1 -- per-channel execute-rest maps, ROI from sustained tasks only
    per_ch = {}; acc_hg = []; acc_b = []
    for idx, (task, run, fp) in enumerate(files):
        d = load_file(fp); hgz, bez, _ = envelopes(d)
        dhg = dmod(hgz, d['smask']); db = dmod(bez, d['smask'])
        per_ch[fp] = (dhg.astype('float32'), db.astype('float32'))
        if task in ('ColorPaced', 'SelfPaced'):
            acc_hg.append(dhg); acc_b.append(db)
    acc_hg = np.mean(acc_hg, 0); acc_b = np.mean(acc_b, 0)
    roi_hg = np.argsort(acc_hg)[::-1][:ROI_K]
    roi_b = np.argsort(acc_b)[:ROI_K]
    print('ROI HG meanDHG', round(float(acc_hg[roi_hg].mean()), 3),
          'beta meanDB', round(float(acc_b[roi_b].mean()), 3), flush=True)

    # PASS 2 -- reduced envelopes on the 100 Hz grid, exec-locked
    meta = dict(Fs=Fs, HG=list(HG), BETA=list(BETA), GRID_FS=GRID_FS, ROI_K=int(ROI_K),
                roi_hg=roi_hg.astype(int).tolist(), roi_b=roi_b.astype(int).tolist(),
                notch=[60, 120, 180], car='median', pad=PAD)
    recs = []; npz = {'acc_dHG': acc_hg.astype('float32'), 'acc_dB': acc_b.astype('float32')}
    for task, run, fp in files:
        d = load_file(fp); hgz, bez, _ = envelopes(d)
        ts = d['ts']; Time = d['Time']; smask = d['smask']; tsamp = d['tsamp']
        exec_bins = np.where(ts == 3)[0]
        if exec_bins.size == 0:
            continue
        t_es = Time[exec_bins.min()]; t_ee = Time[exec_bins.max()]
        trel = tsamp - t_es
        g0, g1 = trel.min(), trel.max()
        grid, hgg = to_grid(hgz.mean(0), trel, g0, g1)
        _, hgr = to_grid(hgz[roi_hg].mean(0), trel, g0, g1)
        _, bgg = to_grid(bez.mean(0), trel, g0, g1)
        _, brr = to_grid(bez[roi_b].mean(0), trel, g0, g1)
        sg = np.interp(grid, trel, smask.astype(float))
        i = len(recs)
        rec = dict(idx=i, task=task, run=run, file=os.path.basename(fp),
                   target=(int(round(d['tg'][0])) if d['tg'].size == 1 else None),
                   exec_end_rel=float(t_ee - t_es))
        npz[f'{i}_grid'] = grid.astype('float32')
        npz[f'{i}_hg_grand'] = hgg.astype('float32'); npz[f'{i}_hg_roi'] = hgr.astype('float32')
        npz[f'{i}_b_grand'] = bgg.astype('float32'); npz[f'{i}_b_roi'] = brr.astype('float32')
        npz[f'{i}_state_grid'] = sg.astype('float32')
        npz[f'{i}_dHG'] = per_ch[fp][0]; npz[f'{i}_dB'] = per_ch[fp][1]
        if d['co'] is not None:
            rec['cue_rel'] = (d['co'] - t_es).astype(float).tolist()
        if d['tg'].size > 1:
            tg = np.round(d['tg']).astype(int); chg = np.where(np.diff(tg) != 0)[0] + 1
            rec['switch_rel'] = [float(Time[b] - t_es) for b in chg]
            rec['switch_target'] = tg[chg].astype(int).tolist(); rec['tgt_seq'] = tg.astype(int).tolist()
        recs.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(meta=meta, files=recs), open(os.path.join(out_dir, 'ol_mi_labels.json'), 'w'))
    np.savez_compressed(os.path.join(out_dir, 'ol_mi_env.npz'), **npz)
    print('SAVED ol_mi_env.npz  (%d trials)' % len(recs), flush=True)
    return roi_hg.astype(int).tolist(), roi_b.astype(int).tolist()


# --------------------------------------------------------- 2) single-channel (ch5)
def build_ch5(root, out_dir, pick5, target=EXEMPLAR_TARGET):
    """One exemplar trial per task (ColorPaced/SelfPaced = first `target` trial;
    RandSwitch = first trial), five channels (the top-5 HG ROI), at native 1 kHz."""
    files = iter_files(root)
    chosen = {}
    for task, run, fp in files:
        if task in chosen:
            continue
        if task == 'RandSwitch':
            chosen[task] = fp
        else:
            d0 = load_file(fp)
            if d0['tg'].size == 1 and int(round(d0['tg'][0])) == target:
                chosen[task] = fp
    npz = {}; meta = {}
    for task in TASKS:
        fp = chosen[task]
        d = load_file(fp); hgz, bez, _ = envelopes(d)
        ts = d['ts']; Time = d['Time']; smask = d['smask']; tsamp = d['tsamp']
        eb = np.where(ts == 3)[0]; t_es = Time[eb.min()]; t_ee = Time[eb.max()]
        trel = tsamp - t_es
        npz[f'{task}_t'] = trel.astype('float32')
        npz[f'{task}_hg'] = hgz[pick5].astype('float32')      # (5, nsamp) at 1 kHz
        npz[f'{task}_be'] = bez[pick5].astype('float32')
        npz[f'{task}_state'] = smask.astype('float32')
        m = dict(task=task, exec_end_rel=float(t_ee - t_es),
                 target=int(round(d['tg'][0])) if d['tg'].size == 1 else None)
        if d['co'] is not None:
            m['cue_rel'] = (d['co'] - t_es).astype(float).tolist()
        meta[task] = m
    np.savez_compressed(os.path.join(out_dir, 'ol_mi_ch5.npz'), **npz)
    json.dump(meta, open(os.path.join(out_dir, 'ol_mi_ch5.json'), 'w'))
    print('SAVED ol_mi_ch5.npz', flush=True)


# ---------------------------------------------------------- 3) own-ROI (bothroi)
def build_bothroi(root, out_dir, roi_hg, roi_b):
    """Per-task channel-average and per-trial ROI-mean envelopes on the 100 Hz grid,
    for both bands each on its own ROI (top-40 / top-5)."""
    ongrid = [c for c in range(256) if c not in OFFGRID]
    spans = {'ColorPaced': 9.6, 'SelfPaced': 9.6, 'RandSwitch': 15.6}
    levels = {'all': ongrid, 'top40': roi_hg, 'top5': roi_hg[:5],
              'top40b': roi_b, 'top5b': roi_b[:5]}
    res = {}
    for task in TASKS:
        span = spans[task]; tg = np.arange(0.0, span, 1.0 / GRID_FS)
        accHG = np.zeros((256, len(tg))); accB = np.zeros((256, len(tg))); ntr = 0
        trHG = {k: [] for k in levels}; trB = {k: [] for k in levels}
        for _, run, fp in [f for f in iter_files(root) if f[0] == task]:
            d = load_file(fp); hgz, bez, _ = envelopes(d); ex = d['smask'] == 3
            if ex.sum() < 1000:
                continue
            th = np.arange(ex.sum()) / Fs
            HGi = np.vstack([np.interp(tg, th, hgz[c, ex], right=np.nan) for c in range(256)])
            BEi = np.vstack([np.interp(tg, th, bez[c, ex], right=np.nan) for c in range(256)])
            m = ~np.isnan(HGi[0]); accHG[:, m] += HGi[:, m]; accB[:, m] += BEi[:, m]; ntr += 1
            for k, chs in levels.items():
                trHG[k].append(HGi[chs].mean(0)); trB[k].append(BEi[chs].mean(0))
        res[f'{task}_chanavg_HG'] = (accHG / ntr).astype('float32')
        res[f'{task}_chanavg_B'] = (accB / ntr).astype('float32')
        res[f'{task}_tg'] = tg.astype('float32'); res[f'{task}_ntr'] = ntr
        for k in levels:
            res[f'{task}_tr_{k}_HG'] = np.array(trHG[k], dtype='float32')
            res[f'{task}_tr_{k}_B'] = np.array(trB[k], dtype='float32')
        print('bothroi', task, 'ntr', ntr, flush=True)
    np.savez_compressed(os.path.join(out_dir, 'ol_mi_bothroi.npz'), **res)
    print('SAVED ol_mi_bothroi.npz', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/mnt/DataDrive/B6Data/20260708',
                    help='directory holding OL_MI_{ColorPaced,SelfPaced,RandSwitch}/')
    ap.add_argument('--out', default='data', help='output directory for the checkpoints')
    args = ap.parse_args()

    roi_hg, roi_b = build_env_checkpoint(args.root, args.out)   # writes env + labels; returns ROI
    build_ch5(args.root, args.out, roi_hg[:5])                  # single-channel figure input
    build_bothroi(args.root, args.out, roi_hg, roi_b)           # own-ROI figures input
    print('done -- all checkpoints written to', args.out, flush=True)


if __name__ == '__main__':
    main()
