"""Noise-robust audio landmark fingerprints, for excluding Phase-2 test recordings from training data.
Peaks of the log spectrogram, paired into (f1, df, dt) hashes; a match = many hashes agreeing on one time offset."""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft

SR = 8000


def _spec(x, sr):
    x = np.asarray(x, np.float32)
    if sr == 16000: x = x[::2]
    _, _, Z = stft(x, SR, nperseg=512, noverlap=384)
    return np.log(np.abs(Z) + 1e-6)


def landmarks(x, sr=16000, fan=5, dtmax=32, dfmax=40):
    S = _spec(x, sr)
    peaks = (S == maximum_filter(S, size=(15, 9))) & (S > np.median(S) + 2.0)
    f, t = np.nonzero(peaks)
    o = np.argsort(t); f, t = f[o], t[o]
    out = []
    for i in range(len(t)):
        n = 0
        for j in range(i+1, len(t)):
            dt = t[j]-t[i]
            if dt > dtmax: break
            if dt < 1 or abs(int(f[j])-int(f[i])) > dfmax: continue
            out.append(((int(f[i]) << 16) | ((int(f[j])-int(f[i])+64) << 8) | dt, int(t[i])))
            n += 1
            if n >= fan: break
    return out


class Index:
    def __init__(self):
        self.h = {}

    def add(self, key, x, sr=16000):
        for hsh, t in landmarks(x, sr):
            self.h.setdefault(hsh, []).append((key, t))

    def query(self, x, sr=16000):
        lm = landmarks(x, sr); votes = {}
        for hsh, t in lm:
            for key, tt in self.h.get(hsh, ()):
                k = (key, tt-t); votes[k] = votes.get(k, 0)+1
        if not votes: return None, 0, len(lm)
        (key, _), c = max(votes.items(), key=lambda kv: kv[1])
        return key, c, len(lm)
