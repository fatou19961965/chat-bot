import numpy as np

from app import audio as A


def _signal():
    sr = A.SR
    rng = np.random.default_rng(0)
    sil = 0.003 * rng.standard_normal(sr)          # 1 s de bruit faible
    t = np.arange(int(1.2 * sr)) / sr
    voice = 0.4 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    return np.concatenate([sil, voice, sil, voice, sil]).astype(np.float32)


def test_vad_trouve_deux_segments():
    segs = A.vad_segments(A.preprocess(_signal()))
    assert len(segs) == 2
    assert 0.8 < segs[0].start < 1.1
    assert all(1.0 < s.duration < 1.8 for s in segs)


def test_decoupage_segments_longs():
    sr = A.SR
    t = np.arange(int(50 * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    segs = A.vad_segments(x, max_chunk_s=20)
    assert len(segs) >= 3 and all(s.duration <= 20.01 for s in segs)


def test_aller_retour_wav_ffmpeg():
    wav = A.to_wav_bytes(_signal(), A.SR)
    back = A.decode_audio(wav)
    assert abs(len(back) - len(_signal())) < 100


# --- Robustesse au bruit ------------------------------------------------------
def _voix_dans_le_bruit(snr_bruit=0.08):
    sr = A.SR
    rng = np.random.default_rng(1)
    t = np.arange(int(1.2 * sr)) / sr
    # « voix » : harmoniques de 150 Hz modulées (syllabes)
    voice = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12))
    voice = 0.3 * voice * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    sil = np.zeros(sr)
    x = np.concatenate([sil, voice, sil, voice, sil])
    hum = 0.15 * np.sin(2 * np.pi * 50 * np.arange(len(x)) / sr)       # ronflement secteur
    return (x + snr_bruit * rng.standard_normal(len(x)) + hum).astype(np.float32)


def test_debruitage_reduit_le_bruit_des_pauses():
    x = _voix_dans_le_bruit()
    y = A.preprocess(x, denoise_strength=1.5)
    x0 = A.preprocess(x)
    pause = slice(int(0.2 * A.SR), int(0.8 * A.SR))
    gain_db = 10 * np.log10(np.mean(x0[pause] ** 2) / np.mean(y[pause] ** 2))
    assert gain_db > 10          # bruit de fond atténué d'au moins 10 dB


def test_vad_robuste_au_bruit():
    y = A.preprocess(_voix_dans_le_bruit(), denoise_strength=1.5)
    segs = A.vad_segments(y, min_snr_db=10)
    assert len(segs) == 2


def test_bruit_seul_ignore():
    rng = np.random.default_rng(2)
    sr = A.SR
    clic = np.zeros(3 * sr)
    clic[sr : sr + 200] = 0.02 * rng.standard_normal(200)   # petit choc isolé
    x = (0.05 * rng.standard_normal(3 * sr) + clic).astype(np.float32)
    segs = A.vad_segments(A.preprocess(x, denoise_strength=1.5), min_snr_db=10)
    assert all(s.duration < 0.6 for s in segs)  # pas de longue « parole » inventée
