"""Audio : décodage (ffmpeg), normalisation, VAD énergétique et découpage.

Les modèles CTC (MMS) saturent au-delà d'environ une minute d'audio en une
passe ; on découpe donc le signal sur les silences, avec une longueur maximale
de segment. Chaque segment est ensuite transcrit puis traduit séparément, ce
qui donne aussi à NLLB des unités proches de la phrase (sa zone de confort).
"""
from __future__ import annotations

import io
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

SR = 16_000


class AudioError(ValueError):
    pass


def decode_audio(data: bytes, sr: int = SR) -> np.ndarray:
    """Décode n'importe quel format lisible par ffmpeg (webm/opus du navigateur,
    wav, mp3, m4a, ogg…) en float32 mono à `sr` Hz."""
    if not data:
        raise AudioError("Fichier audio vide.")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr), "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, check=True, timeout=60)
    except FileNotFoundError as e:
        raise AudioError("ffmpeg est introuvable sur le serveur.") from e
    except subprocess.CalledProcessError as e:
        raise AudioError(f"Audio illisible : {e.stderr.decode(errors='ignore')[:200]}") from e
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioError("Aucun échantillon audio décodé.")
    return audio


# ==========================================================================
# Réduction de bruit (numpy seul, aucune dépendance)
# ==========================================================================
SPEECH_BAND = (250.0, 3800.0)   # bande utile de la voix pour la VAD
NOISE_FLATNESS = 0.35           # platitude spectrale au-delà de laquelle c'est du bruit


def highpass(audio: np.ndarray, sr: int = SR, cutoff: float = 80.0) -> np.ndarray:
    """Passe-haut dans le domaine fréquentiel (rampe douce de cutoff/2 à cutoff) :
    retire le ronflement secteur, le vent et les chocs sur le micro, sous la voix."""
    if audio.size < 2:
        return audio
    spec = np.fft.rfft(audio)
    f = np.fft.rfftfreq(len(audio), 1.0 / sr)
    spec *= np.clip((f - cutoff / 2) / (cutoff / 2), 0.0, 1.0)
    return np.fft.irfft(spec, n=len(audio)).astype(np.float32)


def _stft(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect" if len(x) > pad else "constant")
    n = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1), win


def _istft(spec: np.ndarray, win: np.ndarray, hop: int, length: int) -> np.ndarray:
    n_fft = len(win)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n_fft, dtype=np.float64)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop : i * hop + n_fft] += f
        norm[i * hop : i * hop + n_fft] += win**2
    out /= np.maximum(norm, 1e-8)
    pad = n_fft // 2
    return out[pad : pad + length].astype(np.float32)


def _smooth(m: np.ndarray, kt: int = 3, kf: int = 3) -> np.ndarray:
    """Moyenne glissante temps × fréquence du masque (évite le « bruit musical »)."""
    if kt > 1:
        k = np.ones(kt) / kt
        m = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, m)
    if kf > 1:
        k = np.ones(kf) / kf
        m = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, m)
    return m


def denoise(audio: np.ndarray, sr: int = SR, strength: float = 1.5, floor: float = 0.12) -> np.ndarray:
    """Soustraction spectrale avec porte douce (principe de « noisereduce »).

    1. Le profil de bruit est estimé sur les 20 % de trames les plus faibles
       (pauses entre les mots, début et fin d'enregistrement) ;
    2. chaque case temps-fréquence est atténuée selon son rapport signal/bruit ;
       `strength` > 1 sur-soustrait (bruit plus agressif), `floor` garde un
       minimum de signal pour ne pas abîmer la voix.
    Efficace sur le bruit stationnaire : ventilateur, climatisation, moteur,
    souffle, foule lointaine. Moins sur une voix concurrente proche.
    """
    if audio.size < sr // 4 or strength <= 0:
        return audio
    n_fft, hop = 512, 128
    spec, win = _stft(audio, n_fft, hop)
    mag = np.abs(spec)
    frame_e = np.mean(mag**2, axis=1)
    quiet = frame_e <= np.percentile(frame_e, 20)
    noise = np.mean(mag[quiet], axis=0) if quiet.any() else np.percentile(mag, 10, axis=0)
    gain = 1.0 - strength * noise[None, :] / np.maximum(mag, 1e-10)
    gain = _smooth(np.clip(gain, floor, 1.0))
    return _istft(spec * gain, win, hop, len(audio))


def preprocess(audio: np.ndarray, denoise_strength: float = 0.0, denoise_floor: float = 0.12) -> np.ndarray:
    """Retire la composante continue et les basses (< 80 Hz), réduit le bruit
    de fond si `denoise_strength` > 0, puis normalise le niveau (crête à -1 dBFS)."""
    audio = audio - float(np.mean(audio))
    if denoise_strength > 0:
        audio = denoise(highpass(audio), strength=denoise_strength, floor=denoise_floor)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-4:
        audio = audio * (0.89 / peak)
    return audio.astype(np.float32)


@dataclass
class Segment:
    start: float  # secondes
    end: float
    audio: np.ndarray

    @property
    def duration(self) -> float:
        return self.end - self.start


def vad_segments(
    audio: np.ndarray,
    sr: int = SR,
    frame_ms: int = 30,
    min_silence_s: float = 0.35,
    min_speech_s: float = 0.25,
    max_chunk_s: float = 20.0,
    pad_s: float = 0.15,
    min_snr_db: float = 0.0,
) -> list[Segment]:
    """VAD énergétique à seuil adaptatif (percentiles de l'énergie des trames).

    Pas de dépendance réseau (contrairement à silero-vad via torch.hub) ; la
    robustesse au bruit est moindre : c'est une limite documentée, et le module
    est interchangeable.
    """
    frame = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame
    if n_frames == 0:
        return [Segment(0.0, len(audio) / sr, audio)] if len(audio) else []

    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    # Énergie dans la bande de la voix seulement : les bruits graves (moteur,
    # vent) et aigus (sifflements) ne déclenchent plus la détection.
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    band = (freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])
    energy = 10 * np.log10(np.mean(spec[:, band], axis=1) + 1e-12)
    noise = np.percentile(energy, 15)
    speech_lvl = np.percentile(energy, 90)
    # Seuil à 35 % entre le plancher de bruit et le niveau de parole, borné.
    thr = noise + max(6.0, 0.35 * (speech_lvl - noise))
    voiced = energy > thr

    # Lissage : on comble les silences courts.
    min_sil = max(1, int(min_silence_s * 1000 / frame_ms))
    i = 0
    while i < n_frames:
        if not voiced[i]:
            j = i
            while j < n_frames and not voiced[j]:
                j += 1
            if 0 < i and j < n_frames and (j - i) < min_sil:
                voiced[i:j] = True
            i = j
        else:
            i += 1

    # Extraction des zones voisées.
    raw: list[tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if voiced[i]:
            j = i
            while j < n_frames and voiced[j]:
                j += 1
            if (j - i) * frame_ms / 1000 >= min_speech_s:
                raw.append((i, j))
            i = j
        else:
            i += 1

    if min_snr_db > 0 and raw:
        # Rejette les zones dont le rapport signal/bruit est trop faible :
        # claquement, porte, bruit de fond qui monte… plutôt que de la parole.
        raw = [(a, b) for a, b in raw if np.percentile(energy[a:b], 90) - noise >= min_snr_db]
        if not raw:
            return []

    if not raw and min_snr_db > 0:
        # Aucune variation d'énergie : soit une parole continue (pas de pause),
        # soit un bruit stationnaire. Le bruit a un spectre « plat », la voix un
        # spectre à harmoniques : on mesure la platitude spectrale dans la bande.
        sb = spec[:, band] + 1e-12
        flat = np.exp(np.mean(np.log(sb), axis=1)) / np.mean(sb, axis=1)
        if float(np.median(flat)) > NOISE_FLATNESS:
            return []

    if not raw:
        # Rien de détecté : on renvoie tout (le modèle tranchera, et le score
        # de confiance le signalera).
        return _split_long(Segment(0.0, len(audio) / sr, audio), sr, max_chunk_s)

    pad = int(pad_s * sr)
    segs: list[Segment] = []
    for a, b in raw:
        s = max(0, a * frame - pad)
        e = min(len(audio), b * frame + pad)
        segs.extend(_split_long(Segment(s / sr, e / sr, audio[s:e]), sr, max_chunk_s))
    return segs


def _split_long(seg: Segment, sr: int, max_chunk_s: float) -> list[Segment]:
    """Coupe un segment trop long au point d'énergie minimale le plus proche de
    la limite, pour éviter de trancher au milieu d'un mot."""
    if seg.duration <= max_chunk_s:
        return [seg]
    out: list[Segment] = []
    audio, offset = seg.audio, seg.start
    max_len = int(max_chunk_s * sr)
    win = int(0.02 * sr)
    while len(audio) > max_len:
        search_from = int(max_len * 0.7)
        region = audio[search_from:max_len]
        n = max(1, len(region) // win)
        e = np.mean(region[: n * win].reshape(n, win) ** 2, axis=1)
        cut = search_from + int(np.argmin(e)) * win + win // 2
        out.append(Segment(offset, offset + cut / sr, audio[:cut]))
        offset += cut / sr
        audio = audio[cut:]
    if len(audio):
        out.append(Segment(offset, offset + len(audio) / sr, audio))
    return out


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode un signal float32 en WAV PCM 16 bits (bibliothèque standard)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
