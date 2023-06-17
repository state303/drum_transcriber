import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import settings


def normalize_duration(samples: np.array, sr: int, length: int) -> np.array:
    """
    :param samples: samples in array
    :param sr: sample rate
    :param length: target length in seconds
    :return samples (np.array): padded or trimmed audio
    """
    desired_length = int(sr * length)

    if len(samples) > desired_length:
        trim_amount = (len(samples) - desired_length) // 2
        return samples[trim_amount:len(samples) - trim_amount]
    else:
        add_amount = (desired_length - len(samples)) // 2
        return np.concatenate((np.zeros(add_amount), samples, np.zeros(add_amount)))


def to_onset_frames(samples: np.array, sr: int = 44100) -> list:
    """
    :param samples: samples array of the audio
    :param sr: sample rate
    :return onset_frames (list): formatted wby [(begin, end), ...]
    """
    onset_backtracks = librosa.onset.onset_detect(y=samples, sr=sr, units='samples', backtrack=True)
    onset_backtracks = np.append(onset_backtracks, min(onset_backtracks[-1] + sr, len(samples)))
    onset_frames = list(zip(onset_backtracks[:-1], onset_backtracks[1:]))
    return onset_frames


def to_onset_samples(samples: np.array, sr: int = 44100, onset_frames: list = None) -> list:
    """
    :param samples: samples array of the audio
    :param sr: sample rate
    :param onset_frames: uses provided frames, or compute new frames.
    :return on_set_samples (list): on set list
    """
    if onset_frames is None:
        onset_frames = to_onset_frames(samples, sr)

    onset_samples = [normalize_duration(samples[s:e], sr, 1)
                     for s, e in onset_frames]

    return onset_samples


def to_onset_times(samples: np.array, sr: int = 44100) -> np.array:
    """
    :param samples: samples array of the audio
    :param sr: sample rate
    :return onset_times (np.array): onset times in seconds
    """
    onset_times = librosa.onset.onset_detect(y=samples, sr=sr, units='time', backtrack=False)

    return onset_times


def to_mel_spectrogram(samples: np.array, sr: int = 44100, target_shape=settings['TARGET_SHAPE']) -> np.array:
    """
    :param target_shape: shape for samples
    :param samples: array of the audio
    :param sr: sample rate
    :return mel_spectrogram (np.array): array of mel spectrum in dB
    """
    hop_length = len(samples) // target_shape[0]

    mel_features = librosa.feature.melspectrogram(y=samples, sr=sr, hop_length=hop_length, n_mels=target_shape[0])
    mel_features = mel_features[:, :target_shape[1]]

    mel_in_db = librosa.power_to_db(mel_features, ref=np.max)
    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler.fit_transform(mel_in_db)
