import tensorflow as tf
import numpy as np
import pandas as pd

from config import index

from util import to_mel_spectrogram, to_onset_times, to_onset_samples

class DrumTranscriber:
    def __init__(self, model):
        self.model = tf.keras.models.load_model(model)

    def predict(self, samples: np.array, sr: int) -> pd.DataFrame:
        """
        :return predictions (np.array): Hits probability predicted by the model
        """
        # get onset
        onset_samples = to_onset_samples(samples, sr=sr)

        # convert to mel spectrogram
        mel_specs = np.array([to_mel_spectrogram(s, sr=sr)
                              for s in onset_samples])
        mel_specs = np.expand_dims(mel_specs, axis=-1).repeat(3, axis=-1)

        # onset times for each hit
        hit_times = to_onset_times(samples, sr)

        # get the predicted label
        predictions = self.model.predict(mel_specs)

        df = pd.DataFrame(predictions, columns=list(index.values()))
        df['time'] = hit_times

        return df
