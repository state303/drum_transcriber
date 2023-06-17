import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from pytube import YouTube
from streamlit_player import st_player

from config import index
from drum_transcriber import DrumTranscriber
from midigen import create_midi

###############
# Preparation #
###############


def is_file_exists(path: str):
    return os.path.exists(path) and os.path.isfile(path)


def prepare_path(target: str):
    # check if target is file, not directory
    if is_file_exists(target):
        os.remove(target)
    if not os.path.exists(target):
        os.makedirs(target)

@st.cache_data
def get_title(input):
    return YouTube(input).title


@st.cache_data
def get_predictions(input, offset=0):
    yt = YouTube(input)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(output_path=".")

    base, ext = os.path.splitext(out_file)
    new_file = base + '.wav'
    os.rename(out_file, new_file)

    samples, sr = librosa.load(new_file, sr=44100, offset=offset)
    st.audio(samples, sample_rate=sr)

    os.remove(new_file)

    predictions = transcriber.predict(samples, sr)
    return predictions, samples, sr


@st.cache_resource
def init_transcriber():
    path = os.path.join("model", "drum_model.h5")
    return DrumTranscriber(path)


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def export_midi(title, part):
    return os.path.join("output", title + "-" + part + ".mid")

transcriber = init_transcriber()
st.title('Transcriber')

input = st.text_input('YouTube Link', value='https://www.youtube.com/watch?v=4SDBJp_B5qQ')
start_from = st.number_input(label='Start offset (seconds)', min_value=0)

if input:
    st_player(input)

predictions = None

if input and start_from is not None:
    st.title('Predictions')
    predictions, samples, sr = get_predictions(input, start_from)

    labelled_preds = [index[i] for i in
                      np.argmax(
                          predictions[index.values()].to_numpy(), axis=1)
                      ]

    predictions['prediction'] = labelled_preds
    predictions['confidence'] = predictions.apply(
        lambda x: f"{x[x['prediction']] * 100:.1f}%", axis=1)
    predictions['time'] = predictions['time'].round(2)

    st.write(predictions[['time', 'prediction', 'confidence']].T)

    fig, ax = plt.subplots(sharex='all', nrows=7, figsize=(20, 20))
    librosa.display.waveshow(samples, sr=sr, offset=start_from, ax=ax[0])

    ax[0].set_yticklabels([])
    ax[0].set_xlabel(None)

    tempo = int(round(60000000 / librosa.beat.tempo(y=samples, sr=sr)[0]))
    title = get_title(input)

    prepare_path(title)

    # Traverse through each labels then export the result
    for i in range(1, 7):
        label_name = index[i - 1]
        hit_times = np.array(predictions[predictions['prediction'] == label_name]['time'].to_list())

        ax[i].vlines(hit_times + start_from, -1, 1)
        ax[i].set_ylabel(label_name, rotation=0, fontsize=20)
        ax[i].set_yticklabels([])

        filename = label_name + ".mid"
        filepath = os.path.join(title, filename)
        create_midi(hit_times, filepath, tempo)

        if i == 6:
            ax[i].set_xlabel('time')
            ax[i].set_xlabel('time')

    st.pyplot(fig)

    st.download_button(
        "Download predictions.csv",
        convert_df(predictions),
        "predictions.csv",
        "text/csv",
        key='download-csv'
    )

    st.write(predictions)
