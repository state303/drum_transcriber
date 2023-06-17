import mido


def create_midi(times, filename, tempo):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    ticks_per_beat = midi.ticks_per_beat
    ticks_per_second = ticks_per_beat * 1000000.0 / tempo
    note = 42

    last_tick = 0
    for time in times:
        current_tick = int(round(time * ticks_per_second))
        delta = current_tick - last_tick
        track.append(mido.Message('note_on', note=note, velocity=64, time=delta))
        track.append(mido.Message('note_off', note=note, velocity=64, time=int(round(0.1*ticks_per_second))))
        last_tick = current_tick

    midi.save(filename)