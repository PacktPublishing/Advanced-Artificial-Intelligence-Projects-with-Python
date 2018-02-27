
# Loading track metadata

import csv
import re
import json
tracks_metadata = {}
with open('fma/fma_metadata/raw_tracks.csv', encoding='utf-8') as fin:
    reader = csv.reader(fin, skipinitialspace=True, quotechar="\"")
    headers = []
    rownum = 0
    for row in reader:
        if rownum == 0:
            headers = row
            rownum += 1
        else:
            trackid = '%06d' % int(row[0])
            fname = 'fma/fma_full/%s/%s.mp3' % (trackid[:3], trackid)
            tracks_metadata[fname] = dict(zip(headers, row))
            if len(tracks_metadata[fname]['track_genres']) > 0:
                genres = json.loads(re.sub(r'\'', '"', tracks_metadata[fname]['track_genres'])) # fix json quotes
                tracks_metadata[fname]['track_genres'] = list(map(lambda g: g['genre_title'], genres))

# Extract features for tracks
from essentia.standard import *
def extract_features(track):
    try:
        features, _ = MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                     rhythmStats=['mean', 'stdev'],
                                     tonalStats=['mean', 'stdev'])(track)
        genres = tracks_metadata[track]['track_genres']
        listens = int(tracks_metadata[track]['track_listens'])
        favorites = int(tracks_metadata[track]['track_favorites'])
        interest = int(tracks_metadata[track]['track_interest'])
        loudness = features['lowlevel.loudness_ebu128.integrated']
        loudness_range = features['lowlevel.loudness_ebu128.loudness_range']
        bpm = features['rhythm.bpm']
        beats_loudness = features['rhythm.beats_loudness.mean']
        tonal_key = features['tonal.key_edma.key']
        tonal_scale = features['tonal.key_edma.scale'],
        dissonance = features['lowlevel.dissonance.mean']
        f = {'genres': genres,
             'listens': listens,
             'favorites': favorites,
             'interest': interest,
             'loudness': loudness,
             'loudness_range': loudness_range,
             'bpm': bpm,
             'beats_loudness': beats_loudness,
             'tonal_key': tonal_key,
             'tonal_scale': tonal_scale,
             'dissonance': dissonance
            }
    except:
        f = {}
    return f

import random
import time
ts = list(tracks_metadata.keys())
random.shuffle(ts)

# Time to extract features on 10 tracks: 300s
# 
# Number of tracks in total: 100k
# 
# Estimated time to extract features on all tracks: (100000/10) x 300s = about 35 days
# 
# Estimated time if done in parallel (16 threads) = about 52 hours

# required 285378 seconds actually (79 hours)

import multiprocessing
pool = multiprocessing.Pool(16)
start = time.time()
track_features_list = pool.map(extract_features, ts)
track_features = dict(zip(ts, track_features_list))
pool.close()
pool.join() # see: https://stackoverflow.com/a/47683305
end = time.time()
print(end - start)

# save results of all that processing
import pickle
with open('track_features.pkl', 'wb') as track_features_out:
    pickle.dump(track_features, track_features_out)
with open('tracks_metadata.pkl', 'wb') as track_metadata_out:
    pickle.dump(tracks_metadata, track_metadata_out)

