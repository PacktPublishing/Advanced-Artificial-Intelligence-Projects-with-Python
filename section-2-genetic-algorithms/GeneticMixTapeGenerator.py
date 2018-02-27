
# Load prepared track metadata and features

REQUIRED_GENRE = None #'Electronic'

import pickle
with open('tracks_metadata.pkl', 'rb') as f:
    tracks_metadata = pickle.load(f)
with open('track_features.pkl', 'rb') as f:
    track_features = pickle.load(f)
all_tracks = list(tracks_metadata.keys())

def computeTrackDuration(track):
    duration = tracks_metadata[track]['track_duration']
    if len(duration.split(':')) == 2:
        (mins, secs) = duration.split(':')
        hours = 0
    elif len(duration.split(':')) == 3:
        (hours, mins, secs) = duration.split(':')
    else:
        secs = duration
        mins = 0
        hours = 0
    return (float(hours) * 60.0 + float(mins) + float(secs)/60.0)

# remove tracks that have missing keys
removed_tracks = []
for track in all_tracks:
    if 'genres' not in track_features[track] or       'bpm' not in track_features[track] or       'beats_loudness' not in track_features[track] or       'loudness' not in track_features[track] or       'dissonance' not in track_features[track] or       'genres' not in track_features[track] or       'tonal_key' not in track_features[track] or       'interest' not in track_features[track] or       'listens' not in track_features[track] or       'favorites' not in track_features[track]:
        del tracks_metadata[track]
        del track_features[track]
        removed_tracks.append(track)
        continue
    duration = computeTrackDuration(track)
    if duration < 3 or duration > 10:
        del tracks_metadata[track]
        del track_features[track]
        removed_tracks.append(track)
        continue
    if REQUIRED_GENRE is not None:
        if REQUIRED_GENRE not in track_features[track]['genres']:
            missing_required_genre = True
            del tracks_metadata[track]
            del track_features[track]
            removed_tracks.append(track)
all_tracks = list(set(all_tracks)-set(removed_tracks))

import numpy
import random
import sys
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# To evaluate an individual (a playlist), we'll compute a variety of scores:
# 
# - Total playlist time - desired playlist time (e.g., 1hr), absolute value difference
#   - Minimize
# - Entropy of genres: sum across every possible genre g: - percent-of-songs-with-genre * log(percent-of-songs-with-genre)
#   - Minimize, i.e., have a playlist with mostly the same genre
# - Entropy of tonal keys
#   - Minimize
# - Difference in beats-per-minute in successive songs (absolute sum)
#   - Minimize
# - Absolute difference of largest beats loudness - smallest beats loudness
#   - Minimize
# - Absolute difference of largest loudness - smallest loudness
#   - Minimize
# - Absolute difference of largest dissonance - smallest dissonance
#   - Minimize
# - Average interest
#   - Maximize
# - Average listens
#   - Maximize
# - Average favorites
#   - Maximize

import math
from collections import Counter
def calcEntropy(individual, field, multivalue):
    valcounts = Counter()
    for track in individual:        
        if multivalue:
            valcounts.update(track_features[track][field])
        else:
            valcounts.update([track_features[track][field]])
    sum = 0.0
    for val in valcounts.elements():
        p = float(valcounts[val])/float(len(individual))
        if p > 0:
            sum -= p * math.log(p)
    return sum

def evalPlaylist(individual, desired_play_time):
    # difference in actual play time and desired play time (in minutes)
    play_time = 0.0
    for track in individual:
        play_time += computeTrackDuration(track)
    diff_play_time = abs(play_time - desired_play_time)
    
    genre_entropy = calcEntropy(individual, 'genres', True)
    tonal_keys_entropy = calcEntropy(individual, 'tonal_key', False)
    
    sum_diff_bpm = 0.0
    for i in iter(range(1, len(individual))):
        sum_diff_bpm += abs(track_features[individual[i-1]]['bpm'] - 
                            track_features[individual[i]]['bpm'])

    min_beats_loudness = sys.float_info.max
    max_beats_loudness = 0.0
    min_loudness = sys.float_info.max
    max_loudness = 0.0
    min_dissonance = sys.float_info.max
    max_dissonance = 0.0
    for track in individual:
        if min_beats_loudness > track_features[track]['beats_loudness']:
            min_beats_loudness = track_features[track]['beats_loudness']
        if max_beats_loudness < track_features[track]['beats_loudness']:
            max_beats_loudness = track_features[track]['beats_loudness']
        if min_loudness > track_features[track]['loudness']:
            min_loudness = track_features[track]['loudness']
        if max_loudness < track_features[track]['loudness']:
            max_loudness = track_features[track]['loudness']
        if min_dissonance > track_features[track]['dissonance']:
            min_dissonance = track_features[track]['dissonance']
        if max_dissonance < track_features[track]['dissonance']:
            max_dissonance = track_features[track]['dissonance']
    diff_beats_loudness = max_beats_loudness - min_beats_loudness
    diff_loudness = max_loudness - min_loudness
    diff_dissonance = max_dissonance - min_dissonance
    
    sum_interest = 0
    sum_listens = 0
    sum_favorites = 0
    for track in individual:
        sum_interest += track_features[track]['interest']
        sum_listens += track_features[track]['listens']
        sum_favorites += track_features[track]['favorites']
    avg_interest = sum_interest / float(len(individual))
    avg_listens = sum_listens / float(len(individual))
    avg_favorites = sum_favorites / float(len(individual))
        
    return (diff_play_time, genre_entropy, tonal_keys_entropy,
            sum_diff_bpm, diff_beats_loudness, diff_loudness, diff_dissonance,
            avg_interest, avg_listens, avg_favorites)

# an invalid playlist has <3 songs or repeated songs
def validPlaylist(individual):
    return len(individual) >= 3 and len(set(individual)) == len(individual)

creator.create("FitnessMulti", base.Fitness,
               weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0))

creator.create("Individual", list, fitness=creator.FitnessMulti)

# adds or removes a track not already in the playlist, at a random location
def mutatePlaylist(individual):
    if random.random() > 0.5:
        # add a track
        track = random.choice(all_tracks)
        if track not in individual:
            idx = random.choice(range(0, len(individual)))
            individual = individual[:idx] + [track] + individual[idx:]
    elif len(individual) > 5:
        # delete a track
        del individual[random.choice(range(0, len(individual)))]
    return creator.Individual(individual),

NUM_SONGS = 20

toolbox = base.Toolbox()
toolbox.register("tracks", random.sample, all_tracks, NUM_SONGS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.tracks)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalPlaylist, desired_play_time=120)
invalidPlaylistScore = (sys.float_info.max, sys.float_info.max, sys.float_info.max,
                        sys.float_info.max, sys.float_info.max, sys.float_info.max,
                        sys.float_info.max, 0.0, 0.0, 0.0)
toolbox.decorate("evaluate", tools.DeltaPenalty(validPlaylist, invalidPlaylistScore))
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutatePlaylist)
toolbox.register("select", tools.selNSGA2)

# Simulation parameters:
# Number of generations
NGEN = 5000
# The number of individuals to select for the next generation (eliminate bad ones).
MU = 500
# The number of children to produce at each generation.
LAMBDA = 50
# The probability that an offspring is produced by crossover.
CXPB = 0.5
# The probability that an offspring is produced by mutation.
MUTPB = 0.5

# Initial population
pop = toolbox.population(n=MU)

# The top playlist is the one that is best on all scores in the fitness
hof = tools.ParetoFront()

# fitness is composed of:
# 0: diff_play_time, 1: genre_entropy, 2: tonal_keys_entropy,
# 3: sum_diff_bpm, 4: diff_beats_loudness, 5: diff_loudness, 6: diff_dissonance,
# 7: avg_interest, 8: avg_listens, 9: avg_favorites

# compute some statistics as the simulation proceeds
diff_play_time_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
genre_entropy_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
tonal_keys_entropy_stats = tools.Statistics(key=lambda ind: ind.fitness.values[2])
sum_diff_bpm_stats = tools.Statistics(key=lambda ind: ind.fitness.values[3])
diff_beats_loudness_stats = tools.Statistics(key=lambda ind: ind.fitness.values[4])
diff_loudness_stats = tools.Statistics(key=lambda ind: ind.fitness.values[5])
diff_dissonance_stats = tools.Statistics(key=lambda ind: ind.fitness.values[6])
avg_interest_stats = tools.Statistics(key=lambda ind: ind.fitness.values[7])
avg_listens_stats = tools.Statistics(key=lambda ind: ind.fitness.values[8])
avg_favorites_stats = tools.Statistics(key=lambda ind: ind.fitness.values[9])

stats = tools.MultiStatistics(time=diff_play_time_stats,
                              genre=genre_entropy_stats,
                              tonal=tonal_keys_entropy_stats,
                              bpm=sum_diff_bpm_stats,
                              bloud=diff_beats_loudness_stats,
                              loud=diff_loudness_stats,
                              diss=diff_dissonance_stats,
                              interest=avg_interest_stats,
                              listens=avg_listens_stats,
                              favs=avg_favorites_stats)
stats.register("avg", numpy.mean, axis=0)

# run the simulation
algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                          stats, halloffame=hof, verbose=False)


best = hof[0]
print("Best playlist:")
print("---")
for track in best:
    print("%s - %s / %s (%s, %.2f BPM) %s" % (track, tracks_metadata[track]['artist_name'],
                                              tracks_metadata[track]['track_title'],
                                              tracks_metadata[track]['track_duration'],
                                              track_features[track]['bpm'],
                                              tracks_metadata[track]['track_genres']))
print('---')
print("Diff play time:", best.fitness.values[0])
print("Genre entropy:", best.fitness.values[1])
print("Tonal keys entropy:", best.fitness.values[2])
print("Sum diff BPM:", best.fitness.values[3])
print("Diff beats loudness:", best.fitness.values[4])
print("Diff loudness:", best.fitness.values[5])
print("Diff dissonance:", best.fitness.values[6])
print("Avg interest:", best.fitness.values[7])
print("Avg listens:", best.fitness.values[8])
print("Avg favorites:", best.fitness.values[9])

# write playlist (copy songs and create m3u file, one song per line)
import os
from shutil import copyfile

os.mkdir('output')
with open('output/playlist.m3u', 'wt') as m3u:
    for track in best:
        trackmp3 = track.split('/')[3]
        copyfile(track, 'output/%s' % trackmp3)
        m3u.write('%s\n' % trackmp3)

