import numpy as np
import pandas as pd

def format_empirical_data():
    data = pd.read_excel("data/empirical.xlsx", header=0)
    # remove test data, then remove redundant "tag" field
    for i, tag in enumerate(data['tag'].unique()):
        if type(tag)=='float':
            continue  # data wasn't label with tester tag
#         if type(tag)=='float' and np.isnan(tag):
#             data = data.drop(data[data.tag==tag].index)  # suspicious lack of tag
        if tag in ["Testversion", "TEST JOHANNES / LOL", "999", "Johannes Test 2"]:
            data = data.drop(data[data.tag==tag].index)
    data = data.drop(columns=['tag'])
    # rename sample_size to maxSamples
    data = data.rename(columns={'sample_size': 'maxSamples'})
    # rename ticks to cues, left to A, and right to B, p1 to pA, p2 to pB
    data = data.rename(columns={'ticks': 'cues', 'left': 'A', 'right': 'B', 'p1': 'pA', 'p2': 'pB'})
    # add a "correct" column based on "selected right" and comparing "p1" to "p2"
    chosen_answer = data['selected_right'].to_numpy()
    correct_answer = data['pB'].to_numpy() > data['pA'].to_numpy()
    correct = chosen_answer == correct_answer
    data['correct'] = 1.0*correct
    data = data.drop(columns=[
        'timestamp', 'subrange_key', 'duration_ms', 'empirical_delta', 'empirical_p1', 'empirical_p2',
        'selected_right', 'started_right', 'last_left', 'last_right',
    ])
    # first participant is fake
    data = data.drop(data[data.participant_id=="11e92cd2764348faa18918c94947d4fa"].index)
    data.to_pickle("empirical_data.pkl")

def collapse_empirical_data():
    data = pd.read_pickle("data/empirical_data.pkl").query("maxSamples==12")
    dfs = []
    columns = ('ID', 'dP', 'mean correct', 'mean cues', 'mean cues for chosen option', 'trials')
    for s in data['participant_id'].unique():
        for dP in data['delta'].unique():
            d = data.query("participant_id==@s and delta==@dP")
            dfs.append(pd.DataFrame([[
                s, dP, d['correct'].mean(), d['cues'].mean(), d['cues for chosen option'].mean(), d['correct'].size,
                ]], columns=columns))
    collapsed_data = pd.concat(dfs, ignore_index=True)
    collapsed_data.to_pickle("collapsed_empirical_data.pkl")
    collapsed_data

format_empirical_data()