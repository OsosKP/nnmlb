import pandas as pd
df = pd.read_csv('../core/data/FanGraphsConstants.csv')


def woba_constants(year):
    data = df[df['Season'] == year]
    return data['wOBA'], data['wOBAScale'], data['wBB'], data['wHBP'], \
        data['w1B'], data['w2B'], data['w3B'], data['wHR']
