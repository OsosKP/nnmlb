import pandas as pd
df = pd.read_csv('./data/lahman/mlb_data/Teams.csv')
df = df[['teamID', 'franchID']]
teams = df.set_index('teamID').to_dict()['franchID']

def team_dict():
    return teams