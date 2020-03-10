import pandas as pd
from retroid_dict import get_retroid
from teamid_dict import get_team

df = pd.read_csv('../core/data/lahman/mlb_data/Batting.csv')
metadata_columns = ['playerID', 'yearID', 'teamID']
df = df[metadata_columns]

df['playerID'] = df['playerID'].apply(get_retroid)
df.rename(columns={'playerID': 'retroID'}, inplace=True)
df['teamID'] = df['teamID'].apply(get_team)

team_dict = df.set_index(['retroID', 'yearID']).to_dict()['teamID']


def get_team_for_player_and_year(retroId, yearId):
    return team_dict[(retroId, yearId)]
