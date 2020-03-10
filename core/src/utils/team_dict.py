import pandas as pd
df = pd.read_csv('data/lahman/mlb_data/Teams.csv')
df = df[['teamID', 'franchID']]
team_dict = df.set_index('teamID').to_dict()['franchID']


def get_team(team):
    return team_dict[team] if team_dict[team] is not None else team
