import pandas as pd
df = pd.read_csv('data/lahman/mlb_data/People.csv')
df = df[['playerID', 'retroID']]
id_dict = df.set_index('playerID').to_dict()['retroID']


def get_retroid(player_id):
    return id_dict[player_id] if id_dict[player_id] is not None else player_id
