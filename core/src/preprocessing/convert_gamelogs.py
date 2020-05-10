import os
import pandas as pd
from utils.team_dict import get_team

game_log_columns_all = pd.read_csv(
    'core/data/retrosheet/rs_gl_cols_all.csv', header=None)
game_log_columns = pd.read_csv(
    'core/data/retrosheet/rs_gl_cols.csv', header=None)

columns = game_log_columns_all[1].tolist()
columns_to_keep = game_log_columns[1].tolist()

for year in range(1919, 2020):
    file_path = 'core/data/retrosheet/gamelogs/GL{}'.format(year)
    df = pd.read_csv(file_path + '.TXT', delimiter=',',
                     header=0, names=columns)
    df = df[columns_to_keep]
    df['date'] = df['date'].astype(str)
    df['year'] = df['date'].str[0:4].astype(int)
    df['month'] = df['date'].str[4:6].astype(int)
    df['day'] = df['date'].str[6:8].astype(int)
    night_game = pd.get_dummies(
        df['day_night'], drop_first=(df['day_night'].nunique() > 1))
    df.insert(loc=6, column='night_game', value=night_game)
    df = df.drop(columns=['date', 'day_night'])
    df['visit_team'] = df['visit_team'].apply(get_team)
    df['home_team'] = df['home_team'].apply(get_team)
    df['home_win'] = (df['home_score'] > df['visit_score']).astype(int)
    if os.path.exists(file_path + '.TXT'):
        os.remove(file_path + '.TXT')
    df.to_csv(file_path + '.csv', index=False)
