import pandas as pd
import numpy as np
from itertools import compress
from utils.retroid_dict import get_retroid

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('data/lahman/mlb_data/Pitching.csv')
df['playerID'] = df['playerID'].apply(get_retroid)
df.rename(columns={'playerID': 'retroID'}, inplace=True)

columns_to_drop = ['stint', 'teamID', 'lgID']
df.drop(columns=columns_to_drop, inplace=True)

df['IBB'].fillna(0, inplace=True)
df['SH'].fillna(0, inplace=True)
df['SF'].fillna(0, inplace=True)
df['GIDP'].fillna(0, inplace=True)

df_baopp_missing = df[df['BAOpp'].isnull()].sort_values('retroID')
baopp_checks = df[df['retroID'].isin(
    df_baopp_missing['retroID'])].sort_values('retroID')
val_counts = baopp_checks['retroID'].value_counts()
one_time_players = list(compress(val_counts.index, val_counts.eq(1)))
filled_baopp = df['BAOpp'].mean() + df['BAOpp'].std()
df.loc[df['retroID'].isin(one_time_players), ['BAOpp']] = filled_baopp
df_baopp_missing = df[df['BAOpp'].isnull()].sort_values('retroID')
baopp_checks = df[df['retroID'].isin(
    df_baopp_missing['retroID'])].sort_values('retroID')
players = list(val_counts.index)
df['BAOpp'] = df.groupby("retroID")['BAOpp'].transform(
    lambda baopp: baopp.fillna(baopp.mean()))

df_era_missing = df[df['ERA'].isnull()].sort_values('retroID')
era_checks = df[df['retroID'].isin(
    df_era_missing['retroID'])].sort_values('retroID')
val_counts = era_checks['retroID'].value_counts()
one_time_players = list(compress(val_counts.index, val_counts.eq(1)))
filled_era = df['ERA'].mean() + (df['ERA'].std())/2
df.loc[df['retroID'].isin(one_time_players), ['ERA']] = filled_era
df_era_missing = df[df['ERA'].isnull()].sort_values('retroID')
era_checks = df[df['retroID'].isin(
    df_era_missing['retroID'])].sort_values('retroID')
df['ERA'] = df.groupby("retroID")['ERA'].transform(
    lambda era: era.fillna(era.mean()))

df_bfp_missing = df[df['BFP'].isnull()].sort_values('retroID')
df['BFP'].fillna(df['IPouts'] - df['G'], inplace=True)

df.drop(columns=['yearID'], inplace=True)
average_stats = ['BAOpp', 'ERA']
df_avgs = df[['retroID', 'BAOpp', 'ERA']]
df_sums = df.drop(columns=average_stats)
df_avgs = df_avgs.groupby('retroID').mean().round(4).reset_index()
df_sums = df_sums.groupby('retroID').sum().reset_index()
df = pd.merge(df_avgs, df_sums, on='retroID')

df.to_csv('output/pitching.csv')
