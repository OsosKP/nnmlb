from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('core/output/batters.csv')
df_adv = pd.read_csv('core/output/advanced_batting.csv')

df = df.merge(df_adv, how='left')
df['wOBA'].fillna(0, inplace=True)
df['wRC+'].fillna(0, inplace=True)
df['WAR'].fillna(0, inplace=True)

df.drop(columns=['Rating'], inplace=True)

df['Batting'] = df[['wOBA', 'wRC+', 'WAR']].mean(axis=1).round(3)

scaler = MinMaxScaler()
df['Batting'] = scaler.fit_transform(df[['Batting']])

df.to_csv('core/output/batters.csv', index=False, float_format='%g')
