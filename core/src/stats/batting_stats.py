import pandas as pd
from utils.position_dict import get_pos

df_by_player = pd.read_csv('../core/output/batters.csv')
df_players = pd.read_csv('../core/output/batting.csv')
mean_wrc_years = pd.read_csv('../core/output/mean_wrc_by_year.csv')
mean_woba_years = pd.read_csv('../core/output/mean_woba_by_year.csv')
df_constants = pd.read_csv('../core/data/FanGraphsConstants.csv')
df_league = pd.read_csv('../core/data/FanGraphsLeagueAverages.csv')

df_woba = df_constants[['Season', 'wOBA', 'wOBAScale',
                        'wBB', 'wHBP', 'w1B', 'w2B', 'w3B', 'wHR', 'R/PA']]
df_fip = df_constants[['Season', 'runSB', 'runCS', 'R/PA', 'R/W', 'cFIP']]
batting_stats_dict = {}


def get_player(retroID):
    return df_by_player[df_by_player['retroID'] == retroID].iloc[0]


def get_player_year(retroID, year):
    player_year = df_players[(df_players['yearID'] == year) & (
        df_players['retroID'] == retroID)]
    agg = player_year.groupby('retroID').sum().reset_index()
    return agg.iloc[0]


def get_all_players_for_year(year):
    return df_players[df_players['yearID'] == year]


def get_qualifying_players_for_year(year):
    players = get_all_players_for_year(year)
    return players[players['PA'] > 110]


def get_non_pitchers_for_year(year):
    players = get_all_players_for_year(year)
    filter_list = players.apply(
        lambda player: get_pos(player['retroID']) != 'P', axis=1)
    return players[filter_list.values]


def mean_stat_for_year(stat, year):
    players = get_qualifying_players_for_year(year)
    stat_list = players.apply(batting_stats_dict[stat], axis=1)
    return stat_list.mean()


def mean_stat_for_range(stat, first_year, last_year):
    stat_total = 0
    length = last_year - first_year + 1
    for year in range(first_year, last_year + 1):
        players = df_players[df_players['yearID'] == year]
        mean = players[stat].mean()
        stat_total = stat + mean
    return stat_total / length


def league_runs_per_pa(first_year, last_year):
    total = 0
    tenure = last_year - first_year + 1
    for year in range(first_year, last_year + 1):
        total = total + get_league_average_runs_per_plate_appearance(year)
    return total / tenure


def get_league_average_runs_per_plate_appearance(year):
    return df_woba[df_woba['Season'] == year]['R/PA'].iloc[0].round(3)


def get_pa_by_league_and_year(league, year):
    return df_league[df_league['Season'] == year]['PA_{}'.format(league)].iloc[0]


def get_league_average_stat_for_year(stat, year):
    return df_league[df_league['Season'] == year][stat].iloc[0]


def avg(player):
    return player['H'] / player['AB'] if player['AB'] > 0 else 0


batting_stats_dict['AVG'] = avg


def obp(player):
    # We do not count sacrifice hits/bunts for PA in OBP
    pa = player['PA'] - player['SH'] - player['SF']
    pa = 1 if pa == 0 else player['PA']
    return (player['H'] + player['BB'] + player['HBP']) / pa


batting_stats_dict['OBP'] = obp


def slg(player):
    ab = 1 if player['AB'] == 0 else player['AB']
    return (player['1B'] + 2*player['2B'] + 3*player['3B'] + 4*player['HR']) / ab


batting_stats_dict['SLG'] = slg


def ops(player):
    _obp = obp(player)
    _slg = slg(player)
    return _obp + _slg


batting_stats_dict['OPS'] = ops


def tango_relative_ops(player):
    # Tom Tango estimates OBP to be 1.7x as important as SLG
    _obp = obp(player)
    _slg = slg(player)
    return (1.7 * _obp + _slg) * (5 / 5.7)


def rc(player):
    numerator1 = player['H'] + player['BB'] - \
        player['CS'] + player['HBP'] - player['GIDP']
    total_bases = player['H'] + 2*player['2B'] + \
        3*player['3B'] + 4*player['HR']
    numerator2 = total_bases + \
        (0.26 * (player['BB'] - player['IBB'] + player['HBP']))
    numerator3 = 0.52 * (player['SH'] + player['SF'] + player['SB'])
    denominator = player['AB'] + player['BB'] + \
        player['HBP'] + player['SH'] + player['SF']
    return (numerator1 * numerator2 + numerator3) / denominator


batting_stats_dict['RC'] = rc


def ops_plus(player, year, park_adjustment=1):
    return (ops(player) / (park_adjustment * get_league_average_stat_for_year('OPS', year))) * 100


def ops_plus_career(player, park_adjustment=1):
    first_year = player['debutYear'].item()
    last_year = player['finalYear'].item()
    tenure = last_year - first_year + 1
    league_ops = 0
    for year in range(first_year, last_year + 1):
        league_ops = league_ops + get_league_average_stat_for_year('OPS', year)
    league_ops = league_ops / tenure
    return (ops(player) / (league_ops * park_adjustment)) + 100


def woba(player, year, woba_data=None):
    if woba_data is None:
        woba_data = df_woba[df_woba['Season'] == year].iloc[0]
    bb = (player['BB'] - player['IBB']) * woba_data['wBB']
    hbp = player['HBP'] * woba_data['wHBP']
    s = player['1B'] * woba_data['w1B']
    d = player['2B'] * woba_data['w2B']
    t = player['3B'] * woba_data['w3B']
    hr = player['HR'] * woba_data['wHR']
    numerator = bb + hbp + s + d + t + hr
    denominator = player['AB'] + player['BB'] - \
        player['IBB'] + player['SF'] + player['HBP']
    return numerator / denominator if denominator != 0 else 0


def mean_woba(year, woba_data=None):
    if woba_data is None:
        woba_data = df_woba[df_woba['Season'] == year].iloc[0]
    players = get_all_players_for_year(year)
    woba_array = players.apply(
        lambda player: woba(player, year, woba_data), axis=1)
    return woba_array.mean()


def woba_career(player):
    first_year = player['debutYear'].item()
    last_year = player['finalYear'].item()
    tenure = last_year - first_year + 1
    woba_result = 0
    for year in range(first_year, last_year + 1):
        woba_result = woba_result + woba(player, year)
    return woba_result / tenure


def wraa(player, year):
    woba_data = df_woba[df_woba['Season'] == year].iloc[0]
    woba_scale = woba_data['wOBAScale']
    numerator = woba(player, year) - \
        get_league_average_stat_for_year('wOBA', year)
    return ((numerator / woba_scale) * player['PA']).round(3)


def wrc(player, year, woba_data=None):
    if woba_data is None:
        woba_data = df_woba[df_woba['Season'] == year].iloc[0]
    woba_scale = woba_data['wOBAScale']
    _woba = woba(player, year)
    _league_woba = get_league_average_stat_for_year('wOBA', year)
    _league_rppa = get_league_average_runs_per_plate_appearance(year)
    _pa = player['PA']
    adjusted_woba = woba(player, year, woba_data) - \
        get_league_average_stat_for_year('wOBA', year)
    return ((adjusted_woba / woba_scale) +
            get_league_average_runs_per_plate_appearance(year)) * player['PA']


def wrc_plus(player, year, park_factor=1):
    league = 'NL' if player['NL'] else 'AL'
    numerator1 = (wraa(player, year) / player['PA'])
    numerator2 = get_league_average_runs_per_plate_appearance(year)
    numerator3 = park_factor * (numerator2)
    denominator = get_league_average_stat_for_year('wRC_{}'.format(
        league), year) / get_pa_by_league_and_year(league, year)
    return (((numerator1 + numerator2) + (numerator2 - numerator3)) / (denominator)) * 100


def wrc_plus_career(player):
    first_year = player['debutYear'].item()
    last_year = player['finalYear'].item()
    print('First: {}\nLast: {}'.format(first_year, last_year))
    tenure = last_year - first_year + 1
    wrc_plus_total = 0
    for year in (first_year, last_year):
        print(year)
        wrc_plus_total = wrc_plus_total + wrc_plus(player, year)
    return wrc_plus_total / tenure
