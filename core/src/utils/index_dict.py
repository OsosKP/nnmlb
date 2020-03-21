import pandas as pd

catchers = pd.read_csv('../core/output/catchers.csv').reset_index()
fielders = pd.read_csv('../core/output/fielders.csv').reset_index()
pitchers = pd.read_csv('../core/output/pitchers.csv').reset_index()
catchers_dict = catchers.set_index('index').to_dict()['retroID']
fielders_dict = fielders.set_index('index').to_dict()['retroID']
pitchers_dict = pitchers.set_index('index').to_dict()['retroID']
players_dict = {
    'C': catchers_dict,
    '1B': fielders_dict,
    '2B': fielders_dict,
    'SS': fielders_dict,
    '3B': fielders_dict,
    'OF': fielders_dict,
    'P': pitchers_dict
}


def get_catcher(catcher_id):
    return catchers_dict[catcher_id] if catchers_dict[catcher_id] is not None else ''


def get_fielder(fielder_id):
    return fielders_dict[fielder_id] if fielders_dict[fielder_id] is not None else ''


def get_pitcher(pitcher_id):
    return pitchers_dict[pitcher_id] if pitchers_dict[pitcher_id] is not None else ''


def get_player(player_id, position):
    return players_dict[position][player_id]
