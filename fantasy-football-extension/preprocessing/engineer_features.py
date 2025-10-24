import pandas as pd
import numpy as np

def engineer_features(data, position):
    """
    Create comprehensive features including environment context
    """
    
    # Sort by player and time
    data = data.sort_values(['player_id', 'season', 'week'])
    
    # ============ RECENT PERFORMANCE TREND ============
    # Rolling average of fantasy points (3-game trend)
    if 'fantasy_points_ppr' in data.columns:
        data['fantasy_trend'] = (
            data.groupby('player_id')['fantasy_points_ppr']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    # Opponent defensive strength (points allowed to position)
    if {'opponent_team', 'position', 'season', 'fantasy_points_ppr'}.issubset(data.columns):
        data['opp_def_rank_vs_pos'] = (
            data.groupby(['opponent_team', 'position', 'season'])['fantasy_points_ppr']
            .transform(lambda x: x.rank(pct=True))
        )
    else:
        data['opp_def_rank_vs_pos'] = np.nan
    
    # ============ HOME/AWAY & ENVIRONMENTAL FEATURES ============
    # Home field advantage
    if {'recent_team', 'home_team'}.issubset(data.columns):
        data['is_home'] = (data['recent_team'] == data['home_team']).astype(int)

    # Division game flag if division columns available (not specified in merge)
    if {'team_division', 'opponent_division'}.issubset(data.columns):
        data['is_division_game'] = (data['team_division'] == data['opponent_division']).astype(int)

    # Weather-based features if provided
    if 'temp' in data.columns:
        data['is_cold'] = (data['temp'] < 40).astype(int)
        data['is_hot'] = (data['temp'] > 85).astype(int)

    if 'roof' in data.columns:
        data['is_dome'] = data['roof'].isin(['dome', 'closed']).astype(int)

    # ============ GAME CONTEXT ============
    # Vegas lines and derived features
    if {'spread_line', 'total_line'}.issubset(data.columns):
        data['is_favored'] = (data['spread_line'] < 0).astype(int)
        data['spread_magnitude'] = abs(data['spread_line'])
        data['game_total'] = data['total_line']
        data['implied_team_total'] = (data['game_total'] / 2) - (data['spread_line'] / 2)
    else:
        for col in ['is_favored', 'spread_magnitude', 'game_total', 'implied_team_total']:
            data[col] = np.nan

    # ============ INJURY & GAMELOAD ============
    # Injury severity mapping
    injury_map = {'': 0, 'Healthy': 0, 'Questionable': 1, 'Doubtful': 2, 'Out': 3}
    if 'report_status' in data.columns:
        data['injury_severity'] = data['report_status'].map(injury_map).fillna(0)
    else:
        data['injury_severity'] = 0

    # Games played this season (fatigue factor)
    data['games_played'] = data.groupby(['player_id', 'season']).cumcount() + 1

    # Early and late season flags
    if 'week' in data.columns:
        data['is_early_season'] = (data['week'] <= 4).astype(int)
        data['is_late_season'] = (data['week'] >= 14).astype(int)
        data['week_of_season'] = data['week']

    # Days of rest between games
    if 'gameday' in data.columns:
        data['gameday'] = pd.to_datetime(data['gameday'])
        data['days_rest'] = data.groupby('player_id')['gameday'].diff().dt.days.fillna(7)
    else:
        data['days_rest'] = 7  # Default rest value

    # ============ USAGE & OPPORTUNITY ============
    # Snap percentage and its rolling average
    if {'offense_snaps', 'offense_pct'}.issubset(data.columns):
        data['snap_pct'] = data['offense_pct']
        data['snap_pct_rolling_3'] = (
            data.groupby('player_id')['snap_pct'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        )

    # Target/touch share (calculated if targets and team totals available)
    if position in ['WR', 'TE']:
        if {'targets', 'team', 'week'}.issubset(data.columns):
            data['target_share'] = data['targets'] / data.groupby(['team', 'week'])['targets'].transform('sum')
    elif position == 'RB':
        if {'carries', 'team', 'week'}.issubset(data.columns):
            data['carry_share'] = data['carries'] / data.groupby(['team', 'week'])['carries'].transform('sum')

    # ============ NEXT GEN STATS FEATURES ============
    # Use raw NGS metrics directly, avoid rolling averages as already computed in preprocessing

    # ============ POSITION-SPECIFIC ADDITIONS ============
    if position == 'QB':
        # Pressure rate and deep ball rate (if available)
        if {'times_pressured', 'dropbacks'}.issubset(data.columns):
            data['pressure_rate'] = data['times_pressured'] / data['dropbacks']
        if {'deep_attempts', 'attempts'}.issubset(data.columns):
            data['deep_rate'] = data['deep_attempts'] / data['attempts']

    if position == 'RB':
        if 'redzone_touches' in data.columns:
            data['redzone_touches_rolling_3'] = (
                data.groupby('player_id')['redzone_touches']
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

    if position in ['WR', 'TE']:
        if 'air_yards_share' in data.columns:
            data['air_yards_share_rolling_3'] = (
                data.groupby('player_id')['air_yards_share']
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

    print(f"Feature engineering complete for {position}")
    print(f"Total features: {len(data.columns)}")

    return data

if __name__ == '__main__':
    positions = ['QB', 'RB', 'WR', 'TE']
    
    for pos in positions:
        data = pd.read_parquet(f'data/processed/{pos}_data.parquet')
        data = engineer_features(data, pos)
        data.to_parquet(f'data/processed/{pos}_featured.parquet')
        print(f"Saved {pos} featured data\n")
