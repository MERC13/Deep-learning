# preprocessing/engineer_features.py
import pandas as pd
import numpy as np

def engineer_features(data, position):
    """
    Create comprehensive features including environment context
    """
    
    # Sort by player and time
    data = data.sort_values(['player_id', 'season', 'week'])
    
    # ============ PLAYER PERFORMANCE FEATURES ============
    
    # Rolling averages (recent performance)
    rolling_windows = [2, 3, 5]
    stats_to_roll = ['fantasy_points', 'yards', 'touchdowns', 'targets', 'carries']
    
    for window in rolling_windows:
        for stat in stats_to_roll:
            if stat in data.columns:
                data[f'{stat}_rolling_{window}'] = (
                    data.groupby('player_id')[stat]
                    .rolling(window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
    
    # Season-to-date averages
    data['fantasy_ppg_ytd'] = (
        data.groupby(['player_id', 'season'])['fantasy_points']
        .expanding().mean()
        .reset_index(0, drop=True)
    )
    
    # Momentum (trend over last 3 games)
    data['fantasy_trend'] = (
        data['fantasy_points_rolling_3'] - 
        data.groupby('player_id')['fantasy_points_rolling_3'].shift(3)
    )
    
    # ============ OPPONENT FEATURES ============
    
    # Opponent defensive strength (points allowed to position)
    data['opp_def_rank_vs_pos'] = (
        data.groupby(['opponent', 'position', 'season'])['fantasy_points']
        .transform(lambda x: x.rank(pct=True))
    )
    
    # Opponent injuries (key defensive players out)
    # This would come from injury reports
    data['opp_def_injuries'] = 0  # Placeholder - requires injury data
    
    # ============ HOME/AWAY & ENVIRONMENTAL ============
    
    # Home field advantage
    data['is_home'] = (data['team'] == data['home_team']).astype(int)
    
    # Division game (more familiarity, different dynamics)
    data['is_division_game'] = (
        data['team_division'] == data['opponent_division']
    ).astype(int)
    
    # Weather impact (for outdoor games)
    if 'temperature' in data.columns:
        data['is_cold'] = (data['temperature'] < 40).astype(int)
        data['is_hot'] = (data['temperature'] > 85).astype(int)
        data['has_precipitation'] = (
            (data['weather'].str.contains('rain|snow', case=False, na=False))
            .astype(int)
        )
    
    # Stadium type (dome vs outdoor)
    data['is_dome'] = data['roof'].isin(['dome', 'closed']).astype(int)
    
    # ============ GAME CONTEXT ============
    
    # Vegas betting lines (game script predictor)
    data['is_favored'] = (data['spread'] < 0).astype(int)
    data['spread_magnitude'] = abs(data['spread'])
    data['game_total'] = data['over_under']  # Expected total points
    
    # Implied team total
    data['implied_team_total'] = (
        data['game_total'] / 2 - data['spread'] / 2
    )
    
    # ============ INJURY & GAME NUMBER ============
    
    # Injury status (from injury reports)
    injury_map = {'': 0, 'Questionable': 1, 'Doubtful': 2, 'Out': 3}
    if 'injury_status' in data.columns:
        data['injury_severity'] = data['injury_status'].map(injury_map).fillna(0)
    
    # Games played this season (fatigue factor)
    data['games_played'] = (
        data.groupby(['player_id', 'season']).cumcount() + 1
    )
    
    # Week of season (late season performance changes)
    data['week_of_season'] = data['week']
    data['is_early_season'] = (data['week'] <= 4).astype(int)
    data['is_late_season'] = (data['week'] >= 14).astype(int)
    
    # Days of rest since last game
    data['gameday'] = pd.to_datetime(data['gameday'])
    data['days_rest'] = (
        data.groupby('player_id')['gameday']
        .diff()
        .dt.days
        .fillna(7)
    )
    
    # ============ USAGE & OPPORTUNITY ============
    
    # Snap percentage (playing time)
    data['snap_pct'] = data['snaps'] / data['team_snaps']
    data['snap_pct_rolling_3'] = (
        data.groupby('player_id')['snap_pct']
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    # Target/touch share (opportunity)
    if position in ['WR', 'TE']:
        data['target_share'] = (
            data['targets'] / data.groupby(['team', 'week'])['targets'].transform('sum')
        )
    
    if position == 'RB':
        data['carry_share'] = (
            data['carries'] / data.groupby(['team', 'week'])['carries'].transform('sum')
        )
    
    # ============ NEXT GEN STATS FEATURES ============
    
    if position in ['WR', 'TE']:
        # Average separation from defender (NGS)
        data['avg_separation_rolling_3'] = (
            data.groupby('player_id')['avg_separation']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Cushion given by defense
        data['avg_cushion_rolling_3'] = (
            data.groupby('player_id')['avg_cushion']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    if position == 'RB':
        # Rushing efficiency (yards over expected)
        data['efficiency_rolling_3'] = (
            data.groupby('player_id')['efficiency']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Percentage of rushes against stacked boxes
        data['eight_defenders_pct_rolling_3'] = (
            data.groupby('player_id')['percent_attempts_gte_8_defenders']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    if position == 'QB':
        # Time to throw (NGS)
        data['avg_time_to_throw_rolling_3'] = (
            data.groupby('player_id')['avg_time_to_throw']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Completed air yards
        data['avg_completed_air_yards_rolling_3'] = (
            data.groupby('player_id')['avg_completed_air_yards']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    # ============ POSITION-SPECIFIC FEATURES ============
    
    if position == 'QB':
        # Pressure rate
        if 'times_pressured' in data.columns:
            data['pressure_rate'] = data['times_pressured'] / data['dropbacks']
        
        # Deep ball rate
        if 'deep_attempts' in data.columns:
            data['deep_rate'] = data['deep_attempts'] / data['attempts']
    
    if position in ['RB']:
        # Red zone usage
        data['redzone_touches_rolling_3'] = (
            data.groupby('player_id')['redzone_touches']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    if position in ['WR', 'TE']:
        # Air yards share
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
