import pandas as pd
import numpy as np

def load_and_merge_data():
    """
    Load all data sources and merge into unified dataset for fantasy points prediction
    """
    # Load weekly stats
    weekly = pd.read_parquet('data/raw/weekly.parquet')
    
    # Load NGS data
    ngs_receiving = pd.read_parquet('data/raw/ngs_receiving.parquet')
    ngs_rushing = pd.read_parquet('data/raw/ngs_rushing.parquet')
    ngs_passing = pd.read_parquet('data/raw/ngs_passing.parquet')
    
    # Load context data
    snaps = pd.read_parquet('data/raw/snaps.parquet')
    injuries = pd.read_parquet('data/raw/injuries.parquet')
    vegas = pd.read_parquet('data/raw/vegas.parquet')

    # Select features for predicting fantasy points
    # Target: fantasy_points_ppr
    # Keep identifiers and target only from weekly
    weekly_cols = ['player_id', 'player_name', 'player_display_name', 'position', 
                   'season', 'week', 'opponent_team', 'recent_team', 
                   'fantasy_points_ppr']  # TARGET - remove non-PPR version
    
    # NGS receiving: separation, cushion, and efficiency metrics
    ngs_rec_cols = ['season', 'week', 'player_display_name', 'team_abbr',
                    'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
                    'percent_share_of_intended_air_yards', 'avg_yac', 
                    'avg_yac_above_expectation', 'catch_percentage']
    
    # NGS rushing: efficiency and workload metrics
    ngs_rush_cols = ['season', 'week', 'player_display_name', 'team_abbr',
                     'efficiency', 'percent_attempts_gte_eight_defenders', 
                     'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected',
                     'rush_yards_over_expected_per_att', 'rush_pct_over_expected']
    
    # NGS passing: decision-making and arm talent metrics
    ngs_pass_cols = ['season', 'week', 'player_display_name', 'team_abbr',
                     'avg_time_to_throw', 'avg_completed_air_yards', 
                     'avg_intended_air_yards', 'avg_air_yards_differential',
                     'aggressiveness', 'completion_percentage_above_expectation']
    
    # Snaps: workload is a strong predictor of fantasy points
    snaps_cols = ['season', 'week', 'player', 'team', 'opponent',
                  'offense_snaps', 'offense_pct', 'defense_snaps', 'defense_pct']
    
    # Injuries: status matters for production
    injuries_cols = ['season', 'week', 'gsis_id', 'full_name',
                     'report_primary_injury', 'report_status', 'practice_status']
    
    # Vegas: line movement, rest days, and game environment
    vegas_cols = ['season', 'week', 'away_team', 'home_team', 'spread_line',
                  'total_line', 'temp', 'wind', 'roof']  # Added roof for covered stadiums
    
    # Filter to relevant columns
    weekly = weekly[weekly_cols]
    ngs_receiving = ngs_receiving[ngs_rec_cols]
    ngs_rushing = ngs_rushing[ngs_rush_cols]
    ngs_passing = ngs_passing[ngs_pass_cols]
    snaps = snaps[snaps_cols]
    injuries = injuries[injuries_cols]
    vegas = vegas[vegas_cols]
    
    # Merge on player_display_name (common key across NGS datasets)
    data = weekly.merge(
        ngs_receiving, 
        left_on=['player_display_name', 'season', 'week'],
        right_on=['player_display_name', 'season', 'week'],
        how='left',
        suffixes=('', '_rec')
    )
    
    data = data.merge(
        ngs_rushing,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['player_display_name', 'season', 'week'],
        how='left',
        suffixes=('', '_rush')
    )
    
    data = data.merge(
        ngs_passing,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['player_display_name', 'season', 'week'],
        how='left',
        suffixes=('', '_pass')
    )
    
    # Merge snaps on player name and team
    data = data.merge(
        snaps,
        left_on=['player_display_name', 'recent_team', 'season', 'week'],
        right_on=['player', 'team', 'season', 'week'],
        how='left'
    )
    
    # Merge injuries
    data = data.merge(
        injuries,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['full_name', 'season', 'week'],
        how='left',
        suffixes=('', '_inj')
    )
    
    # Merge vegas context (handling home/away teams)
    data_away = data.merge(
        vegas,
        left_on=['recent_team', 'season', 'week', 'opponent_team'],
        right_on=['away_team', 'season', 'week', 'home_team'],
        how='left'
    )
    
    data_home = data.merge(
        vegas,
        left_on=['recent_team', 'season', 'week', 'opponent_team'],
        right_on=['home_team', 'season', 'week', 'away_team'],
        how='left'
    )
    
    data = data_away.fillna(data_home)
    
    print(f"Merged dataset shape: {data.shape}")
    return data

def filter_by_position(data, position):
    """
    Filter dataset to specific position with position-relevant columns
    """
    position_filters = {
        'QB': ['QB'],
        'RB': ['RB', 'FB'],  # Include fullbacks
        'WR': ['WR'],
        'TE': ['TE']
    }
    
    filtered = data[data['position'].isin(position_filters[position])].copy()
    
    print(f"Filtered to {len(filtered)} {position} player-weeks")
    return filtered

def handle_missing_values(data, position):
    """
    Intelligent missing value imputation by position and data source.
    Strategy depends on why data is missing (position-irrelevant, inactive, data gaps, etc.)
    """
    data = data.copy()
    
    # 1. Missing because stat doesn't apply to position -> fill with 0
    position_inapplicable = {
        'RB': {
            'passing': ['avg_time_to_throw', 'avg_completed_air_yards', 
                       'avg_intended_air_yards', 'avg_air_yards_differential',
                       'aggressiveness', 'completion_percentage_above_expectation']
        },
        'WR': {
            'passing': ['avg_time_to_throw', 'avg_completed_air_yards', 
                       'avg_intended_air_yards', 'avg_air_yards_differential',
                       'aggressiveness', 'completion_percentage_above_expectation'],
            'rushing': ['efficiency', 'percent_attempts_gte_eight_defenders',
                       'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected',
                       'rush_yards_over_expected_per_att', 'rush_pct_over_expected']
        },
        'TE': {
            'passing': ['avg_time_to_throw', 'avg_completed_air_yards', 
                       'avg_intended_air_yards', 'avg_air_yards_differential',
                       'aggressiveness', 'completion_percentage_above_expectation'],
            'rushing': ['efficiency', 'percent_attempts_gte_eight_defenders',
                       'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected',
                       'rush_yards_over_expected_per_att', 'rush_pct_over_expected']
        },
        'QB': {
            'receiving': ['avg_cushion', 'avg_separation', 'avg_intended_air_yards',
                         'percent_share_of_intended_air_yards', 'avg_yac', 
                         'avg_yac_above_expectation', 'catch_percentage'],
            'rushing': ['efficiency', 'percent_attempts_gte_eight_defenders',
                       'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected',
                       'rush_yards_over_expected_per_att', 'rush_pct_over_expected']
        }
    }
    
    if position in position_inapplicable:
        for stat_type, cols in position_inapplicable[position].items():
            for col in cols:
                if col in data.columns:
                    data[col] = data[col].fillna(0)
    
    # 2. Missing because player was inactive/didn't play
    # If offense_snaps is 0 or NaN, player didn't play -> fantasy_points = 0
    if 'offense_snaps' in data.columns:
        data['was_inactive'] = (data['offense_snaps'] == 0) | (data['offense_snaps'].isna())
        data.loc[data['was_inactive'], 'fantasy_points_ppr'] = 0
    
    # 3. Missing NGS metrics (not all seasons/weeks tracked)
    # Use player-level forward fill, then backward fill
    ngs_cols = [col for col in data.columns 
                if any(x in col for x in ['avg_', 'efficiency', 'percent_', 'above_expectation'])]
    
    for col in ngs_cols:
        if col in data.columns:
            # Forward fill within each player's timeline
            def safe_fill(x):
                if x.isna().all():
                    return x  # Return as-is if all NaN
                return x.fillna(method='ffill').fillna(method='bfill')
            
            data[col] = data.groupby('player_id')[col].transform(safe_fill)
    
    # 4. Missing snap count data -> impute with player's season median
    if 'offense_pct' in data.columns:
        def safe_median(x):
            if x.isna().all() or len(x.dropna()) == 0:
                return x  # Return as-is if all NaN
            return x.fillna(x.median())
        
        data['offense_pct'] = data.groupby('player_id')['offense_pct'].transform(safe_median)
    
    # 5. Missing injury status -> assume no report = healthy
    injury_status_cols = ['report_status', 'practice_status']
    for col in injury_status_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Healthy')
    
    # 6. Missing primary injury -> no injury
    if 'report_primary_injury' in data.columns:
        data['report_primary_injury'] = data['report_primary_injury'].fillna('None')
    
    # 7. Missing Vegas metrics (weather, spreads) -> use league average or median
    vegas_numeric = ['spread_line', 'total_line', 'temp', 'wind']
    for col in vegas_numeric:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # 8. Missing roof (stadium type) -> impute with mode or 'Unknown'
    if 'roof' in data.columns:
        data['roof'] = data['roof'].fillna('Unknown')
    
    # 9. Catch percentage, efficiency metrics might be 0 if no attempts
    # Don't fill these to 0 - keep NaN so model knows it wasn't applicable
    efficiency_cols = ['catch_percentage', 'efficiency', 'aggressiveness']
    for col in efficiency_cols:
        if col in data.columns:
            # Only fill when we have data to interpolate
            def safe_fill(x):
                if x.isna().all():
                    return x  # Return as-is if all NaN
                return x.fillna(method='ffill').fillna(method='bfill')
            
            data[col] = data.groupby('player_id')[col].transform(safe_fill)
    
    # 10. Final pass: remaining NaNs -> 0 (safest default)
    remaining_nans = data.select_dtypes(include=[np.number]).columns
    data[remaining_nans] = data[remaining_nans].fillna(0)
    
    print(f"Missing value handling complete for {position}")
    print(f"Final NaNs remaining: {data.isna().sum().sum()}")
    if data.isna().sum().sum() > 0:
        print(f"Columns with NaNs:\n{data.isna().sum()[data.isna().sum() > 0]}")
    
    return data

if __name__ == '__main__':
    # Load and merge all data
    data = load_and_merge_data()
    
    # Process each position separately
    positions = ['QB', 'RB', 'WR', 'TE']
    
    for pos in positions:
        pos_data = filter_by_position(data, pos)
        pos_data = handle_missing_values(pos_data, pos)
        
        # Save position-specific dataset
        pos_data.to_parquet(f'data/processed/{pos}_data.parquet')
        print(f"Saved {pos} data: {len(pos_data)} rows\n")
