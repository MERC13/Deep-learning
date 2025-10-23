import pandas as pd
import numpy as np

def load_and_merge_data():
    """
    Load all data sources and merge into unified dataset
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
    
    # Merge on player_id, week, season
    data = weekly.merge(
        ngs_receiving, 
        on=['player_id', 'player_name', 'week', 'season'], 
        how='left'
    )
    data = data.merge(
        ngs_rushing,
        on=['player_id', 'player_name', 'week', 'season'],
        how='left'
    )
    data = data.merge(
        ngs_passing,
        on=['player_id', 'player_name', 'week', 'season'],
        how='left'
    )
    data = data.merge(
        snaps,
        on=['player_id', 'week', 'season', 'team'],
        how='left'
    )
    
    # Add game context (opponent, home/away, vegas lines)
    data = data.merge(
        vegas,
        on=['week', 'season', 'team'],
        how='left'
    )
    
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
    Intelligent missing value imputation by position
    Strategy depends on why data is missing
    """
    
    # 1. Missing because stat doesn't apply to position
    # Example: RBs don't have passing stats -> fill with 0
    if position == 'RB':
        passing_cols = ['passing_yards', 'passing_tds', 'completions', 
                       'avg_time_to_throw', 'avg_completed_air_yards']
        data[passing_cols] = data[passing_cols].fillna(0)
    
    if position == 'QB':
        receiving_cols = ['receptions', 'receiving_yards', 'receiving_tds',
                         'avg_cushion', 'avg_separation']
        data[receiving_cols] = data[receiving_cols].fillna(0)
    
    # 2. Missing because player didn't play that week
    # Check if player was injured or inactive
    data['was_inactive'] = (data['snaps'] == 0) | (data['snaps'].isna())
    
    # For inactive weeks, set fantasy_points = 0
    data.loc[data['was_inactive'], 'fantasy_points'] = 0
    
    # 3. Missing NGS data (not all plays tracked early seasons)
    # Use forward-fill then backward-fill for NGS metrics
    ngs_cols = [col for col in data.columns if 'avg_' in col or 'efficiency' in col]
    data[ngs_cols] = data.groupby('player_id')[ngs_cols].fillna(method='ffill').fillna(method='bfill')
    
    # 4. Missing snap count data -> use median for player
    data['snap_pct'] = data['snaps'] / data['team_total_snaps']
    data['snap_pct'] = data.groupby('player_id')['snap_pct'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # 5. Missing opponent defensive stats -> use league average
    def_stats = ['opponent_pass_def_rank', 'opponent_run_def_rank']
    for col in def_stats:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mean())
    
    # 6. Remaining missing values -> forward fill by player, then 0
    data = data.groupby('player_id').fillna(method='ffill').fillna(0)
    
    print(f"Missing value handling complete for {position}")
    print(f"Remaining NaNs: {data.isna().sum().sum()}")
    
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
