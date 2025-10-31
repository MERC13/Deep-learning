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
    depth_charts = pd.read_parquet('data/raw/depth.parquet')

    # Select features for predicting fantasy points
    weekly_cols = ['player_id', 'player_display_name', 'season', 'week',
                   'position', 'recent_team', 'opponent_team', 'fantasy_points', 'fantasy_points_ppr']
    ngs_rec_cols = ['season', 'week', 'player_display_name', 'player_gsis_id',
                    'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
                    'percent_share_of_intended_air_yards', 'avg_yac', 'avg_expected_yac',
                    'avg_yac_above_expectation', 'catch_percentage', 'receptions',
                    'targets', 'yards', 'rec_touchdowns']
    ngs_rush_cols = ['season', 'week', 'player_display_name', 'player_gsis_id',
                     'efficiency', 'percent_attempts_gte_eight_defenders', 
                     'avg_time_to_los', 'avg_rush_yards', 'rush_yards_over_expected', 'rush_attempts',
                     'rush_yards', 'rush_touchdowns', 'expected_rush_yards']
    ngs_pass_cols = ['season', 'week', 'player_display_name', 'player_gsis_id',
                     'avg_time_to_throw', 'avg_completed_air_yards', 
                     'avg_intended_air_yards', 'avg_air_yards_differential',
                     'aggressiveness', 'completion_percentage', 'avg_air_distance', 'max_air_distance',
                     'attempts', 'pass_yards', 'pass_touchdowns', 'interceptions', 'passer_rating', 'completions']
    snaps_cols = ['season', 'week', 'player',
                  'offense_snaps', 'offense_pct', 'st_snaps', 'st_pct']
    injuries_cols = ['season', 'week', 'gsis_id', 'full_name',
                     'report_primary_injury', 'report_status', 'practice_status']
    depth_cols = ['season', 'week', 'full_name', 'gsis_id',
                  'depth_team']
    
    
    # Filter to relevant columns
    weekly = weekly[weekly_cols]
    ngs_receiving = ngs_receiving[ngs_rec_cols]
    ngs_rushing = ngs_rushing[ngs_rush_cols]
    ngs_passing = ngs_passing[ngs_pass_cols]
    snaps = snaps[snaps_cols]
    injuries = injuries[injuries_cols]
    depth_charts = depth_charts[depth_cols]
    
    # Normalize key types and clean strings to improve join quality
    def _to_int_series(s: pd.Series) -> pd.Series:
        # Use pandas nullable Int64 to support missing values while enabling numeric sort/merge
        return pd.to_numeric(s, errors='coerce').astype('Int64')

    for df in [weekly, ngs_receiving, ngs_rushing, ngs_passing, snaps, injuries, depth_charts]:
        if 'season' in df.columns:
            df['season'] = _to_int_series(df['season'])
        if 'week' in df.columns:
            df['week'] = _to_int_series(df['week'])
        # Strip whitespace in common name keys
        for name_col in ['player_display_name', 'full_name', 'player']:
            if name_col in df.columns and df[name_col].dtype == object:
                df[name_col] = df[name_col].astype(str).str.strip()
    
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

    data = data.merge(
        snaps,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['player', 'season', 'week'],
        how='left',
        suffixes=('', '_snaps')
    )
    data = data.merge(
        injuries,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['full_name', 'season', 'week'],
        how='left',
        suffixes=('', '_inj')
    )
    
    data = data.merge(
        depth_charts,
        left_on=['player_display_name', 'season', 'week'],
        right_on=['full_name', 'season', 'week'],
        how='left',
        suffixes=('', '_depth')
    )
    
    print(f"Merged dataset shape: {data.shape}")
    return data

def filter_by_position(data, position):
    """
    Filter dataset to specific position with position-relevant columns
    """
    position_filters = {
        'QB': ['QB'],
        'RB': ['RB', 'FB'],
        'WR': ['WR'],
        'TE': ['TE']
    }
    
    filtered = data[data['position'].isin(position_filters[position])].copy()
    
    print(f"Filtered to {len(filtered)} {position} player-weeks")
    return filtered

def handle_missing_values(data, position):
    """
    Handle missing values with position-aware logic and time-aware imputation.

    Rules
    - Position-inapplicable feature families get filled with 0 (e.g., passing metrics for RB/WR/TE, receiving for QB).
    - If a player didn't play (offense_snaps == 0 or NaN), set fantasy_points_ppr to 0 for that week.
    - Time-series NGS metrics are forward/backward filled per player along their weekly timeline.
    - Categorical statuses default to sensible labels (Healthy/None/Unknown).
    - Remaining numeric NaNs are filled with 0 as a safe default.
    """
    data = data.copy()

    # Ensure season/week are numeric for stable time sorting
    for col in ['season', 'week']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    # Clean common string identifiers
    for name_col in ['player_display_name', 'full_name', 'player']:
        if name_col in data.columns and data[name_col].dtype == object:
            data[name_col] = data[name_col].astype(str).str.strip()

    # Choose a player key available in the merged dataset
    possible_keys = ['player_display_name', 'full_name', 'player', 'player_id', 'player_gsis_id', 'gsis_id']
    player_key = next((k for k in possible_keys if k in data.columns), None)

    # Helper: time-aware ffill/bfill within each player's weekly timeline
    def ffill_bfill_by_player(df: pd.DataFrame, col: str) -> pd.Series:
        if player_key is None or col not in df.columns:
            return df[col]
        # Preserve original order and create a stable per-row order key
        order_key = np.arange(len(df))
        tmp = df[[player_key, 'season', 'week', col]].copy()
        tmp['_order'] = order_key
        # Sort by player, season, week, then original order to stabilize
        sort_cols = [player_key]
        if 'season' in tmp.columns:
            sort_cols.append('season')
        if 'week' in tmp.columns:
            sort_cols.append('week')
        sort_cols.append('_order')
        tmp_sorted = tmp.sort_values(sort_cols)
        def safe_fill(x: pd.Series) -> pd.Series:
            return x.ffill().bfill() if not x.isna().all() else x
        tmp_sorted[col] = tmp_sorted.groupby(player_key, dropna=False)[col].transform(safe_fill)
        # Restore original order
        tmp_restored = tmp_sorted.sort_values('_order')
        return tmp_restored[col].reset_index(drop=True)

    # Define feature families present in our merged dataset (keep only columns that exist)
    passing_family = [
        'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards',
        'avg_air_yards_differential', 'aggressiveness', 'completion_percentage',
        'avg_air_distance', 'max_air_distance', 'attempts', 'pass_yards',
        'pass_touchdowns', 'interceptions', 'passer_rating', 'completions'
    ]
    receiving_family = [
        'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
        'percent_share_of_intended_air_yards', 'avg_yac', 'avg_expected_yac',
        'avg_yac_above_expectation', 'catch_percentage', 'receptions',
        'targets', 'yards', 'rec_touchdowns'
    ]
    rushing_family = [
        'efficiency', 'percent_attempts_gte_eight_defenders', 'avg_time_to_los',
        'avg_rush_yards', 'rush_yards_over_expected', 'rush_attempts',
        'rush_yards', 'rush_touchdowns', 'expected_rush_yards',
        'rush_yards_over_expected_per_att'
    ]

    passing_cols = [c for c in passing_family if c in data.columns]
    receiving_cols = [c for c in receiving_family if c in data.columns]
    rushing_cols = [c for c in rushing_family if c in data.columns]

    # 1) Position-inapplicable families -> fill with 0 when missing
    if position == 'QB':
        cols_to_zero = receiving_cols + rushing_cols  # QBs may rush, but missing values imply 0
    elif position == 'RB':
        cols_to_zero = passing_cols
    elif position in ('WR', 'TE'):
        cols_to_zero = passing_cols + rushing_cols
    else:
        cols_to_zero = passing_cols + receiving_cols + rushing_cols

    if cols_to_zero:
        data[cols_to_zero] = data[cols_to_zero].fillna(0)

    # 2) Player inactive -> fantasy points 0
    if 'offense_snaps' in data.columns:
        # Ensure numeric for comparisons
        data['offense_snaps'] = pd.to_numeric(data['offense_snaps'], errors='coerce')
        data['was_inactive'] = (data['offense_snaps'] == 0) | (data['offense_snaps'].isna())
        if 'fantasy_points_ppr' in data.columns:
            data.loc[data['was_inactive'], 'fantasy_points_ppr'] = 0

    # 3) Time-series imputation for NGS metrics
    ngs_like_cols = [
        c for c in data.columns
        if any(x in c for x in ['avg_', 'efficiency', 'percent_', 'above_expectation', 'distance', 'rating', 'yards'])
    ]
    # We'll include completion/catch percentage as well
    for c in ['completion_percentage', 'catch_percentage', 'expected_completion_percentage', 'completion_percentage_above_expectation', 'avg_air_yards_to_sticks']:
        if c in data.columns and c not in ngs_like_cols:
            ngs_like_cols.append(c)

    for col in ngs_like_cols:
        try:
            data[col] = ffill_bfill_by_player(data, col)
        except Exception:
            pass

    # 4) Snap percentage imputation: player's median
    if 'offense_pct' in data.columns and player_key is not None:
        def safe_median_fill(s: pd.Series) -> pd.Series:
            if s.isna().all():
                return s
            return s.fillna(s.median())
        data['offense_pct'] = (
            data.groupby(player_key)['offense_pct']
                .transform(safe_median_fill)
        )

    # 5) Categorical defaults
    for col in ['report_status', 'practice_status']:
        if col in data.columns:
            data[col] = data[col].fillna('Healthy')
    if 'report_primary_injury' in data.columns:
        data['report_primary_injury'] = data['report_primary_injury'].fillna('None')
    if 'roof' in data.columns:
        data['roof'] = data['roof'].fillna('Unknown')
    if 'surface' in data.columns:
        data['surface'] = data['surface'].fillna('Unknown')
    for wcol in ['temp', 'wind']:
        if wcol in data.columns:
            data[wcol] = pd.to_numeric(data[wcol], errors='coerce').fillna(0)
    if 'depth_team' in data.columns:
        data['depth_team'] = data['depth_team'].fillna('Unknown')

    # 6) Final: fill remaining numeric gaps with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)

    # Debug summary
    total_nans = int(data.isna().sum().sum())
    print(f"Missing value handling complete for {position}")
    print(f"Final NaNs remaining: {total_nans}")
    if total_nans > 0:
        nan_cols = data.isna().sum()
        print(f"Columns with NaNs:\n{nan_cols[nan_cols > 0]}")

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
