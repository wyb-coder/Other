"""
Phase 2: Distribution-Aware Data Augmentation Script
=====================================================
This script performs advanced data preprocessing and augmentation for the
traffic monitoring data, expanding 2 days (Fri + Sat) to 7 days.

Key Features:
1. Data Cleaning: Remove anomalies (negative values, unrealistic speeds)
2. Distribution Learning: Learn hourly patterns separately for workday/weekend
3. Cross-Domain Augmentation: Use Friday template for workdays, Saturday for weekends
4. Friday Evening Effect: Apply β=1.15 coefficient for school dismissal peak
5. Noise Injection: Prevent identical rows with Gaussian noise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================================
# Configuration
# ============================================================================
INPUT_FILE = "../../Data/82.csv"
OUTPUT_FILE = "../../Data/82_processed.csv"
STATS_FILE = "../../Data/processing_stats.txt"

# Augmentation parameters
FRIDAY_EVENING_COEFFICIENT = 1.15  # β for Friday 17:00+ traffic boost
FRIDAY_EVENING_START_HOUR = 17
NOISE_MEAN = 0
NOISE_STD = 0.03  # 3% Gaussian noise

# ============================================================================
# Step 1: Data Loading and Cleaning
# ============================================================================
def load_and_clean_data(filepath):
    """Load CSV and remove anomalies."""
    print("=" * 60)
    print("Step 1: Loading and Cleaning Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Original rows: {len(df)}")
    
    # Parse datetime
    df['data_time'] = pd.to_datetime(df['data_time'])
    
    # Record original stats
    original_count = len(df)
    
    # Remove negative volume
    df = df[df['volume'] >= 0]
    neg_vol_removed = original_count - len(df)
    print(f"  - Removed {neg_vol_removed} rows with negative volume")
    
    # Remove negative or unrealistic speed
    before = len(df)
    df = df[(df['speed'] >= 0) & (df['speed'] <= 150)]
    speed_removed = before - len(df)
    print(f"  - Removed {speed_removed} rows with invalid speed (< 0 or > 150)")
    
    # Handle duplicates (same road_seg_id and same timestamp)
    before = len(df)
    df = df.groupby(['road_seg_id', 'data_time']).agg({
        'volume': 'sum',  # Sum volumes for same point at same time
        'speed': 'mean'   # Average speeds
    }).reset_index()
    dup_merged = before - len(df)
    print(f"  - Merged {dup_merged} duplicate timestamp entries")
    
    print(f"Cleaned rows: {len(df)}")
    return df

# ============================================================================
# Step 2: Distribution Learning
# ============================================================================
def learn_distributions(df):
    """Learn hourly distributions for Friday (workday) and Saturday (weekend)."""
    print("\n" + "=" * 60)
    print("Step 2: Learning Hourly Distributions")
    print("=" * 60)
    
    df['day'] = df['data_time'].dt.day
    df['hour'] = df['data_time'].dt.hour
    df['day_name'] = df['data_time'].dt.day_name()
    
    # Separate Friday and Saturday data
    friday_data = df[df['day'] == 1]  # March 1st is Friday
    saturday_data = df[df['day'] == 2]  # March 2nd is Saturday
    
    print(f"Friday (workday) records: {len(friday_data)}")
    print(f"Saturday (weekend) records: {len(saturday_data)}")
    
    # Learn hourly distributions for each road segment
    distributions = {
        'workday': {},  # Friday patterns
        'weekend': {}   # Saturday patterns
    }
    
    for road_id in df['road_seg_id'].unique():
        distributions['workday'][road_id] = {}
        distributions['weekend'][road_id] = {}
        
        for hour in range(24):
            # Friday (workday) distribution
            fri_hour_data = friday_data[
                (friday_data['road_seg_id'] == road_id) & 
                (friday_data['hour'] == hour)
            ]
            if len(fri_hour_data) > 0:
                distributions['workday'][road_id][hour] = {
                    'volume_mean': fri_hour_data['volume'].mean(),
                    'volume_std': max(fri_hour_data['volume'].std(), 1),  # Min std=1
                    'speed_mean': fri_hour_data['speed'].mean(),
                    'speed_std': max(fri_hour_data['speed'].std(), 1)
                }
            else:
                # Fallback to overall Friday average
                distributions['workday'][road_id][hour] = {
                    'volume_mean': friday_data[friday_data['road_seg_id'] == road_id]['volume'].mean(),
                    'volume_std': 5,
                    'speed_mean': friday_data[friday_data['road_seg_id'] == road_id]['speed'].mean(),
                    'speed_std': 5
                }
            
            # Saturday (weekend) distribution
            sat_hour_data = saturday_data[
                (saturday_data['road_seg_id'] == road_id) & 
                (saturday_data['hour'] == hour)
            ]
            if len(sat_hour_data) > 0:
                distributions['weekend'][road_id][hour] = {
                    'volume_mean': sat_hour_data['volume'].mean(),
                    'volume_std': max(sat_hour_data['volume'].std(), 1),
                    'speed_mean': sat_hour_data['speed'].mean(),
                    'speed_std': max(sat_hour_data['speed'].std(), 1)
                }
            else:
                distributions['weekend'][road_id][hour] = {
                    'volume_mean': saturday_data[saturday_data['road_seg_id'] == road_id]['volume'].mean(),
                    'volume_std': 5,
                    'speed_mean': saturday_data[saturday_data['road_seg_id'] == road_id]['speed'].mean(),
                    'speed_std': 5
                }
    
    # Calculate and print Friday vs Saturday comparison
    fri_total = friday_data['volume'].sum()
    sat_total = saturday_data['volume'].sum()
    print(f"\nDistribution Comparison:")
    print(f"  - Friday total volume: {fri_total:,.0f}")
    print(f"  - Saturday total volume: {sat_total:,.0f}")
    print(f"  - Saturday/Friday ratio: {sat_total/fri_total:.3f}")
    
    # Friday evening analysis
    fri_evening = friday_data[friday_data['hour'] >= FRIDAY_EVENING_START_HOUR]['volume'].sum()
    sat_evening = saturday_data[saturday_data['hour'] >= FRIDAY_EVENING_START_HOUR]['volume'].sum()
    print(f"\nEvening (17:00+) Comparison:")
    print(f"  - Friday evening volume: {fri_evening:,.0f}")
    print(f"  - Saturday evening volume: {sat_evening:,.0f}")
    print(f"  - Friday/Saturday ratio: {fri_evening/sat_evening:.2f}x (validates school effect)")
    
    return distributions, df['road_seg_id'].unique()

# ============================================================================
# Step 3: Data Augmentation
# ============================================================================
def augment_data(original_df, distributions, road_ids):
    """Generate 7 days of data using learned distributions."""
    print("\n" + "=" * 60)
    print("Step 3: Augmenting Data to 7 Days")
    print("=" * 60)
    
    # Define target dates and their types
    # March 2024: Fri(1), Sat(2), Sun(3), Mon(4), Tue(5), Wed(6), Thu(7)
    target_days = [
        (1, 'Friday', 'workday'),
        (2, 'Saturday', 'weekend'),
        (3, 'Sunday', 'weekend'),
        (4, 'Monday', 'workday'),
        (5, 'Tuesday', 'workday'),
        (6, 'Wednesday', 'workday'),
        (7, 'Thursday', 'workday'),
    ]
    
    all_data = []
    
    for day_num, day_name, day_type in target_days:
        print(f"\nGenerating Day {day_num} ({day_name}, {day_type})...")
        
        if day_num <= 2:
            # Use original data for Day 1 (Fri) and Day 2 (Sat)
            day_data = original_df[original_df['day'] == day_num].copy()
            # Apply Friday evening effect to original Friday data
            if day_num == 1:
                mask = day_data['hour'] >= FRIDAY_EVENING_START_HOUR
                day_data.loc[mask, 'volume'] = (
                    day_data.loc[mask, 'volume'] * FRIDAY_EVENING_COEFFICIENT
                ).astype(int)
            print(f"  Using original data: {len(day_data)} records")
        else:
            # Generate synthetic data for Day 3-7
            day_data = generate_synthetic_day(
                day_num, day_name, day_type, distributions, road_ids
            )
            print(f"  Generated synthetic data: {len(day_data)} records")
        
        all_data.append(day_data)
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal augmented records: {len(result)}")
    return result

def generate_synthetic_day(day_num, day_name, day_type, distributions, road_ids):
    """Generate synthetic data for a single day."""
    records = []
    base_date = datetime(2024, 3, day_num)
    
    dist = distributions[day_type]
    
    for road_id in road_ids:
        for hour in range(24):
            # Generate ~50-60 records per hour (one per minute on average)
            num_records = min(np.random.randint(50, 60), 60)
            
            for minute in np.random.choice(60, num_records, replace=False):
                # Get distribution parameters
                params = dist[road_id][hour]
                
                # Sample from distribution with noise
                volume = np.random.normal(
                    params['volume_mean'], 
                    params['volume_std']
                )
                speed = np.random.normal(
                    params['speed_mean'], 
                    params['speed_std']
                )
                
                # Apply Friday evening effect
                if day_name == 'Friday' and hour >= FRIDAY_EVENING_START_HOUR:
                    volume *= FRIDAY_EVENING_COEFFICIENT
                
                # Add additional noise
                volume *= (1 + np.random.normal(NOISE_MEAN, NOISE_STD))
                speed *= (1 + np.random.normal(NOISE_MEAN, NOISE_STD))
                
                # Ensure non-negative and realistic values
                volume = max(0, int(round(volume)))
                speed = max(0, min(150, round(speed, 3)))
                
                timestamp = base_date + timedelta(hours=hour, minutes=int(minute))
                
                records.append({
                    'road_seg_id': road_id,
                    'data_time': timestamp,
                    'volume': volume,
                    'speed': speed
                })
    
    return pd.DataFrame(records)

# ============================================================================
# Step 4: Save Results and Statistics
# ============================================================================
def save_results(df, output_path, stats_path):
    """Save processed data and generate statistics report."""
    print("\n" + "=" * 60)
    print("Step 4: Saving Results")
    print("=" * 60)
    
    # Sort by road_seg_id and time
    df = df.sort_values(['road_seg_id', 'data_time'])
    
    # Format data_time for output
    df['data_time'] = df['data_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select output columns
    output_df = df[['road_seg_id', 'data_time', 'volume', 'speed']]
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    print(f"Total records: {len(output_df)}")
    
    # Generate statistics report
    stats = []
    stats.append("=" * 60)
    stats.append("Data Preprocessing Statistics Report")
    stats.append("=" * 60)
    stats.append(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append(f"Output file: {output_path}")
    stats.append(f"Total records: {len(output_df)}")
    stats.append(f"\nDate range: 2024-03-01 to 2024-03-07 (7 days)")
    stats.append(f"Road segments: {output_df['road_seg_id'].nunique()}")
    stats.append(f"\nVolume statistics:")
    stats.append(f"  - Mean: {output_df['volume'].mean():.2f}")
    stats.append(f"  - Std: {output_df['volume'].std():.2f}")
    stats.append(f"  - Min: {output_df['volume'].min()}")
    stats.append(f"  - Max: {output_df['volume'].max()}")
    stats.append(f"\nSpeed statistics:")
    stats.append(f"  - Mean: {output_df['speed'].mean():.2f}")
    stats.append(f"  - Std: {output_df['speed'].std():.2f}")
    stats.append(f"  - Min: {output_df['speed'].min()}")
    stats.append(f"  - Max: {output_df['speed'].max()}")
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats))
    print(f"Statistics saved to: {stats_path}")

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    print("\n" + "=" * 60)
    print("Phase 2: Distribution-Aware Data Augmentation")
    print("=" * 60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Step 1: Load and clean
    df = load_and_clean_data(INPUT_FILE)
    
    # Step 2: Learn distributions
    distributions, road_ids = learn_distributions(df)
    
    # Step 3: Augment data
    augmented_df = augment_data(df, distributions, road_ids)
    
    # Step 4: Save results
    save_results(augmented_df, OUTPUT_FILE, STATS_FILE)
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
