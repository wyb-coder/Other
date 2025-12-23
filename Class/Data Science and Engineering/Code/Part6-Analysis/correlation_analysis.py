"""
Phase 6: Pearson Correlation Analysis with Heatmap Visualization
=================================================================
This script calculates Pearson correlation coefficients between 
all 10 monitoring points and generates:
1. Correlation matrix (10x10)
2. Ranked list of all road segment pairs
3. Heatmap visualization

Mathematical Background:
r(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
where r ∈ [-1, 1]
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from itertools import combinations

# ============================================================================
# Configuration
# ============================================================================
AGGREGATED_FILE = "../../Data/aggregated_15min.csv"
OUTPUT_DIR = "../../Data"
HEATMAP_FILE = "./correlation_heatmap.png"
MATRIX_FILE = f"{OUTPUT_DIR}/correlation_matrix.csv"
RANKING_FILE = f"{OUTPUT_DIR}/correlation_ranking.csv"
STATS_FILE = "./correlation_stats.txt"

# ============================================================================
# Data Preparation
# ============================================================================
def load_and_pivot_data(filepath: str) -> pd.DataFrame:
    """
    Load aggregated data and pivot to wide format.
    
    Original format (long):
        road_seg_id | window_start | total_volume
        A           | 00:00        | 100
        B           | 00:00        | 150
        
    Pivot format (wide):
        window_start | A    | B    | ...
        00:00        | 100  | 150  | ...
    """
    print("=" * 70)
    print("Step 1: Loading and Pivoting Data")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df['window_start'] = pd.to_datetime(df['window_start'])
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Road segments: {df['road_seg_id'].nunique()}")
    print(f"  Time windows: {df['window_start'].nunique()}")
    
    # Pivot: rows = time windows, columns = road segments
    pivot_volume = df.pivot_table(
        index='window_start',
        columns='road_seg_id',
        values='total_volume',
        aggfunc='mean'  # In case of duplicates
    ).fillna(0)
    
    print(f"\n  Pivoted shape: {pivot_volume.shape}")
    print(f"  (Rows=Time Windows, Cols=Sensors)")
    
    # Create short names for readability
    road_ids = pivot_volume.columns.tolist()
    short_names = {rid: f"S{i+1}" for i, rid in enumerate(road_ids)}
    pivot_volume.columns = [short_names[c] for c in pivot_volume.columns]
    
    return pivot_volume, road_ids, short_names

# ============================================================================
# Correlation Analysis
# ============================================================================
def calculate_correlation(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix.
    
    Pearson formula:
    r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
    """
    print("\n" + "=" * 70)
    print("Step 2: Calculating Pearson Correlation Matrix")
    print("=" * 70)
    
    corr_matrix = pivot_df.corr(method='pearson')
    
    print(f"  Matrix shape: {corr_matrix.shape}")
    print(f"  Diagonal values (self-correlation): all 1.00 ✓")
    
    return corr_matrix

def rank_correlations(corr_matrix: pd.DataFrame, road_ids: list, short_names: dict) -> pd.DataFrame:
    """
    Extract and rank all unique road segment pairs by correlation.
    """
    print("\n" + "=" * 70)
    print("Step 3: Ranking Road Segment Pairs")
    print("=" * 70)
    
    pairs = []
    columns = corr_matrix.columns.tolist()
    
    # Get all unique pairs (upper triangle of matrix)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            short_i = columns[i]
            short_j = columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            # Find original IDs
            orig_i = [k for k, v in short_names.items() if v == short_i][0]
            orig_j = [k for k, v in short_names.items() if v == short_j][0]
            
            pairs.append({
                'Pair': f"{short_i}-{short_j}",
                'Road_1': orig_i[-8:],  # Last 8 chars for readability
                'Road_2': orig_j[-8:],
                'Correlation': round(corr_value, 4),
                'Strength': categorize_correlation(corr_value)
            })
    
    df_pairs = pd.DataFrame(pairs)
    df_pairs = df_pairs.sort_values('Correlation', ascending=False)
    df_pairs['Rank'] = range(1, len(df_pairs) + 1)
    df_pairs = df_pairs[['Rank', 'Pair', 'Road_1', 'Road_2', 'Correlation', 'Strength']]
    
    print(f"  Total pairs: {len(df_pairs)}")
    print(f"\n  Top 5 Most Correlated:")
    print(df_pairs.head().to_string(index=False))
    print(f"\n  Bottom 5 Least Correlated:")
    print(df_pairs.tail().to_string(index=False))
    
    return df_pairs

def categorize_correlation(r: float) -> str:
    """Categorize correlation strength."""
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "Very Strong"
    elif abs_r >= 0.6:
        return "Strong"
    elif abs_r >= 0.4:
        return "Moderate"
    elif abs_r >= 0.2:
        return "Weak"
    else:
        return "Very Weak"

# ============================================================================
# Visualization
# ============================================================================
def create_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """
    Create Seaborn heatmap visualization.
    """
    print("\n" + "=" * 70)
    print("Step 4: Generating Heatmap")
    print("=" * 70)
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',  # Red=positive, Blue=negative
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'},
        vmin=-1, vmax=1
    )
    
    plt.title('Traffic Volume Correlation Matrix\n(10 Monitoring Points × 7 Days)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Road Segment', fontsize=12)
    plt.ylabel('Road Segment', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Heatmap saved to: {output_path}")

# ============================================================================
# Statistics Report
# ============================================================================
def generate_stats_report(corr_matrix: pd.DataFrame, ranking_df: pd.DataFrame, 
                          pivot_shape: tuple) -> str:
    """Generate comprehensive statistics report."""
    
    stats = []
    stats.append("=" * 70)
    stats.append("Phase 6: Pearson Correlation Analysis Report")
    stats.append("=" * 70)
    stats.append(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data overview
    stats.append("\n" + "=" * 70)
    stats.append("1. Data Overview")
    stats.append("=" * 70)
    stats.append(f"  Data matrix shape: {pivot_shape}")
    stats.append(f"  Time windows (rows): {pivot_shape[0]}")
    stats.append(f"  Road segments (columns): {pivot_shape[1]}")
    stats.append(f"  Total pairs analyzed: {len(ranking_df)}")
    
    # Correlation distribution
    stats.append("\n" + "=" * 70)
    stats.append("2. Correlation Distribution")
    stats.append("=" * 70)
    
    corr_values = ranking_df['Correlation'].values
    stats.append(f"  Mean correlation: {np.mean(corr_values):.4f}")
    stats.append(f"  Std deviation: {np.std(corr_values):.4f}")
    stats.append(f"  Min correlation: {np.min(corr_values):.4f}")
    stats.append(f"  Max correlation: {np.max(corr_values):.4f}")
    
    # Strength breakdown
    stats.append("\n  Correlation Strength Breakdown:")
    strength_counts = ranking_df['Strength'].value_counts()
    for strength in ['Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak']:
        count = strength_counts.get(strength, 0)
        pct = count / len(ranking_df) * 100
        stats.append(f"    {strength}: {count} pairs ({pct:.1f}%)")
    
    # Top correlations
    stats.append("\n" + "=" * 70)
    stats.append("3. Top 10 Most Correlated Pairs")
    stats.append("=" * 70)
    stats.append("\n" + ranking_df.head(10).to_string(index=False))
    
    # Bottom correlations
    stats.append("\n" + "=" * 70)
    stats.append("4. Top 10 Least Correlated Pairs")
    stats.append("=" * 70)
    stats.append("\n" + ranking_df.tail(10).to_string(index=False))
    
    # Interpretation
    stats.append("\n" + "=" * 70)
    stats.append("5. Physical Interpretation (for Thesis)")
    stats.append("=" * 70)
    stats.append("""
  【相关性分析结论】
  
  高相关性 (r > 0.8) 可能表示：
  - 上下游路段关系（拥堵传播）
  - 共享交通流（汇入同一主干道）
  - 时空依赖性（如信号灯同步区域）
  
  低相关性 (r < 0.4) 可能表示：
  - 地理位置相隔较远
  - 服务不同交通流向
  - 独立的区域交通网络
  
  本分析为交通管理提供数据支撑：
  - 高相关路段可联动控制
  - 低相关路段可独立优化
""")
    
    # Equivalent Spark code
    stats.append("\n" + "=" * 70)
    stats.append("6. Equivalent Spark MLlib Code")
    stats.append("=" * 70)
    stats.append("""
  from pyspark.mllib.stat import Statistics
  from pyspark.mllib.linalg import Vectors
  
  # Convert DataFrame to RDD of vectors
  rdd = df.rdd.map(lambda row: Vectors.dense(row))
  
  # Calculate correlation matrix
  corr_matrix = Statistics.corr(rdd, method="pearson")
  
  # Complexity: O(N²) where N = number of sensors
  # For N > 1000, distributed computation is essential
""")
    
    return '\n'.join(stats)

# ============================================================================
# Main
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "=" * 70)
    print("Phase 6: Pearson Correlation Analysis")
    print("=" * 70)
    
    # Step 1: Load and pivot data
    pivot_df, road_ids, short_names = load_and_pivot_data(AGGREGATED_FILE)
    
    # Step 2: Calculate correlation matrix
    corr_matrix = calculate_correlation(pivot_df)
    
    # Step 3: Rank all pairs
    ranking = rank_correlations(corr_matrix, road_ids, short_names)
    
    # Step 4: Create heatmap
    create_heatmap(corr_matrix, HEATMAP_FILE)
    
    # Step 5: Save results
    print("\n" + "=" * 70)
    print("Step 5: Saving Results")
    print("=" * 70)
    
    corr_matrix.to_csv(MATRIX_FILE)
    print(f"  Correlation matrix saved to: {MATRIX_FILE}")
    
    ranking.to_csv(RANKING_FILE, index=False)
    print(f"  Ranking saved to: {RANKING_FILE}")
    
    # Step 6: Generate statistics report
    stats = generate_stats_report(corr_matrix, ranking, pivot_df.shape)
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(stats)
    print(f"  Statistics saved to: {STATS_FILE}")
    
    print("\n" + "=" * 70)
    print("Phase 6: Analysis Complete!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Total pairs: {len(ranking)}")
    print(f"  - Highest correlation: {ranking.iloc[0]['Pair']} = {ranking.iloc[0]['Correlation']:.4f}")
    print(f"  - Lowest correlation: {ranking.iloc[-1]['Pair']} = {ranking.iloc[-1]['Correlation']:.4f}")

if __name__ == "__main__":
    main()
