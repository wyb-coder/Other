"""
图 2-2: 原始数据与合成数据的分布对比直方图

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 数据加载
# ============================================================================
def load_data():
    """加载原始数据和处理后的数据"""
    # 原始数据（82.csv 是未处理的原始数据）
    original_df = pd.read_csv('../Data/82.csv')
    original_df['data_time'] = pd.to_datetime(original_df['data_time'])
    
    # 处理后的完整7天数据
    processed_df = pd.read_csv('../Data/82_processed.csv')
    processed_df['data_time'] = pd.to_datetime(processed_df['data_time'])
    
    return original_df, processed_df

# ============================================================================
# 分离原始数据和合成数据
# ============================================================================
def separate_data(original_df, processed_df):
    """
    分离原始数据和合成数据
    - 原始数据：周五(3月1日) 和 周六(3月2日)
    - 合成数据：周日(3月3日) 到 周四(3月7日)
    """
    # 原始数据：3月1日和3月2日
    original_days = processed_df[
        (processed_df['data_time'].dt.day == 1) | 
        (processed_df['data_time'].dt.day == 2)
    ]
    
    # 合成数据：3月3日到3月7日
    synthetic_days = processed_df[
        processed_df['data_time'].dt.day >= 3
    ]
    
    return original_days, synthetic_days

# ============================================================================
# 绘制对比直方图
# ============================================================================
def plot_distribution_comparison(original, synthetic, column, title, xlabel, output_file):
    """绘制原始数据与合成数据的分布对比直方图"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算合适的 bins
    all_data = pd.concat([original[column], synthetic[column]])
    bins = np.linspace(all_data.min(), all_data.quantile(0.99), 40)
    
    # 绘制原始数据直方图（蓝色）
    ax.hist(original[column], bins=bins, alpha=0.6, color='#3498db', 
            label=f'原始数据 (周五/周六)\nn={len(original):,}', 
            density=True, edgecolor='white', linewidth=0.5)
    
    # 绘制合成数据直方图（橙色）
    ax.hist(synthetic[column], bins=bins, alpha=0.6, color='#e74c3c', 
            label=f'合成数据 (周日~周四)\nn={len(synthetic):,}', 
            density=True, edgecolor='white', linewidth=0.5)
    
    # 添加统计信息文本框
    stats_text = (
        f"原始数据: μ={original[column].mean():.2f}, σ={original[column].std():.2f}\n"
        f"合成数据: μ={synthetic[column].mean():.2f}, σ={synthetic[column].std():.2f}\n"
        f"均值偏差: {abs(original[column].mean() - synthetic[column].mean()) / original[column].mean() * 100:.2f}%"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('密度 (Density)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存: {output_file}")
    return output_file

# ============================================================================
# 绘制双图对比（流量 + 速度）
# ============================================================================
def plot_combined_comparison(original, synthetic, output_file):
    """绘制流量和速度的双图对比"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 流量分布对比
    ax1 = axes[0]
    vol_bins = np.linspace(0, original['volume'].quantile(0.99), 40)
    
    ax1.hist(original['volume'], bins=vol_bins, alpha=0.6, color='#3498db', 
             label=f'原始数据 (n={len(original):,})', density=True, 
             edgecolor='white', linewidth=0.5)
    ax1.hist(synthetic['volume'], bins=vol_bins, alpha=0.6, color='#e74c3c', 
             label=f'合成数据 (n={len(synthetic):,})', density=True,
             edgecolor='white', linewidth=0.5)
    
    vol_bias = abs(original['volume'].mean() - synthetic['volume'].mean()) / original['volume'].mean() * 100
    ax1.text(0.97, 0.97, f'均值偏差: {vol_bias:.2f}%', transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax1.set_xlabel('流量 (Volume)', fontsize=12)
    ax1.set_ylabel('密度 (Density)', fontsize=12)
    ax1.set_title('(a) 流量分布对比', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 速度分布对比
    ax2 = axes[1]
    spd_bins = np.linspace(0, original['speed'].quantile(0.99), 40)
    
    ax2.hist(original['speed'], bins=spd_bins, alpha=0.6, color='#3498db', 
             label=f'原始数据 (n={len(original):,})', density=True,
             edgecolor='white', linewidth=0.5)
    ax2.hist(synthetic['speed'], bins=spd_bins, alpha=0.6, color='#e74c3c', 
             label=f'合成数据 (n={len(synthetic):,})', density=True,
             edgecolor='white', linewidth=0.5)
    
    spd_bias = abs(original['speed'].mean() - synthetic['speed'].mean()) / original['speed'].mean() * 100
    ax2.text(0.97, 0.97, f'均值偏差: {spd_bias:.2f}%', transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlabel('速度 (Speed, km/h)', fontsize=12)
    ax2.set_ylabel('密度 (Density)', fontsize=12)
    ax2.set_title('(b) 速度分布对比', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('图 2-2 原始数据与合成数据的分布一致性验证', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"双图对比已保存: {output_file}")
    return output_file

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("生成图 2-2: 原始数据与合成数据分布对比")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载数据...")
    original_df, processed_df = load_data()
    print(f"   原始CSV: {len(original_df):,} 行")
    print(f"   处理后CSV: {len(processed_df):,} 行")
    
    # 分离原始和合成数据
    print("\n2. 分离原始数据与合成数据...")
    original_days, synthetic_days = separate_data(original_df, processed_df)
    print(f"   原始数据 (周五/周六): {len(original_days):,} 行")
    print(f"   合成数据 (周日~周四): {len(synthetic_days):,} 行")
    
    # 生成统计对比
    print("\n3. 统计对比:")
    print(f"   流量均值 - 原始: {original_days['volume'].mean():.2f}, 合成: {synthetic_days['volume'].mean():.2f}")
    print(f"   速度均值 - 原始: {original_days['speed'].mean():.2f}, 合成: {synthetic_days['speed'].mean():.2f}")
    
    # 绘制单独的直方图
    print("\n4. 生成直方图...")
    
    # 流量分布对比
    plot_distribution_comparison(
        original_days, synthetic_days, 'volume',
        '原始数据与合成数据的流量分布对比',
        '流量 (Volume)',
        './fig2_2_volume_comparison.png'
    )
    
    # 速度分布对比
    plot_distribution_comparison(
        original_days, synthetic_days, 'speed',
        '原始数据与合成数据的速度分布对比',
        '速度 (Speed, km/h)',
        './fig2_2_speed_comparison.png'
    )
    
    # 绘制组合双图
    plot_combined_comparison(
        original_days, synthetic_days,
        './fig2_2_combined_comparison.png'
    )
    
    print("\n" + "=" * 60)
    print("所有图片生成完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
