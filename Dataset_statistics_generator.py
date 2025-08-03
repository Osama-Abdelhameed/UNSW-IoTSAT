#!/usr/bin/env python3
"""
Satellite Attack Dataset Statistics Analyzer

"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SatelliteAttackStatistics:
    def __init__(self, csv_file: str, output_dir: str = "statistics"):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f" Loading data from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        print(f" Loaded {len(self.df):,} records")
        
        # Professional color palettes for publication
        self.colors = {
            'normal': '#2E86AB',  # Blue
            'attack': '#A23B72',  # Red/Purple
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#F4E285'],
            'diverging': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'],
            'sequential': ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
        }
        
        # Set publication-quality parameters
        self._set_publication_style()
    
    def _set_publication_style(self):
        """Set matplotlib parameters for publication-quality figures"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'lines.linewidth': 2.5,
            'patch.edgecolor': 'black',
            'patch.linewidth': 1
        })
    
    def save_publication_figure(self, fig, filename):
        """Save figure with publication-quality settings"""
        fig.tight_layout()
        
        # Save as PNG
        fig.savefig(self.output_dir / filename, 
                    dpi=300, 
                    bbox_inches='tight', 
                    facecolor='white',
                    edgecolor='none',
                    format='png',
                    pad_inches=0.2)
        
        # Also save as PDF for publications
        fig.savefig(self.output_dir / filename.replace('.png', '.pdf'), 
                    dpi=300, 
                    bbox_inches='tight',
                    format='pdf',
                    pad_inches=0.2)
        
        plt.close(fig)
    
    def generate_all_statistics(self):
        """Generate all statistics and visualizations"""
        print("\n Generating Comprehensive Statistics...")
        print("=" * 60)
        
        # 1. Basic Dataset Overview
        self.basic_overview()
        
        # 2. Attack Type Distribution
        self.attack_type_distribution()
        
        # 3. Attack Subtype Analysis
        self.attack_subtype_analysis()
        
        # 4. Severity Analysis
        self.severity_analysis()
        
        # 5. RF Metrics Analysis
        self.rf_metrics_analysis()
        
        # 6. Detection Confidence Analysis
        self.detection_confidence_analysis()
        
        # 7. Temporal Analysis
        self.temporal_analysis()
        
        # 8. Satellite-specific Analysis
        self.satellite_analysis()
        
        # 9. Attack Duration Analysis
        self.attack_duration_analysis()
        
        # 10. Quality Score Analysis
        self.quality_score_analysis()
        
        # 11. Generate Summary Report
        self.generate_summary_report()
        
        print("\n All statistics generated successfully!")
        print(f" Results saved in: {self.output_dir}")
    
    def basic_overview(self):
        """Generate basic dataset overview"""
        print("\n BASIC DATASET OVERVIEW")
        print("-" * 60)
        
        total_records = len(self.df)
        attack_records = len(self.df[self.df['Attack_Flag'] == 1])
        normal_records = len(self.df[self.df['Attack_Flag'] == 0])
        attack_ratio = attack_records / total_records * 100
        
        overview = {
            "Total Records": f"{total_records:,}",
            "Attack Records": f"{attack_records:,} ({attack_ratio:.1f}%)",
            "Normal Records": f"{normal_records:,} ({100-attack_ratio:.1f}%)",
            "Unique Satellites": self.df['Satellite_ID'].nunique(),
            "Time Range": f"{self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}",
            "Total Columns": len(self.df.columns)
        }
        
        for key, value in overview.items():
            print(f"{key:.<30} {value}")
        
        # Save to file
        with open(self.output_dir / "basic_overview.json", 'w') as f:
            json.dump(overview, f, indent=2)
    
    def attack_type_distribution(self):
        """Analyze attack type distribution with publication-quality visualizations"""
        print("\n ATTACK TYPE DISTRIBUTION")
        print("-" * 60)
        
        # Count by attack type
        attack_counts = self.df['Attack_Type'].value_counts()
        
        print("Attack Type Counts:")
        for attack_type, count in attack_counts.items():
            percentage = count / len(self.df) * 100
            print(f"{attack_type:.<20} {count:>8,} ({percentage:>5.1f}%)")
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Bar chart
        ax1 = plt.subplot(121)
        colors = self.colors['primary'][:len(attack_counts)]
        attack_counts.plot(kind='bar', ax=ax1, color=colors, 
                          edgecolor='black', linewidth=1.5)
        ax1.set_title('Attack Type Distribution', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Attack Type', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add value labels on bars
        for i, (idx, v) in enumerate(attack_counts.items()):
            ax1.text(i, v + 500, f'{v:,}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # Pie chart with better spacing
        ax2 = plt.subplot(122)
        
        # Explode small slices more than large ones
        explode = []
        for count in attack_counts.values:
            if count / attack_counts.sum() < 0.05:  # If less than 5%
                explode.append(0.15)
            elif count / attack_counts.sum() < 0.10:  # If less than 10%
                explode.append(0.10)
            else:
                explode.append(0.02)
        
        wedges, texts, autotexts = ax2.pie(attack_counts.values, 
                                           labels=attack_counts.index,
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           startangle=45,
                                           pctdistance=0.85,
                                           explode=explode,
                                           shadow=True)
        
        # Enhance text readability
        for text in texts:
            text.set_fontsize(13)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        ax2.set_title('Attack Type Distribution (%)', fontsize=18, fontweight='bold', pad=20)
        
        # Add total count in center
        centre_circle = plt.Circle((0,0), 0.70, fc='white', linewidth=2, edgecolor='black')
        ax2.add_artist(centre_circle)
        ax2.text(0, 0, f'Total\n{len(self.df):,}', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Save with high quality
        self.save_publication_figure(fig, 'attack_type_distribution.png')
        
        # Save data
        attack_counts.to_csv(self.output_dir / 'attack_type_counts.csv')
    
    def attack_subtype_analysis(self):
        """Analyze attack subtypes with improved visualization"""
        print("\n ATTACK SUBTYPE ANALYSIS")
        print("-" * 60)
        
        # Get attack records only
        attack_df = self.df[self.df['Attack_Flag'] == 1].copy()
        
        # Count by attack type and subtype
        subtype_counts = attack_df.groupby(['Attack_Type', 'Attack_Subtype']).size().reset_index(name='Count')
        
        # Create stacked bar chart for better readability
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Pivot data for stacked bar chart
        pivot_data = subtype_counts.pivot(index='Attack_Type', columns='Attack_Subtype', values='Count').fillna(0)
        
        # Create stacked bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_data.columns)))
        pivot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, 
                       edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Attack Type', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Attack Subtype Distribution by Attack Type', fontsize=18, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fmt='%g', fontsize=10, fontweight='bold')
        
        # Improve legend
        ax.legend(title='Attack Subtype', bbox_to_anchor=(1.05, 1), loc='upper left',
                 frameon=True, fancybox=True, shadow=True, title_fontsize=12)
        
        self.save_publication_figure(fig, 'attack_subtype_distribution.png')
        
        # Print summary
        print("\nSubtype Distribution by Attack Type:")
        for attack_type in subtype_counts['Attack_Type'].unique():
            print(f"\n{attack_type.upper()}:")
            subtypes = subtype_counts[subtype_counts['Attack_Type'] == attack_type]
            for _, row in subtypes.iterrows():
                print(f"  {row['Attack_Subtype']:.<25} {row['Count']:>6,}")
        
        # Save data
        subtype_counts.to_csv(self.output_dir / 'attack_subtype_counts.csv', index=False)
    
    def severity_analysis(self):
        """Analyze attack severity distribution with improved visualizations"""
        print("\n  ATTACK SEVERITY ANALYSIS")
        print("-" * 60)
        
        # Get attack records only
        attack_df = self.df[self.df['Attack_Flag'] == 1].copy()
        
        # Calculate severity statistics by attack type
        severity_stats = attack_df.groupby('Attack_Type')['Attack_Severity'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        print("Severity Statistics by Attack Type:")
        print(severity_stats)
        
        # Create visualizations
        fig = plt.figure(figsize=(18, 12))
        
        # Box plot by attack type
        ax1 = plt.subplot(211)
        attack_types = attack_df['Attack_Type'].unique()
        severity_data = [attack_df[attack_df['Attack_Type'] == at]['Attack_Severity'].values 
                        for at in attack_types]
        
        bp = ax1.boxplot(severity_data, labels=attack_types, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Customize box plots
        for i, (patch, color) in enumerate(zip(bp['boxes'], self.colors['primary'])):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        plt.setp(bp['means'], color='red', linewidth=2)
        
        ax1.set_title('Attack Severity Distribution by Type', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Attack Type', fontsize=14)
        ax1.set_ylabel('Severity', fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Histogram of overall severity
        ax2 = plt.subplot(212)
        n, bins, patches = ax2.hist(attack_df['Attack_Severity'], bins=50, 
                                   edgecolor='black', linewidth=1)
        
        # Color gradient for histogram
        cm = plt.cm.get_cmap('YlOrRd')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax2.set_title('Overall Attack Severity Distribution', fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlabel('Severity', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)
        
        # Add statistical lines
        mean_val = attack_df['Attack_Severity'].mean()
        median_val = attack_df['Attack_Severity'].median()
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_val:.3f}')
        ax2.axvline(median_val, color='blue', linestyle='--', linewidth=2.5,
                   label=f'Median: {median_val:.3f}')
        
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        self.save_publication_figure(fig, 'attack_severity_analysis.png')
        
        # Save statistics
        severity_stats.to_csv(self.output_dir / 'severity_statistics.csv')
    
    def rf_metrics_analysis(self):
        """Analyze RF metrics with publication-quality visualizations"""
        print("\n RF METRICS ANALYSIS")
        print("-" * 60)
        
        rf_columns = ['RF_Signal_Strength_dBm', 'RF_SNR_dB', 'RF_Bit_Error_Rate', 
                     'RF_Packet_Error_Rate', 'RF_Throughput_bps', 'RF_Constellation_Error']
        
        # Calculate statistics for normal vs attack
        normal_df = self.df[self.df['Attack_Flag'] == 0]
        attack_df = self.df[self.df['Attack_Flag'] == 1]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.ravel()
        
        for i, col in enumerate(rf_columns):
            ax = axes[i]
            
            # Get data
            normal_data = normal_df[col].dropna()
            attack_data = attack_df[col].dropna()
            
            # Determine if we need log scale
            use_log = col in ['RF_Bit_Error_Rate', 'RF_Packet_Error_Rate']
            
            if len(normal_data) > 0 and len(attack_data) > 0:
                # Create density plots
                if not use_log or (use_log and normal_data.min() > 0 and attack_data.min() > 0):
                    # Use KDE for smooth curves
                    density_normal = stats.gaussian_kde(normal_data)
                    density_attack = stats.gaussian_kde(attack_data)
                    
                    # Create smooth x-axis
                    x_min = min(normal_data.min(), attack_data.min())
                    x_max = max(normal_data.max(), attack_data.max())
                    x = np.linspace(x_min, x_max, 200)
                    
                    # Plot densities
                    ax.plot(x, density_normal(x), color=self.colors['normal'], 
                           linewidth=3, label='Normal', linestyle='-')
                    ax.fill_between(x, density_normal(x), alpha=0.3, 
                                   color=self.colors['normal'])
                    
                    ax.plot(x, density_attack(x), color=self.colors['attack'], 
                           linewidth=3, label='Attack', linestyle='-')
                    ax.fill_between(x, density_attack(x), alpha=0.3, 
                                   color=self.colors['attack'])
                else:
                    # Use histograms for log scale data
                    ax.hist(normal_data[normal_data > 0], bins=50, alpha=0.5, 
                           label='Normal', color=self.colors['normal'], 
                           density=True, edgecolor='black')
                    ax.hist(attack_data[attack_data > 0], bins=50, alpha=0.5, 
                           label='Attack', color=self.colors['attack'], 
                           density=True, edgecolor='black')
                    if use_log:
                        ax.set_xscale('log')
            
            # Formatting
            ax.set_title(col.replace('RF_', '').replace('_', ' '), fontsize=14, fontweight='bold')
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add mean lines
            if len(normal_data) > 0:
                ax.axvline(normal_data.mean(), color=self.colors['normal'], 
                          linestyle=':', linewidth=2, alpha=0.8)
            if len(attack_data) > 0:
                ax.axvline(attack_data.mean(), color=self.colors['attack'], 
                          linestyle=':', linewidth=2, alpha=0.8)
        
        plt.suptitle('RF Metrics Distribution: Normal vs Attack', fontsize=20, fontweight='bold')
        
        self.save_publication_figure(fig, 'rf_metrics_comparison.png')
        
        # Calculate and save statistics
        rf_stats = pd.DataFrame()
        for col in rf_columns:
            stats_dict = {
                'Metric': col,
                'Normal_Mean': normal_df[col].mean(),
                'Normal_Std': normal_df[col].std(),
                'Attack_Mean': attack_df[col].mean(),
                'Attack_Std': attack_df[col].std(),
                'Difference': attack_df[col].mean() - normal_df[col].mean(),
                'Percent_Change': ((attack_df[col].mean() - normal_df[col].mean()) / normal_df[col].mean() * 100)
            }
            rf_stats = pd.concat([rf_stats, pd.DataFrame([stats_dict])], ignore_index=True)
        
        print("RF Metrics Comparison (Normal vs Attack):")
        for _, row in rf_stats.iterrows():
            print(f"\n{row['Metric']}:")
            print(f"  Normal: {row['Normal_Mean']:.4f} ± {row['Normal_Std']:.4f}")
            print(f"  Attack: {row['Attack_Mean']:.4f} ± {row['Attack_Std']:.4f}")
            print(f"  Change: {row['Difference']:.4f} ({row['Percent_Change']:.1f}%)")
        
        rf_stats.to_csv(self.output_dir / 'rf_metrics_statistics.csv', index=False)
    
    def detection_confidence_analysis(self):
        """Analyze detection confidence with improved violin plots"""
        print("\n DETECTION CONFIDENCE ANALYSIS")
        print("-" * 60)
        
        # Get attack records only
        attack_df = self.df[self.df['Attack_Flag'] == 1].copy()
        
        # Calculate detection confidence statistics
        detection_stats = attack_df.groupby('Attack_Type')['Detection_Confidence'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        
        print("Detection Confidence by Attack Type:")
        print(detection_stats)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create violin plot with better styling
        attack_types = sorted(attack_df['Attack_Type'].unique())
        data_to_plot = [attack_df[attack_df['Attack_Type'] == at]['Detection_Confidence'].values 
                       for at in attack_types]
        
        parts = ax.violinplot(data_to_plot, positions=range(len(attack_types)), 
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Customize violin plot colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors['primary'][i % len(self.colors['primary'])])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Customize other elements
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.set_xlabel('Attack Type', fontsize=14)
        ax.set_ylabel('Detection Confidence', fontsize=14)
        ax.set_title('Detection Confidence Distribution by Attack Type', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add mean values as text
        for i, attack_type in enumerate(attack_types):
            mean_val = attack_df[attack_df['Attack_Type'] == attack_type]['Detection_Confidence'].mean()
            ax.text(i, 1.05, f'{mean_val:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        
        self.save_publication_figure(fig, 'detection_confidence_analysis.png')
        
        # Save statistics
        detection_stats.to_csv(self.output_dir / 'detection_confidence_statistics.csv')
    
    def temporal_analysis(self):
        """Analyze temporal patterns with improved visualizations"""
        print("\n TEMPORAL ANALYSIS")
        print("-" * 60)
        
        # Convert timestamp to datetime
        self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'])
        
        # Extract time components
        self.df['Hour'] = self.df['Datetime'].dt.hour
        self.df['DayOfWeek'] = self.df['Datetime'].dt.day_name()
        
        # Analyze attacks by hour
        hourly_attacks = self.df[self.df['Attack_Flag'] == 1].groupby('Hour').size()
        
        # Create visualization
        fig = plt.figure(figsize=(18, 14))
        
        # Hourly attack count with better styling
        ax1 = plt.subplot(211)
        bars = ax1.bar(hourly_attacks.index, hourly_attacks.values, 
                      color=self.colors['attack'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Color gradient based on attack count
        norm = plt.Normalize(hourly_attacks.values.min(), hourly_attacks.values.max())
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
        sm.set_array([])
        
        for bar, height in zip(bars, hourly_attacks.values):
            bar.set_facecolor(sm.to_rgba(height))
        
        ax1.set_xlabel('Hour of Day', fontsize=14)
        ax1.set_ylabel('Number of Attacks', fontsize=14)
        ax1.set_title('Attack Distribution by Hour of Day', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xticks(range(24))
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add value labels on bars
        for i, v in enumerate(hourly_attacks.values):
            ax1.text(hourly_attacks.index[i], v + 50, f'{v:,}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Attack types over time
        ax2 = plt.subplot(212)
        attack_df = self.df[self.df['Attack_Flag'] == 1].copy()
        attack_timeline = attack_df.groupby(['Datetime', 'Attack_Type']).size().unstack(fill_value=0)
        
        # Resample to hourly for better visualization
        attack_timeline_hourly = attack_timeline.resample('1H').sum()
        
        # Plot stacked area chart with better colors
        attack_timeline_hourly.plot(kind='area', ax=ax2, alpha=0.8, 
                                   color=self.colors['primary'][:len(attack_timeline_hourly.columns)],
                                   linewidth=2)
        
        ax2.set_xlabel('Time', fontsize=14)
        ax2.set_ylabel('Number of Attacks', fontsize=14)
        ax2.set_title('Attack Types Over Time', fontsize=18, fontweight='bold', pad=20)
        ax2.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left',
                  frameon=True, fancybox=True, shadow=True, title_fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        self.save_publication_figure(fig, 'temporal_analysis.png')
        
        print(f"Average attacks per hour: {hourly_attacks.mean():.1f}")
        print(f"Peak attack hour: {hourly_attacks.idxmax()}:00 ({hourly_attacks.max()} attacks)")
        print(f"Lowest attack hour: {hourly_attacks.idxmin()}:00 ({hourly_attacks.min()} attacks)")
    
    def satellite_analysis(self):
        """Analyze attacks by satellite with improved visualizations"""
        print("\n  SATELLITE-SPECIFIC ANALYSIS")
        print("-" * 60)
        
        # Count attacks by satellite
        satellite_stats = self.df.groupby('Satellite_ID').agg({
            'Attack_Flag': ['count', 'sum', 'mean']
        })
        satellite_stats.columns = ['Total_Records', 'Attack_Count', 'Attack_Ratio']
        satellite_stats['Normal_Count'] = satellite_stats['Total_Records'] - satellite_stats['Attack_Count']
        
        print("Attack Statistics by Satellite:")
        for sat_id, row in satellite_stats.iterrows():
            print(f"\nSatellite {sat_id}:")
            print(f"  Total Records: {row['Total_Records']:,}")
            print(f"  Attacks: {row['Attack_Count']:,} ({row['Attack_Ratio']*100:.1f}%)")
            print(f"  Normal: {row['Normal_Count']:,} ({(1-row['Attack_Ratio'])*100:.1f}%)")
        
        # Attack types by satellite
        attack_by_satellite = self.df[self.df['Attack_Flag'] == 1]\
            .groupby(['Satellite_ID', 'Attack_Type'])\
            .size().unstack(fill_value=0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Bar chart
        satellite_data = satellite_stats[['Attack_Count', 'Normal_Count']]
        satellite_data.plot(kind='bar', stacked=True, ax=ax1,
                            color=[self.colors['attack'], self.colors['normal']],
                            alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Satellite ID', fontsize=14)
        ax1.set_ylabel('Number of Records', fontsize=14)
        ax1.set_title('Attack vs Normal Records by Satellite', fontsize=18, fontweight='bold', pad=20)
        ax1.legend(['Attacks', 'Normal'], frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        for container in ax1.containers:
            ax1.bar_label(container, label_type='center', fmt='%g', fontsize=11, fontweight='bold')
        
        # Heatmap
        cmap = plt.cm.YlOrRd
        im = ax2.imshow(attack_by_satellite.T.values, aspect='auto', cmap=cmap)
        ax2.set_xticks(np.arange(len(attack_by_satellite.index)))
        ax2.set_yticks(np.arange(len(attack_by_satellite.columns)))
        ax2.set_xticklabels(attack_by_satellite.index, fontsize=12)
        ax2.set_yticklabels(attack_by_satellite.columns, fontsize=12)
        for i in range(len(attack_by_satellite.columns)):
            for j in range(len(attack_by_satellite.index)):
                value = attack_by_satellite.iloc[j, i]
                if value > 0:
                    ax2.text(j, i, f'{int(value):,}',
                             ha='center', va='center',
                             color='white' if value > attack_by_satellite.values.max()/2 else 'black',
                             fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('Satellite ID', fontsize=14)
        ax2.set_ylabel('Attack Type', fontsize=14)
        ax2.set_title('Attack Type Distribution by Satellite', fontsize=18, fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel('Number of Attacks', rotation=270, labelpad=20, fontsize=12)
        
        self.save_publication_figure(fig, 'satellite_analysis.png')
        satellite_stats.to_csv(self.output_dir / 'satellite_statistics.csv')
        attack_by_satellite.to_csv(self.output_dir / 'attack_types_by_satellite.csv')
    
    def attack_duration_analysis(self):
        """Analyze attack duration patterns with improved visualizations"""
        print("\n  ATTACK DURATION ANALYSIS")
        print("-" * 60)
        
        # Get attack records only
        attack_df = self.df[self.df['Attack_Flag'] == 1].copy()
        
        # Duration statistics by attack type
        duration_stats = attack_df.groupby('Attack_Type')['Attack_Duration'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(1)
        
        print("Attack Duration Statistics (seconds):")
        print(duration_stats)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Box plot of duration by attack type
        attack_types = sorted(attack_df['Attack_Type'].unique())
        duration_data = [attack_df[attack_df['Attack_Type'] == at]['Attack_Duration'].values 
                        for at in attack_types]
        
        bp = ax1.boxplot(duration_data, labels=attack_types, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Customize box plots
        for i, (patch, color) in enumerate(zip(bp['boxes'], self.colors['primary'])):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        plt.setp(bp['means'], color='red', linewidth=2)
        
        ax1.set_xlabel('Attack Type', fontsize=14)
        ax1.set_ylabel('Duration (seconds)', fontsize=14)
        ax1.set_title('Attack Duration Distribution by Type', fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.set_yscale('log')  # Log scale for better visualization
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--', which='both')
        ax1.set_axisbelow(True)
        
        # Add median values as text
        for i, attack_type in enumerate(attack_types):
            median_val = attack_df[attack_df['Attack_Type'] == attack_type]['Attack_Duration'].median()
            ax1.text(i+1, median_val*1.2, f'{median_val:.1f}s', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Scatter plot of duration vs severity
        # Create scatter plot with better visibility
        attack_type_codes = pd.Categorical(attack_df['Attack_Type']).codes
        scatter = ax2.scatter(attack_df['Attack_Duration'], 
                            attack_df['Attack_Severity'], 
                            c=attack_type_codes, 
                            alpha=0.6, 
                            cmap='tab10',
                            s=50,
                            edgecolors='black',
                            linewidth=0.5)
        
        ax2.set_xlabel('Duration (seconds)', fontsize=14)
        ax2.set_ylabel('Severity', fontsize=14)
        ax2.set_title('Attack Duration vs Severity', fontsize=18, fontweight='bold', pad=20)
        ax2.set_xscale('log')  # Log scale for duration
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Add colorbar with attack type labels
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Attack Type', rotation=270, labelpad=20, fontsize=12)
        cbar.set_ticks(range(len(attack_types)))
        cbar.set_ticklabels(attack_types)
        
        self.save_publication_figure(fig, 'attack_duration_analysis.png')
        
        # Save statistics
        duration_stats.to_csv(self.output_dir / 'duration_statistics.csv')
    
    def quality_score_analysis(self):
        """Analyze quality scores with improved visualizations"""
        print("\n QUALITY SCORE ANALYSIS")
        print("-" * 60)
        
        # Quality score statistics
        quality_stats = self.df.groupby('Attack_Flag')['Quality_Score'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        quality_stats.index = ['Normal', 'Attack']
        
        print("Quality Score Statistics:")
        print(quality_stats)
        
        # Quality score by attack type
        attack_quality = self.df[self.df['Attack_Flag'] == 1].groupby('Attack_Type')['Quality_Score'].agg(['mean', 'std']).round(3)
        
        print("\nQuality Score by Attack Type:")
        print(attack_quality)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Distribution comparison with KDE
        normal_scores = self.df[self.df['Attack_Flag'] == 0]['Quality_Score'].dropna()
        attack_scores = self.df[self.df['Attack_Flag'] == 1]['Quality_Score'].dropna()
        
        # Create smooth density plots
        if len(normal_scores) > 0:
            density_normal = stats.gaussian_kde(normal_scores)
            x_normal = np.linspace(normal_scores.min(), normal_scores.max(), 200)
            ax1.plot(x_normal, density_normal(x_normal), 
                    color=self.colors['normal'], linewidth=3, label='Normal')
            ax1.fill_between(x_normal, density_normal(x_normal), 
                           alpha=0.3, color=self.colors['normal'])
        
        if len(attack_scores) > 0:
            density_attack = stats.gaussian_kde(attack_scores)
            x_attack = np.linspace(attack_scores.min(), attack_scores.max(), 200)
            ax1.plot(x_attack, density_attack(x_attack), 
                    color=self.colors['attack'], linewidth=3, label='Attack')
            ax1.fill_between(x_attack, density_attack(x_attack), 
                           alpha=0.3, color=self.colors['attack'])
        
        ax1.set_xlabel('Quality Score', fontsize=14)
        ax1.set_ylabel('Density', fontsize=14)
        ax1.set_title('Quality Score Distribution: Normal vs Attack', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        ax1.set_xlim(0, 1.05)
        
        # Add mean lines
        ax1.axvline(normal_scores.mean(), color=self.colors['normal'], 
                   linestyle=':', linewidth=2.5, alpha=0.8)
        ax1.axvline(attack_scores.mean(), color=self.colors['attack'], 
                   linestyle=':', linewidth=2.5, alpha=0.8)
        
        # Bar chart by attack type with error bars
        attack_types = sorted(attack_quality.index)
        colors = [self.colors['primary'][i % len(self.colors['primary'])] 
                 for i in range(len(attack_types))]
        
        bars = ax2.bar(range(len(attack_types)), attack_quality['mean'], 
                       yerr=attack_quality['std'], capsize=5,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('Attack Type', fontsize=14)
        ax2.set_ylabel('Mean Quality Score', fontsize=14)
        ax2.set_title('Average Quality Score by Attack Type', 
                     fontsize=18, fontweight='bold', pad=20)
        ax2.set_xticks(range(len(attack_types)))
        ax2.set_xticklabels(attack_types, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (bar, attack_type) in enumerate(zip(bars, attack_types)):
            height = attack_quality.loc[attack_type, 'mean']
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        
        # Add horizontal line for normal average
        normal_mean = self.df[self.df['Attack_Flag'] == 0]['Quality_Score'].mean()
        ax2.axhline(y=normal_mean, color=self.colors['normal'], 
                   linestyle='--', linewidth=2.5,
                   label=f'Normal Average: {normal_mean:.3f}')
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        self.save_publication_figure(fig, 'quality_score_analysis.png')
        
        # Save statistics
        quality_stats.to_csv(self.output_dir / 'quality_score_statistics.csv')
        attack_quality.to_csv(self.output_dir / 'quality_score_by_attack_type.csv')
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n GENERATING SUMMARY REPORT")
        print("-" * 60)
        
        report_file = self.output_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("SATELLITE ATTACK DATASET COMPREHENSIVE STATISTICS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.csv_file}\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset Overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Records: {len(self.df):,}\n")
            f.write(f"Attack Records: {len(self.df[self.df['Attack_Flag'] == 1]):,} ({len(self.df[self.df['Attack_Flag'] == 1])/len(self.df)*100:.1f}%)\n")
            f.write(f"Normal Records: {len(self.df[self.df['Attack_Flag'] == 0]):,} ({len(self.df[self.df['Attack_Flag'] == 0])/len(self.df)*100:.1f}%)\n")
            f.write(f"Time Range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}\n\n")
            
            # Attack Type Distribution
            f.write("2. ATTACK TYPE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            attack_counts = self.df['Attack_Type'].value_counts()
            for attack_type, count in attack_counts.items():
                f.write(f"{attack_type:<15} {count:>8,} ({count/len(self.df)*100:>5.1f}%)\n")
            f.write("\n")
            
            # Attack Severity Summary
            f.write("3. ATTACK SEVERITY SUMMARY\n")
            f.write("-" * 40 + "\n")
            attack_df = self.df[self.df['Attack_Flag'] == 1]
            severity_by_type = attack_df.groupby('Attack_Type')['Attack_Severity'].agg(['mean', 'std', 'min', 'max'])
            f.write(f"{'Attack Type':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n")
            for attack_type, row in severity_by_type.iterrows():
                f.write(f"{attack_type:<15} {row['mean']:>8.3f} {row['std']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f}\n")
            f.write("\n")
            
            # RF Metrics Impact
            f.write("4. RF METRICS IMPACT (Attack vs Normal)\n")
            f.write("-" * 40 + "\n")
            rf_columns = ['RF_SNR_dB', 'RF_Packet_Error_Rate', 'RF_Throughput_bps']
            normal_df = self.df[self.df['Attack_Flag'] == 0]
            
            for col in rf_columns:
                normal_mean = normal_df[col].mean()
                attack_mean = attack_df[col].mean()
                change = ((attack_mean - normal_mean) / normal_mean * 100)
                f.write(f"{col:<25} Normal: {normal_mean:>10.3f}  Attack: {attack_mean:>10.3f}  Change: {change:>7.1f}%\n")
            f.write("\n")
            
            # Detection Confidence
            f.write("5. DETECTION CONFIDENCE BY ATTACK TYPE\n")
            f.write("-" * 40 + "\n")
            detection_conf = attack_df.groupby('Attack_Type')['Detection_Confidence'].agg(['mean', 'min', 'max'])
            f.write(f"{'Attack Type':<15} {'Mean':>8} {'Min':>8} {'Max':>8}\n")
            for attack_type, row in detection_conf.iterrows():
                f.write(f"{attack_type:<15} {row['mean']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f}\n")
            f.write("\n")
            
            # Satellite Analysis
            f.write("6. SATELLITE IMPACT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            satellite_stats = self.df.groupby('Satellite_ID')['Attack_Flag'].agg(['count', 'sum', 'mean'])
            for sat_id, row in satellite_stats.iterrows():
                f.write(f"Satellite {sat_id}: {row['count']:,} records, {row['sum']:,} attacks ({row['mean']*100:.1f}%)\n")
            f.write("\n")
            
            # Key Findings
            f.write("7. KEY FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            # Most common attack
            most_common_attack = attack_counts.index[0]
            f.write(f"• Most common attack type: {most_common_attack} ({attack_counts.iloc[0]:,} occurrences)\n")
            
            # Highest severity attack
            highest_severity = attack_df.groupby('Attack_Type')['Attack_Severity'].mean().idxmax()
            f.write(f"• Highest average severity: {highest_severity} ({attack_df[attack_df['Attack_Type']==highest_severity]['Attack_Severity'].mean():.3f})\n")
            
            # Most detectable attack
            most_detectable = attack_df.groupby('Attack_Type')['Detection_Confidence'].mean().idxmax()
            f.write(f"• Most detectable attack: {most_detectable} ({attack_df[attack_df['Attack_Type']==most_detectable]['Detection_Confidence'].mean():.3f})\n")
            
            # Longest duration attack
            longest_duration = attack_df.groupby('Attack_Type')['Attack_Duration'].mean().idxmax()
            f.write(f"• Longest average duration: {longest_duration} ({attack_df[attack_df['Attack_Type']==longest_duration]['Attack_Duration'].mean():.1f} seconds)\n")
            
            # Most impactful on SNR
            snr_impact = {}
            for attack_type in attack_df['Attack_Type'].unique():
                if attack_type != 'normal':
                    attack_snr = attack_df[attack_df['Attack_Type']==attack_type]['RF_SNR_dB'].mean()
                    normal_snr = normal_df['RF_SNR_dB'].mean()
                    snr_impact[attack_type] = normal_snr - attack_snr
            
            most_impactful = max(snr_impact, key=snr_impact.get)
            f.write(f"• Most impactful on SNR: {most_impactful} ({snr_impact[most_impactful]:.1f} dB reduction)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated successfully.\n")
        
        print(f" Summary report saved to: {report_file}")
        
        # Also generate a JSON summary for programmatic access
        summary_json = {
            "dataset_info": {
                "file": str(self.csv_file),
                "total_records": int(len(self.df)),
                "attack_records": int(len(self.df[self.df['Attack_Flag'] == 1])),
                "normal_records": int(len(self.df[self.df['Attack_Flag'] == 0])),
                "attack_ratio": float(len(self.df[self.df['Attack_Flag'] == 1]) / len(self.df) * 100),
                "time_range": {
                    "start": str(self.df['Timestamp'].min()),
                    "end": str(self.df['Timestamp'].max())
                }
            },
            "attack_distribution": attack_counts.to_dict(),
            "severity_statistics": severity_by_type.to_dict(),
            "detection_confidence": detection_conf.to_dict(),
            "key_findings": {
                "most_common_attack": most_common_attack,
                "highest_severity_attack": highest_severity,
                "most_detectable_attack": most_detectable,
                "longest_duration_attack": longest_duration,
                "most_impactful_on_snr": most_impactful
            }
        }
        
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_json, f, indent=2, default=str)
        
        print(f" JSON summary saved to: {self.output_dir / 'summary_statistics.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive statistics from satellite attack dataset with publication-quality visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--input', required=True, 
                       help='Input CSV file (e.g., balanced_50_enhanced_realtime.csv)')
    parser.add_argument('--output-dir', default='statistics',
                       help='Output directory for statistics and visualizations (default: statistics)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f" Error: Input file not found: {args.input}")
        return
    
    # Create analyzer and generate statistics
    analyzer = SatelliteAttackStatistics(args.input, args.output_dir)
    analyzer.generate_all_statistics()
    
    print("\n Statistics generation complete!")
    print(f" All results saved in: {args.output_dir}/")
    print("\nGenerated files:")
    print("  Visualizations: *.png (300 DPI)")
    print("  PDF versions: *.pdf")
    print("  Data files: *.csv")
    print("  Reports: summary_report.txt, summary_statistics.json")


if __name__ == "__main__":
    main()