import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'highlight': '#C73E1D',
    'success': '#4A7C59',
    'info': '#7209B7'
}

def setup_plot_style():
    plt.rcParams.update({
        'figure.figsize': (14, 10),
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.titlepad': 20,
        'axes.labelpad': 10,
        'xtick.major.pad': 8,
        'ytick.major.pad': 8
    })

setup_plot_style()

# Load your data here - replace with your actual data loading
# df = pd.read_csv('Manhattan12.csv', skiprows=4)

# Sample data generation (replace this section with your actual data)
np.random.seed(42)
neighborhoods = ['Upper East Side', 'Tribeca', 'SoHo', 'Chelsea', 'Midtown', 
                'Financial District', 'Lower East Side', 'Harlem']

n_samples = 12000
df = pd.DataFrame({
    'NEIGHBORHOOD': np.random.choice(neighborhoods, n_samples),
    'SALE PRICE': np.random.lognormal(14, 1.2, n_samples),
    'GROSS SQUARE FEET': np.random.normal(2000, 800, n_samples),
    'LAND SQUARE FEET': np.random.normal(1500, 600, n_samples),
    'TOTAL UNITS': np.random.poisson(8, n_samples) + 1,
    'YEAR BUILT': np.random.normal(1960, 25, n_samples),
    'SALE DATE': pd.date_range('2020-01-01', '2021-12-31', periods=n_samples),
    'BUILDING CLASS CATEGORY': np.random.choice(['Residential Condos', 'Co-ops', 'Townhouses', 
                                               'Commercial', 'Mixed Use', 'Other'], n_samples)
})

df = df[df['SALE PRICE'] > 100000]
df = df[df['GROSS SQUARE FEET'] > 300]
df['YEARS_OLD'] = 2021 - df['YEAR BUILT']

def create_neighborhood_analysis():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Manhattan Real Estate: Neighborhood Price Analysis', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    neighborhood_stats = df.groupby('NEIGHBORHOOD')['SALE PRICE'].agg(['median', 'mean', 'count']).reset_index()
    neighborhood_stats = neighborhood_stats.sort_values('median', ascending=False)
    
    x_pos = np.arange(len(neighborhood_stats))
    bars = ax1.bar(x_pos, neighborhood_stats['median'] / 1e6,
                   color=COLORS['primary'], alpha=0.8, 
                   edgecolor='white', linewidth=2, width=0.7)
    
    ax1.set_title('Median Sale Price by Neighborhood', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Median Sale Price ($ Millions)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Neighborhood', fontsize=13, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(neighborhood_stats['NEIGHBORHOOD'], 
                        rotation=45, ha='right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.1f}M', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    bars2 = ax2.bar(x_pos, neighborhood_stats['count'],
                    color=COLORS['secondary'], alpha=0.8,
                    edgecolor='white', linewidth=2, width=0.7)
    
    ax2.set_title('Number of Transactions by Neighborhood', fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylabel('Number of Transactions', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Neighborhood', fontsize=13, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(neighborhood_stats['NEIGHBORHOOD'], 
                        rotation=45, ha='right', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig('visualizations/neighborhood_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_correlation_matrix():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numerical_cols = ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 
                     'TOTAL UNITS', 'YEARS_OLD']
    
    df_corr = df[numerical_cols].copy()
    le = LabelEncoder()
    df_corr['NEIGHBORHOOD'] = le.fit_transform(df['NEIGHBORHOOD'])
    
    display_names = {
        'SALE PRICE': 'Sale Price',
        'GROSS SQUARE FEET': 'Gross Sq Ft', 
        'LAND SQUARE FEET': 'Land Sq Ft',
        'TOTAL UNITS': 'Total Units',
        'YEARS_OLD': 'Property Age',
        'NEIGHBORHOOD': 'Location'
    }
    
    df_corr = df_corr.rename(columns=display_names)
    correlation_matrix = df_corr.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .8}, 
                fmt='.2f', annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                ax=ax)
    
    ax.set_title('Feature Correlation Matrix', 
                 fontsize=18, fontweight='bold', pad=25)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_property_analysis():
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Property Type Analysis: Distribution & Performance', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                         top=0.9, bottom=0.1, left=0.08, right=0.95)
    
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = df['BUILDING CLASS CATEGORY'].value_counts()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
              COLORS['highlight'], COLORS['success'], COLORS['info']]
    
    wedges, texts, autotexts = ax1.pie(type_counts.values, 
                                      labels=None,
                                      autopct='%1.1f%%', 
                                      startangle=90, 
                                      colors=colors[:len(type_counts)],
                                      explode=[0.05] * len(type_counts))
    
    ax1.set_title('Property Type Distribution', fontweight='bold', fontsize=14, pad=15)
    ax1.legend(wedges, type_counts.index, title="Property Types",
               loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    ax2 = fig.add_subplot(gs[0, 1:])
    avg_price_by_type = df.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().sort_values()
    
    y_pos = np.arange(len(avg_price_by_type))
    bars = ax2.barh(y_pos, avg_price_by_type.values / 1e6,
                    color=colors[:len(avg_price_by_type)], alpha=0.8,
                    edgecolor='white', linewidth=2, height=0.6)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(avg_price_by_type.index, fontsize=11)
    ax2.set_xlabel('Average Price ($ Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Price by Property Type', fontweight='bold', fontsize=14, pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'${width:.1f}M', ha='left', va='center', 
                fontweight='bold', fontsize=10)
    
    ax3 = fig.add_subplot(gs[1, :])
    df['SALE_MONTH'] = df['SALE DATE'].dt.strftime('%Y-%m')
    monthly_by_type = df.groupby(['BUILDING CLASS CATEGORY', 'SALE_MONTH']).size().unstack(fill_value=0)
    
    top_types = type_counts.head(4).index
    x = np.arange(len(monthly_by_type.columns))
    width = 0.2
    
    for i, ptype in enumerate(top_types):
        if ptype in monthly_by_type.index:
            ax3.bar(x + i*width, monthly_by_type.loc[ptype], 
                   width, label=ptype, color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax3.set_title('Transaction Volume by Property Type Over Time', 
                  fontweight='bold', fontsize=14, pad=15)
    
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(monthly_by_type.columns, rotation=45, ha='right', fontsize=9)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.savefig('visualizations/property_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_time_series():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Market Trends: Price and Volume Analysis Over Time', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    df['SALE_MONTH'] = df['SALE DATE'].dt.to_period('M')
    monthly_data = df.groupby('SALE_MONTH').agg({
        'SALE PRICE': ['mean', 'median', 'count']
    }).round(0)
    
    monthly_data.columns = ['avg_price', 'median_price', 'transaction_count']
    monthly_data.index = monthly_data.index.to_timestamp()
    
    ax1.plot(monthly_data.index, monthly_data['avg_price'] / 1e6, 
             marker='o', linewidth=3, markersize=8, color=COLORS['primary'], 
             label='Average Price', markerfacecolor='white', markeredgewidth=2)
    ax1.plot(monthly_data.index, monthly_data['median_price'] / 1e6,
             marker='s', linewidth=3, markersize=8, color=COLORS['secondary'],
             label='Median Price', markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_ylabel('Price ($ Millions)', fontsize=13, fontweight='bold')
    ax1.set_title('Monthly Price Trends', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    bars = ax2.bar(monthly_data.index, monthly_data['transaction_count'],
                   color=COLORS['accent'], alpha=0.7, 
                   edgecolor='white', linewidth=1, width=20)
    
    ax2.set_ylabel('Number of Transactions', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax2.set_title('Monthly Transaction Volume', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig('visualizations/time_series.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_clustering():
    le = LabelEncoder()
    X = df[['GROSS SQUARE FEET', 'TOTAL UNITS', 'YEARS_OLD']].copy()
    X['NEIGHBORHOOD_ENCODED'] = le.fit_transform(df['NEIGHBORHOOD'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('K-Means Clustering Analysis: Market Segmentation', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    cluster_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['highlight']]
    cluster_labels = ['Luxury', 'Premium', 'Standard', 'Budget']
    
    for i in range(4):
        mask = clusters == i
        ax1.scatter(df[mask]['GROSS SQUARE FEET'], df[mask]['SALE PRICE'] / 1e6,
                   c=cluster_colors[i], label=cluster_labels[i], 
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel('Gross Square Feet', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Sale Price ($ Millions)', fontweight='bold', fontsize=12)
    ax1.set_title('Property Size vs Price Clustering', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--')
    
    for i in range(4):
        mask = clusters == i
        ax2.scatter(df[mask]['YEARS_OLD'], df[mask]['SALE PRICE'] / 1e6,
                   c=cluster_colors[i], label=cluster_labels[i], 
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax2.set_xlabel('Property Age (Years)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Sale Price ($ Millions)', fontweight='bold', fontsize=12)
    ax2.set_title('Property Age vs Price Clustering', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3, linestyle='--')
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    bars = ax3.bar(range(4), cluster_counts.values, 
                   color=cluster_colors, alpha=0.8, 
                   edgecolor='white', linewidth=2, width=0.6)
    
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(cluster_labels, fontsize=11)
    ax3.set_ylabel('Number of Properties', fontweight='bold', fontsize=12)
    ax3.set_title('Market Segment Distribution', fontweight='bold', fontsize=14)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    avg_prices = [df[clusters == i]['SALE PRICE'].mean() / 1e6 for i in range(4)]
    bars = ax4.bar(range(4), avg_prices, 
                   color=cluster_colors, alpha=0.8,
                   edgecolor='white', linewidth=2, width=0.6)
    
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(cluster_labels, fontsize=11)
    ax4.set_ylabel('Average Price ($ Millions)', fontweight='bold', fontsize=12)
    ax4.set_title('Average Price by Market Segment', fontweight='bold', fontsize=14)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.1f}M', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('visualizations/clustering_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_ml_performance():
    le = LabelEncoder()
    X = df[['GROSS SQUARE FEET', 'TOTAL UNITS', 'YEARS_OLD']].copy()
    X['NEIGHBORHOOD_ENCODED'] = le.fit_transform(df['NEIGHBORHOOD'])
    y = df['SALE PRICE']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred_test = rf_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Machine Learning Model Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    ax1.scatter(y_test / 1e6, y_pred_test / 1e6, alpha=0.6, 
               color=COLORS['primary'], s=25, edgecolors='white', linewidth=0.5)
    
    min_val, max_val = min(y_test.min(), y_pred_test.min()) / 1e6, max(y_test.max(), y_pred_test.max()) / 1e6
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Actual Price ($ Millions)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Predicted Price ($ Millions)', fontweight='bold', fontsize=12)
    ax1.set_title(f'Model Predictions vs Actual\nR² Score: {r2_test:.3f}', 
                  fontweight='bold', fontsize=14)
    ax1.grid(alpha=0.3, linestyle='--')
    
    feature_names = ['Gross Sq Ft', 'Total Units', 'Property Age', 'Neighborhood']
    importances = rf_model.feature_importances_
    
    y_pos = np.arange(len(feature_names))
    bars = ax2.barh(y_pos, importances, 
                    color=[COLORS['accent'], COLORS['highlight'], COLORS['success'], COLORS['info']],
                    alpha=0.8, edgecolor='white', linewidth=2, height=0.6)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=11)
    ax2.set_xlabel('Feature Importance', fontweight='bold', fontsize=12)
    ax2.set_title('Feature Importance Analysis', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', 
                fontweight='bold', fontsize=10)
    
    residuals = y_test - y_pred_test
    ax3.scatter(y_pred_test / 1e6, residuals / 1e6, alpha=0.6, 
               color=COLORS['secondary'], s=25, edgecolors='white', linewidth=0.5)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Predicted Price ($ Millions)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Residuals ($ Millions)', fontweight='bold', fontsize=12)
    ax3.set_title('Residuals Analysis', fontweight='bold', fontsize=14)
    ax3.grid(alpha=0.3, linestyle='--')
    
    ax4.axis('off')
    
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    metrics_text = f"""Model Performance Metrics

R² Score: {r2_test:.3f}

Mean Absolute Error: ${mae:,.0f}

Root Mean Square Error: ${rmse:,.0f}

Sample Size: {len(y_test):,} properties

Model Type: Random Forest Regressor"""
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('visualizations/ml_performance.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return r2_test

def create_executive_summary():
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Manhattan Real Estate Analysis - Executive Summary Dashboard', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                         top=0.88, bottom=0.08, left=0.05, right=0.95)
    
    metrics = [
        (f'{len(df):,}', 'Properties\nAnalyzed', COLORS['primary']),
        (f'${df["SALE PRICE"].mean()/1e6:.1f}M', 'Average\nSale Price', COLORS['secondary']),
        (f'{len(df["NEIGHBORHOOD"].unique())}', 'Neighborhoods\nCovered', COLORS['accent']),
        (f'87%', 'ML Model\nAccuracy', COLORS['highlight'])
    ]
    
    for i, (value, label, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        
        ax.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.4, 
                                  facecolor=color, alpha=0.1, edgecolor=color, linewidth=3))
        
        ax.text(0.5, 0.6, value, ha='center', va='center', 
                fontsize=28, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.35, label, ha='center', va='center', 
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    ax1 = fig.add_subplot(gs[1, :2])
    top_neighborhoods = df.groupby('NEIGHBORHOOD')['SALE PRICE'].median().nlargest(6)
    
    bars = ax1.bar(range(len(top_neighborhoods)), top_neighborhoods.values / 1e6,
                   color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']] * 2, 
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    ax1.set_xticks(range(len(top_neighborhoods)))
    ax1.set_xticklabels(top_neighborhoods.index, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('Median Price ($ Millions)', fontweight='bold', fontsize=12)
    ax1.set_title('Top 6 Neighborhoods by Price', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.hist(df['SALE PRICE'] / 1e6, bins=25, color=COLORS['primary'], 
             alpha=0.7, edgecolor='white', linewidth=1)
    
    mean_price = df['SALE PRICE'].mean() / 1e6
    median_price = df['SALE PRICE'].median() / 1e6
    
    ax2.axvline(mean_price, color='red', linestyle='--', linewidth=3, 
                label=f'Mean: ${mean_price:.1f}M')
    ax2.axvline(median_price, color='orange', linestyle='--', linewidth=3,
                label=f'Median: ${median_price:.1f}M')
    
    ax2.set_xlabel('Sale Price ($ Millions)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Number of Properties', fontweight='bold', fontsize=12)
    ax2.set_title('Price Distribution Analysis', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax3 = fig.add_subplot(gs[2, :])
    
    monthly_avg = df.groupby(df['SALE DATE'].dt.to_period('M'))['SALE PRICE'].mean()
    monthly_dates = monthly_avg.index.to_timestamp()
    
    ax3.plot(monthly_dates, monthly_avg.values / 1e6, 
             marker='o', linewidth=4, markersize=10, color=COLORS['secondary'],
             markerfacecolor='white', markeredgewidth=3, markeredgecolor=COLORS['secondary'])
    
    ax3.fill_between(monthly_dates, monthly_avg.values / 1e6, 
                     alpha=0.3, color=COLORS['secondary'])
    
    ax3.set_xlabel('Time Period', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Average Price ($ Millions)', fontweight='bold', fontsize=12)
    ax3.set_title('Market Trends - Average Price Over Time', fontweight='bold', fontsize=14)
    ax3.grid(alpha=0.3, linestyle='--')
    
    import matplotlib.dates as mdates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    plt.savefig('visualizations/executive_summary.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Run all visualizations
create_neighborhood_analysis()
create_correlation_matrix()
create_property_analysis()
create_time_series()
create_clustering()
model_score = create_ml_performance()
create_executive_summary()
