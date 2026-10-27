import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

os.makedirs('visualizations', exist_ok=True)

# 1. Load dataset
df = pd.read_csv('dataset.csv')
print("Dataset Loaded Successfully.")
print(f"Shape: {df.shape}")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Color palette
palette = sns.color_palette("mako", 10)

# -------------------------------------------------------------
# Visualization 1: Intent Distribution Across Product Categories
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
intent_counts = df['intent'].value_counts()
sns.barplot(x=intent_counts.values, y=intent_counts.index, palette="viridis", ax=ax)
ax.set_title('Figure 1: Distribution of Customer Support Inquiries Across 10 Categories', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Total Turns / Interactions', fontsize=12)
ax.set_ylabel('Intent Category', fontsize=12)
for i, v in enumerate(intent_counts.values):
    ax.text(v + 15, i, f"{v} ({v/len(df)*100:.1f}%)", va='center', fontsize=10, fontweight='semibold')
plt.tight_layout()
fig.savefig('visualizations/fig1_intent_distribution.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig1_intent_distribution.png")

# -------------------------------------------------------------
# Visualization 2: Escalation Rate by Intent and Dialogue Turn
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Escalation by intent
intent_esc = df.groupby('intent')['escalated'].mean().sort_values(ascending=False) * 100
sns.barplot(x=intent_esc.values, y=intent_esc.index, palette="rocket", ax=ax1)
ax1.set_title('(A) Escalation Rate (%) by Intent Category', fontsize=12, fontweight='bold')
ax1.set_xlabel('Escalation Rate (%)', fontsize=11)
ax1.set_ylabel('Intent', fontsize=11)
for i, v in enumerate(intent_esc.values):
    ax1.text(v + 0.3, i, f"{v:.1f}%", va='center', fontsize=9)

# Escalation by turn_id
turn_esc = df.groupby('turn_id')['escalated'].mean() * 100
sns.lineplot(x=turn_esc.index, y=turn_esc.values, marker='o', color='#e74c3c', linewidth=2.5, markersize=8, ax=ax2)
ax2.set_title('(B) Escalation Rate (%) vs Conversation Turn Number', fontsize=12, fontweight='bold')
ax2.set_xlabel('Dialogue Turn Index (turn_id)', fontsize=11)
ax2.set_ylabel('Escalation Rate (%)', fontsize=11)
ax2.set_xticks(range(1, 11))
ax2.grid(True, linestyle='--', alpha=0.6)
for x, y in zip(turn_esc.index, turn_esc.values):
    ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.suptitle('Figure 2: Escalation Trigger Dynamics across Intents and Conversation Length', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('visualizations/fig2_escalation_dynamics.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig2_escalation_dynamics.png")

# -------------------------------------------------------------
# Visualization 3: NLU Confidence Score vs Customer Satisfaction
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(x='satisfaction_score', y='confidence_score', data=df, palette="crest", ax=ax, width=0.5, boxprops=dict(alpha=0.8))
sns.stripplot(x='satisfaction_score', y='confidence_score', data=df.sample(600, random_state=42), color='black', alpha=0.2, jitter=0.2, size=3, ax=ax)
ax.set_title('Figure 3: NLU Confidence Score Distribution across Satisfaction Ratings (1-5)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Customer Satisfaction Score (CSAT Likert Scale)', fontsize=12)
ax.set_ylabel('NLU Confidence Score', fontsize=12)
ax.axhline(0.68, color='red', linestyle='--', linewidth=1.5, label='Escalation / Fallback Threshold (0.68)')
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()
fig.savefig('visualizations/fig3_confidence_vs_satisfaction.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig3_confidence_vs_satisfaction.png")

# -------------------------------------------------------------
# Visualization 4: Response Latency Distribution & Outlier/Faults Detection
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.histplot(df['response_time_ms'], kde=True, color='#2980b9', bins=40, ax=ax, stat='density', alpha=0.5)
ax.set_title('Figure 4: End-to-End Response Latency (ms) with Anomaly Latency Zone', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Response Time (milliseconds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.axvline(df['response_time_ms'].mean(), color='navy', linestyle='-', linewidth=2, label=f"Mean Latency ({df['response_time_ms'].mean():.0f} ms)")
ax.axvline(4500, color='crimson', linestyle='--', linewidth=2, label='Service Level Agreement (SLA) Anomaly Threshold (4,500 ms)')
ax.fill_betweenx([0, 0.00045], 4500, 5000, color='red', alpha=0.15, label='Anomaly/Fault Zone (Latency Spikes ~3.5%)')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
fig.savefig('visualizations/fig4_response_time_latency.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig4_response_time_latency.png")

# -------------------------------------------------------------
# Visualization 5: Correlation Matrix
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
numeric_df = df[['turn_id', 'confidence_score', 'escalated', 'satisfaction_score', 'response_time_ms']]
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'Pearson Correlation'}, ax=ax, vmin=-0.6, vmax=0.6)
ax.set_title('Figure 5: Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('visualizations/fig5_correlation_matrix.png', bbox_inches='tight')
plt.close(fig)
print("Saved fig5_correlation_matrix.png")

print("All 5 Visualizations Successfully Generated in visualizations/ directory!")
