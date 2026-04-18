"""
Visualization: All Steps Score Progression & Model Comparisons
================================================================
Generates comprehensive figures showing:
1. AUROC progression across all steps
2. PR-AUC and F1-Macro progression
3. Model comparisons at each step
4. Paper vs Ours comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# =============================================================================
# DATA COLLECTION - All results from each step
# =============================================================================

step_data = {
    'Step': [
        'Phase 1\n(Arthritis)',
        'Step 02\nBaseline',
        'Step 03\nKDM-BA',
        'Step 05-07\nTuning',
        'Step 08\nXGB Fix',
        'Step 09\nAdvanced',
        'Step 10\nPaper Repl.',
    ],
    'best_auroc': [0.7058, 0.8937, 0.8652, 0.8924, 0.8967, 0.8967, 0.8914],
    'pr_auc':    [None,   0.7783, None,    None,    0.7783, 0.7876, 0.7628],
    'f1_macro':  [0.45,   0.8644, None,    None,    0.8644, 0.8718, 0.8692],
    'f1_binary': [0.45,   0.7596, None,    None,    0.7596, 0.7488, 0.7554],
    'best_model': [
        'Soft Voting',
        'RF (raw17)',
        'RF (raw17+KDM)',
        'RF_tuned',
        'RF (raw17, youden)',
        'RF (raw17)',
        'RF (paper_13)',
    ],
    'n_features': [17, 17, 21, 17, 17, 17, 13],
    'paper_auroc': [0.9078]*7,
}

step_data_models = {
    'Step 02 Baseline': {
        'RF':           {'auroc': 0.8937, 'f1': 0.7578},
        'XGBoost':      {'auroc': 0.8048, 'f1': 0.3139},
        'LightGBM':     {'auroc': 0.7757, 'f1': 0.1342},
        'CatBoost':     {'auroc': 0.7113, 'f1': 0.0598},
        'LR':           {'auroc': 0.6579, 'f1': 0.0072},
    },
    'Step 05-07 Tuning': {
        'RF_tuned':     {'auroc': 0.8924, 'f1': None},
        'Ensemble':     {'auroc': 0.8740, 'f1': None},
        'XGBoost_tuned':{'auroc': 0.8497, 'f1': None},
        'LGBM_tuned':  {'auroc': 0.8370, 'f1': None},
        'CatBoost_tuned':{'auroc':0.8239, 'f1': None},
    },
    'Step 08 XGB Fix': {
        'RF (int, 0.5)':      {'auroc': 0.8967, 'f1': 0.7596},
        'RF (int, youden)':    {'auroc': 0.8967, 'f1': 0.6701},
        'RF (OHE, 0.5)':      {'auroc': 0.8954, 'f1': 0.7580},
        'XGB_spw (int, 0.5)': {'auroc': 0.7654, 'f1': 0.3963},
        'XGB_spw (OHE, 0.5)': {'auroc': 0.7651, 'f1': 0.3949},
        'LGBM_spw (OHE, 0.5)': {'auroc': 0.8012, 'f1': 0.4541},
    },
    'Step 09 Advanced': {
        'RF (raw17)':           {'auroc': 0.8967, 'pr_auc': 0.7783, 'f1_macro': 0.8644},
        'XGB_optuna':           {'auroc': 0.8563, 'pr_auc': 0.7272, 'f1_macro': 0.8630},
        'LGBM_optuna':          {'auroc': 0.8652, 'pr_auc': 0.7639, 'f1_macro': 0.8696},
        'Stack RF+XGB+LGBM':   {'auroc': 0.8925, 'pr_auc': 0.7876, 'f1_macro': 0.8718},
        'Stack RF+XGB':        {'auroc': 0.8914, 'pr_auc': 0.7829, 'f1_macro': 0.8707},
        'Stack RFbal+XGB':     {'auroc': 0.8903, 'pr_auc': 0.7812, 'f1_macro': 0.8712},
    },
    'Step 10 Paper Repl.': {
        'RF (paper_13_contBMI)':  {'auroc': 0.8914, 'pr_auc': 0.7628, 'f1_macro': 0.8692},
        'XGB_optuna (paper_13)':  {'auroc': 0.8603, 'pr_auc': 0.7314, 'f1_macro': 0.8598},
        'RF (paper_13_BMI_New)':  {'auroc': 0.8544, 'pr_auc': 0.6030, 'f1_macro': 0.8394},
        'RF (paper_11)':          {'auroc': 0.8497, 'pr_auc': 0.5837, 'f1_macro': 0.8361},
        'XGB_default (paper_13)': {'auroc': 0.7677, 'pr_auc': 0.3987, 'f1_macro': 0.6572},
        'XGB_spw (paper_11)':     {'auroc': 0.7279, 'pr_auc': 0.3142, 'f1_macro': 0.6053},
    },
}

feature_impact = {
    'raw17':             0.8937,
    'raw17+biomarkers':  0.7269,
    'raw17+KDM_log':     0.8626,
    'raw17+KDM_orig':    0.8652,
    'raw17+all':         0.6958,
    'raw17+BIR':         0.7949,
    'raw17+interactions':0.8143,
    'OHE (no biomarkers)': 0.8954,
    'OHE+PCA+biomarkers':  0.7481,
    'OHE+KDM':             0.8700,
}

colors = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'accent': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'paper': '#DC2626',
    'ours': '#2563EB',
    'light_blue': '#93C5FD',
    'light_purple': '#C4B5FD',
    'light_green': '#6EE7B7',
    'gray': '#9CA3AF',
}

# =============================================================================
# FIGURE 1: AUROC Progression Across Steps
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1a: AUROC progression
ax = axes[0]
steps = step_data['Step']
aurocs = step_data['best_auroc']
paper_aucs = step_data['paper_auroc']

x = range(len(steps))
bars = ax.bar(x, aurocs, color=colors['ours'], alpha=0.85, width=0.6, zorder=3)
ax.plot(x, paper_aucs, 'r--', linewidth=2, label='Paper (0.9078)', zorder=4)

for i, (bar, auc) in enumerate(zip(bars, aurocs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=8)
ax.set_ylabel('AUROC', fontweight='bold')
ax.set_title('AUROC Progression Across Steps', fontweight='bold', fontsize=11)
ax.set_ylim(0.65, 0.95)
ax.axhline(y=0.9078, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 1b: Feature impact
ax = axes[1]
feat_names = list(feature_impact.keys())
feat_aucs = list(feature_impact.values())
colors_feat = [colors['primary'] if v >= 0.89 else colors['danger'] for v in feat_aucs]
bars = ax.barh(range(len(feat_names)), feat_aucs, color=colors_feat, alpha=0.85)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=9)
ax.set_xlabel('AUROC', fontweight='bold')
ax.set_title('Feature Set Impact on RF Performance', fontweight='bold', fontsize=11)
ax.axvline(x=0.8937, color=colors['primary'], linestyle=':', alpha=0.7, label='raw17 baseline')
ax.axvline(x=0.9078, color='red', linestyle='--', alpha=0.7, label='Paper')
for bar, auc in zip(bars, feat_aucs):
    ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2, f'{auc:.4f}',
            va='center', fontsize=8)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0.65, 0.95)
ax.grid(axis='x', alpha=0.3)

# 1c: Model comparison at best step (Step 10)
ax = axes[2]
models_s10 = list(step_data_models['Step 10 Paper Repl.'].keys())
aurocs_s10 = [v['auroc'] for v in step_data_models['Step 10 Paper Repl.'].values()]
colors_s10 = [colors['primary'] if v >= 0.85 else colors['warning'] if v >= 0.75 else colors['danger'] for v in aurocs_s10]
bars = ax.barh(range(len(models_s10)), aurocs_s10, color=colors_s10, alpha=0.85)
ax.set_yticks(range(len(models_s10)))
ax.set_yticklabels(models_s10, fontsize=9)
ax.set_xlabel('AUROC', fontweight='bold')
ax.set_title('Step 10: Paper Replication Models', fontweight='bold', fontsize=11)
ax.axvline(x=0.9078, color='red', linestyle='--', alpha=0.7, label='Paper (0.9078)')
for bar, auc in zip(bars, aurocs_s10):
    ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2, f'{auc:.4f}',
            va='center', fontsize=8)
ax.legend(fontsize=8)
ax.set_xlim(0.65, 0.95)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'fig1_progression_and_comparisons.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig1_progression_and_comparisons.png")

# =============================================================================
# FIGURE 2: Detailed Step 09 & Step 10 Comparisons
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 2a: Step 09 - All models AUROC
ax = axes[0, 0]
s09_models = list(step_data_models['Step 09 Advanced'].keys())
s09_aucs = [v['auroc'] for v in step_data_models['Step 09 Advanced'].values()]
s09_praucs = [v['pr_auc'] for v in step_data_models['Step 09 Advanced'].values()]
s09_f1s = [v['f1_macro'] for v in step_data_models['Step 09 Advanced'].values()]

x = np.arange(len(s09_models))
w = 0.25
bars1 = ax.bar(x - w, s09_aucs, w, label='AUROC', color=colors['primary'], alpha=0.85)
bars2 = ax.bar(x, s09_praucs, w, label='PR-AUC', color=colors['accent'], alpha=0.85)
bars3 = ax.bar(x + w, s09_f1s, w, label='F1-Macro', color=colors['secondary'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(s09_models, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Step 09: Advanced Pipeline (Optuna + Stacking)', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.70, 0.95)
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

# 2b: Step 10 - Paper replication models
ax = axes[0, 1]
s10_fs = ['paper_11', 'paper_13\n(BMI_New)', 'paper_13\n(cont BMI)']
s10_rf = [0.8497, 0.8544, 0.8914]
s10_xgb_def = [0.7613, 0.7677, 0.8079]
s10_xgb_spw = [0.7279, 0.7318, 0.7659]
s10_xgb_opt = [None, None, 0.8603]

x = np.arange(len(s10_fs))
w = 0.2
ax.bar(x - 1.5*w, s10_rf, w, label='RF', color=colors['primary'], alpha=0.85)
ax.bar(x - 0.5*w, s10_xgb_def, w, label='XGB_default', color=colors['secondary'], alpha=0.85)
ax.bar(x + 0.5*w, s10_xgb_spw, w, label='XGB_spw', color=colors['warning'], alpha=0.85)
s10_opt = [v if v is not None else 0 for v in s10_xgb_opt]
ax.bar(x + 1.5*w, s10_opt, w, label='XGB_optuna', color=colors['accent'], alpha=0.85)
ax.axhline(y=0.9078, color='red', linestyle='--', linewidth=1.5, label='Paper (0.9078)')
ax.set_xticks(x)
ax.set_xticklabels(s10_fs, fontsize=10)
ax.set_ylabel('AUROC (5-fold CV)', fontweight='bold')
ax.set_title('Step 10: Paper Feature Space\n(5-fold CV × 3 seeds)', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(0.65, 0.95)
ax.grid(axis='y', alpha=0.3)

# 2c: Feature set comparison (paper features)
ax = axes[1, 0]
feat_sets = ['raw17\n(17 feat)', 'paper_11\n(12 feat)', 'paper_13\n(14 feat)', 'paper_13_contBMI\n(14 feat)']
feat_aucs_bar = [0.8967, 0.8497, 0.8544, 0.8914]
feat_f1s_bar = [0.8695, 0.8361, 0.8394, 0.8692]
colors_bar = [colors['primary'] if v >= 0.89 else colors['warning'] if v >= 0.85 else colors['danger'] for v in feat_aucs_bar]

x = np.arange(len(feat_sets))
w = 0.35
bars1 = ax.bar(x - w/2, feat_aucs_bar, w, label='AUROC', color=colors_bar, alpha=0.85)
bars2 = ax.bar(x + w/2, feat_f1s_bar, w, label='F1-Macro', color=[colors['secondary']]*4, alpha=0.6)
ax.axhline(y=0.9078, color='red', linestyle='--', linewidth=1.5, label='Paper AUROC')
ax.set_xticks(x)
ax.set_xticklabels(feat_sets, fontsize=9)
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Feature Set Comparison\n(RF, 5-fold CV, Best Model)', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.75, 0.95)
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2d: Gap to paper analysis
ax = axes[1, 1]
approaches = [
    'Phase 1\n(Arthritis)',
    'Step 02\nBaseline',
    'Step 08\nXGB Fix',
    'Step 09\nStacking',
    'Step 10\nPaper Repl.',
    'Best Overall\n(raw17 + RF)',
]
our_aucs = [0.7058, 0.8937, 0.8967, 0.8925, 0.8914, 0.8967]
gaps = [0.9078 - a for a in our_aucs]

colors_gap = [colors['danger'] if g > 0.05 else colors['warning'] if g > 0.02 else colors['accent'] for g in gaps]
bars = ax.bar(range(len(approaches)), gaps, color=colors_gap, alpha=0.85)
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{gap:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'({our_aucs[i]:.4f})', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_xticks(range(len(approaches)))
ax.set_xticklabels(approaches, fontsize=9)
ax.set_ylabel('Gap to Paper (AUROC)', fontweight='bold')
ax.set_title('Gap to Paper (0.9078)', fontweight='bold')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.01, color='green', linestyle=':', alpha=0.5, label='0.01 gap')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'fig2_detailed_comparisons.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig2_detailed_comparisons.png")

# =============================================================================
# FIGURE 3: Comprehensive Dashboard
# =============================================================================
fig = plt.figure(figsize=(18, 12))

# 3a: AUROC timeline
ax1 = fig.add_subplot(2, 3, 1)
steps_short = ['Phase\n1', 'S02', 'S03', 'S05\n-07', 'S08', 'S09', 'S10']
aurocs_timeline = [0.7058, 0.8937, 0.8652, 0.8924, 0.8967, 0.8925, 0.8914]
ax1.plot(steps_short, aurocs_timeline, 'o-', color=colors['primary'], linewidth=2, markersize=8, zorder=5)
ax1.axhline(y=0.9078, color='red', linestyle='--', linewidth=1.5, label='Paper (0.9078)')
ax1.fill_between(range(len(steps_short)), 0.88, 0.91, alpha=0.1, color=colors['primary'])
for i, auc in enumerate(aurocs_timeline):
    ax1.annotate(f'{auc:.4f}', (i, auc), textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=8, fontweight='bold')
ax1.set_ylabel('AUROC', fontweight='bold')
ax1.set_title('AUROC Timeline', fontweight='bold')
ax1.set_ylim(0.68, 0.93)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# 3b: Model radar for Step 09
ax2 = fig.add_subplot(2, 3, 2)
s09_names = ['RF\n(raw17)', 'XGB\nOptuna', 'LGBM\nOptuna', 'Stack\n(RF+XGB)', 'Stack\n(RF+XGB+LGBM)']
s09_auc = [0.8967, 0.8563, 0.8652, 0.8914, 0.8925]
s09_prauc = [0.7783, 0.7272, 0.7639, 0.7829, 0.7876]
s09_f1m = [0.8695, 0.8630, 0.8696, 0.8707, 0.8718]

x = np.arange(len(s09_names))
w = 0.25
ax2.bar(x - w, s09_auc, w, label='AUROC', color=colors['primary'], alpha=0.85)
ax2.bar(x, s09_prauc, w, label='PR-AUC', color=colors['accent'], alpha=0.85)
ax2.bar(x + w, s09_f1m, w, label='F1-Macro', color=colors['secondary'], alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(s09_names, fontsize=8)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Step 09: Advanced Pipeline', fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_ylim(0.65, 0.95)
ax2.grid(axis='y', alpha=0.3)

# 3c: Feature impact waterfall
ax3 = fig.add_subplot(2, 3, 3)
feat_names_short = list(feature_impact.keys())
feat_aucs_vals = list(feature_impact.values())
baseline = 0.8937
colors_water = [colors['accent'] if v >= baseline else colors['danger'] for v in feat_aucs_vals]
sorted_idx = np.argsort(feat_aucs_vals)[::-1]
ax3.barh(range(len(feat_names_short)), [feat_aucs_vals[i] for i in sorted_idx],
         color=[colors_water[i] for i in sorted_idx], alpha=0.85)
ax3.set_yticks(range(len(feat_names_short)))
ax3.set_yticklabels([feat_names_short[i] for i in sorted_idx], fontsize=8)
ax3.axvline(x=baseline, color=colors['primary'], linestyle=':', linewidth=1.5, label=f'raw17 ({baseline:.4f})')
ax3.set_xlabel('AUROC', fontweight='bold')
ax3.set_title('Feature Set Impact', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(axis='x', alpha=0.3)
for i, idx in enumerate(sorted_idx):
    ax3.text(feat_aucs_vals[idx] + 0.002, i, f'{feat_aucs_vals[idx]:.4f}', va='center', fontsize=7)

# 3d: BMI type comparison (Step 10 key finding)
ax4 = fig.add_subplot(2, 3, 4)
bmi_types = ['BMI_New\n(Categorical)', 'BMI\n(Continuous)']
rf_aucs = [0.8544, 0.8914]
xgb_aucs = [0.7677, 0.8603]
x = np.arange(len(bmi_types))
w = 0.3
ax4.bar(x - w/2, rf_aucs, w, label='RF', color=colors['primary'], alpha=0.85)
ax4.bar(x + w/2, xgb_aucs, w, label='XGB', color=colors['secondary'], alpha=0.85)
ax4.axhline(y=0.9078, color='red', linestyle='--', linewidth=1.5, label='Paper (0.9078)')
for i in range(len(bmi_types)):
    ax4.text(i - w/2, rf_aucs[i] + 0.005, f'{rf_aucs[i]:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax4.text(i + w/2, xgb_aucs[i] + 0.005, f'{xgb_aucs[i]:.4f}', ha='center', fontsize=9, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(bmi_types, fontsize=10)
ax4.set_ylabel('AUROC', fontweight='bold')
ax4.set_title('BMI Encoding Impact\n(Paper 13 Features, CV)', fontweight='bold')
ax4.legend(fontsize=8)
ax4.set_ylim(0.70, 0.95)
ax4.grid(axis='y', alpha=0.3)

# 3e: 70/30 vs CV comparison (Step 10)
ax5 = fig.add_subplot(2, 3, 5)
models_10 = ['RF\npaper_13\ncontBMI', 'XGB_opt\npaper_13\ncontBMI', 'RF\npaper_11', 'XGB_opt\npaper_11']
cv_aucs_10 = [0.8914, 0.8603, 0.8497, None]
split_aucs_10 = [0.8622, 0.8292, 0.8183, None]
x = np.arange(len(models_10))
w = 0.3
bars1 = ax5.bar(x - w/2, [v if v else 0 for v in cv_aucs_10], w, label='5-fold CV', color=colors['primary'], alpha=0.85)
bars2 = ax5.bar(x + w/2, [v if v else 0 for v in split_aucs_10], w, label='70/30 Split', color=colors['accent'], alpha=0.85)
ax5.axhline(y=0.9078, color='red', linestyle='--', linewidth=1.5, label='Paper (0.9078)')
for i, (cv, sp) in enumerate(zip(cv_aucs_10, split_aucs_10)):
    if cv: ax5.text(i - w/2, cv + 0.005, f'{cv:.4f}', ha='center', fontsize=8, fontweight='bold')
    if sp: ax5.text(i + w/2, sp + 0.005, f'{sp:.4f}', ha='center', fontsize=8, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(models_10, fontsize=8)
ax5.set_ylabel('AUROC', fontweight='bold')
ax5.set_title('Step 10: Evaluation Method Impact', fontweight='bold')
ax5.legend(fontsize=8)
ax5.set_ylim(0.70, 0.95)
ax5.grid(axis='y', alpha=0.3)

# 3f: Summary statistics
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_text = (
    "SUMMARY\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "Paper (Fu et al. 2025):  AUROC = 0.9078\n"
    "Our Best (RF, raw17):     AUROC = 0.8967\n"
    "Gap:                       0.0111\n\n"
    "Key Findings:\n"
    "• RF > XGBoost on this data (13.3% class)\n"
    "• Biomarkers & KDM-BA hurt performance\n"
    "• Continuous BMI > Categorical BMI_New\n"
    "  (0.8914 vs 0.8544, +0.037)\n"
    "• Clinical interactions (delta_BA,\n"
    "  CVD*KDM_BA) add noise, not signal\n"
    "• Stacking PR-AUC: 0.7876 > RF alone: 0.7783\n"
    "• Paper gap likely due to:\n"
    "  - Different data (9,505 vs 12,329)\n"
    "  - Different class ratio (10.5% vs 13.3%)\n"
    "  - KDM-BA SHAP=0.6 vs our BA SHAP=0.44"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'fig3_comprehensive_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig3_comprehensive_dashboard.png")

# =============================================================================
# FIGURE 4: Waterfall - What Helped / What Hurt
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

interventions = [
    'Phase 1→2:\nKOA target', '+0.190',
    'Step 03:\nBiomarkers', '-0.167',
    'Step 03:\nKDM-BA', '-0.029',
    'Step 08:\nOHE + Youden', '+0.000',
    'Step 09:\nOptuna XGB', '-0.040',
    'Step 09:\nStacking', '-0.004',
    'Step 09:\nInteractions', '-0.079',
    'Step 10:\nPaper features', '0.000',
    'Step 10:\nContinuous BMI', '+0.037',
]

labels = [interventions[i] for i in range(0, len(interventions), 2)]
values = [float(interventions[i]) for i in range(1, len(interventions), 2)]

colors_int = [colors['accent'] if v > 0 else colors['danger'] for v in values]

ax.barh(range(len(labels)), values, color=colors_int, alpha=0.85, height=0.6)
for i, (v, label) in enumerate(zip(values, labels)):
    ax.text(v + (0.005 if v >= 0 else -0.005), i,
            f'{v:+.3f}', ha='left' if v >= 0 else 'right',
            va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('AUROC Change', fontweight='bold')
ax.set_title('What Helped / What Hurt: Interventions and Their Impact on AUROC',
             fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'fig4_waterfall_interventions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: fig4_waterfall_interventions.png")

print("\nAll visualizations saved to:", VIS_DIR)
print("  - fig1_progression_and_comparisons.png")
print("  - fig2_detailed_comparisons.png")
print("  - fig3_comprehensive_dashboard.png")
print("  - fig4_waterfall_interventions.png")