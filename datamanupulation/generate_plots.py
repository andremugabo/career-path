import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create figures directory if it doesn't exist
fig_dir = '../THESIS/figures/'
os.makedirs(fig_dir, exist_ok=True)

# 1. Label Co-occurrence Heatmap
print("Generating Label Co-occurrence Heatmap...")
df = pd.read_csv('../DATA/processed_multilabel_data.csv')
targets = ['Target_Medicine', 'Target_Pharmacy', 'Target_Nursing', 
           'Target_PublicHealth', 'Target_BiomedicalEng']

co_occurence = df[targets].T.dot(df[targets])
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurence, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Multi-Label Co-occurrence Matrix')
plt.xlabel('Target Paths')
plt.ylabel('Target Paths')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'label_cooccurrence.pdf'))
plt.close()

# 2. XGBoost Feature Importance
print("Generating Feature Importance Plot...")
# Since we used ClassifierChain, we can extract feature importances from the first base estimator (Target_Medicine)
model = joblib.load('../DATA/best_cc_model.pkl')
# The ClassifierChain stores its estimators in .estimators_
base_xgb = model.estimators_[0]

features = ['Math', 'Physics', 'Chemistry', 'Biology']
# Note: XGBoost uses its own internal naming if not given pandas df, but we know the order
importances = base_xgb.feature_importances_

plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=importances, palette='viridis')
plt.title('XGBoost Feature Importance (Medicine Track - Base Estimator)')
plt.ylabel('Relative Importance')
plt.xlabel('STEM Subjects')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'feature_importance.pdf'))
plt.close()

# 3. Target Distribution Bar Chart
print("Generating Target Distribution...")
sums = df[targets].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=sums.values, y=sums.index, palette='magma')
plt.title('Distribution of Positive Labels across Health Sectors')
plt.xlabel('Number of Qualifying Students')
plt.ylabel('Health Sector Target')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'target_distribution.pdf'))
plt.close()

print("Plots generated successfully in ../THESIS/figures/")

# 4. STEM Correlation Matrix
print("Generating STEM Correlation Matrix...")
features_only = df[['Math_Score_Normalized', 'Physics_Score_Normalized', 'Chemistry_Score_Normalized', 'Biology_Score_Normalized']]
corr = features_only.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation between STEM Subjects')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'stem_correlation.pdf'))
plt.close()

# 5. Boxplot of STEM Scores
print("Generating STEM Boxplots...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=features_only, palette='pastel')
plt.title('Distribution of Normalized STEM Scores')
plt.ylabel('Normalized Score [0, 1]')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'stem_boxplot.pdf'))
plt.close()

# 6. Model Performance Comparison
print("Generating Model Performance Bar Chart...")
models = ['XGBoost', 'Decision Tree', 'Random Forest', 'MLP']
subset_acc = [0.9980, 0.9970, 0.9970, 0.9830]
micro_f1 = [0.9859, 0.9787, 0.9784, 0.8321]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, subset_acc, width, label='Subset Accuracy', color='#2ca02c')
rects2 = ax.bar(x + width/2, micro_f1, width, label='Micro F1-Score', color='#1f77b4')

ax.set_ylabel('Scores')
ax.set_title('Classifier Chain Performance by Algorithm')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower left')
ax.set_ylim([0.8, 1.05]) # Focus on the high performance differences

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'model_performance.pdf'))
plt.close()

print("Extra plots generated successfully!")
