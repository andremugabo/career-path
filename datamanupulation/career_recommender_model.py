import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style="whitegrid")
try:
    df = pd.read_csv('../DATA/cleaned_skill_profiles.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Dataset not found. Please ensure you have run the data_cleaning_pipeline.ipynb first.")
    # For demonstration purposes, creating a mock dataset if the file isn't there yet
    np.random.seed(42)
    mock_data = {
        'Student_Number': range(1, 1001),
        'Math_Score_Normalized': np.random.beta(5, 2, 1000),
        'Physics_Score_Normalized': np.random.beta(4, 2, 1000),
        'Chemistry_Score_Normalized': np.random.beta(2, 5, 1000),
        'Biology_Score_Normalized': np.random.beta(2, 5, 1000)
    }
    # Injecting specific profiles
    # STEM Heavy
    mock_data['Math_Score_Normalized'][:300] = np.random.uniform(0.7, 1.0, 300)
    mock_data['Physics_Score_Normalized'][:300] = np.random.uniform(0.7, 1.0, 300)
    # Bio/Chem Heavy
    mock_data['Biology_Score_Normalized'][300:600] = np.random.uniform(0.7, 1.0, 300)
    mock_data['Chemistry_Score_Normalized'][300:600] = np.random.uniform(0.7, 1.0, 300)
    
    df = pd.DataFrame(mock_data)
    print(f"Using generated Mock Data for demonstration. Shape: {df.shape}")

features = ['Math_Score_Normalized', 'Physics_Score_Normalized', 'Chemistry_Score_Normalized', 'Biology_Score_Normalized']
# K-Means requires no NaNs, so we drop rows with missing essential scores
ml_df = df.dropna(subset=features).copy()
X = ml_df[features]

inertia = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
OPTIMAL_K = 4 # Adjust this based on the elbow plot above
final_kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42)
ml_df['Aptitude_Cluster'] = final_kmeans.fit_predict(X)

# Calculate the mean scores for each cluster to profile them
cluster_profiles = ml_df.groupby('Aptitude_Cluster')[features].mean()
print("Cluster Centroids (Average Normalized Scores):")
display(cluster_profiles)

# Visualize the profiles
cluster_profiles.T.plot(kind='bar', figsize=(10, 6))
plt.title('Aptitude Profiles by Cluster')
plt.ylabel('Average Normalized Score')
plt.xticks(rotation=45)
plt.legend(title='Cluster ID')
plt.tight_layout()
plt.show()
# EXAMINE THE CHART ABOVE TO DEFINE THESE RULES
career_mapping = {
    0: "Life Sciences & Healthcare (e.g., Medicine, Nursing, Biotech)", # Assumes high Bio/Chem
    1: "Engineering & Technology (e.g., Computer Science, Civil Engineering)", # Assumes high Math/Physics
    2: "Generalist / Administration (e.g., Business, Law, Humanities)", # Assumes average scores across the board
    3: "Technical Vocational (TVET) / Specialized Trades" # Assumes lower academic scores but high potential in vocational tracks
}

ml_df['Recommended_Career_Path'] = ml_df['Aptitude_Cluster'].map(career_mapping)
print("Sample of Recommendations:")
display(ml_df[['Student_Number', 'Aptitude_Cluster', 'Recommended_Career_Path']].head(10))
def recommend_career_for_student(math, physics, chemistry, biology, model=final_kmeans, mapping=career_mapping):
    # Assuming scores are provided as percentages (0-100)
    # Normalize to 0-1 as the model expects
    normalized_scores = np.array([[math/100, physics/100, chemistry/100, biology/100]])
    
    predicted_cluster = model.predict(normalized_scores)[0]
    recommendation = mapping.get(predicted_cluster, "Unknown")
    
    print(f"--- Recommendation Report ---")
    print(f"Scores: Math={math}, Phys={physics}, Chem={chemistry}, Bio={biology}")
    print(f"Assigned Cluster: {predicted_cluster}")
    print(f"Recommended Path: {recommendation}")
    print("-----------------------------")
    return recommendation

# Test the function
recommend_career_for_student(85, 90, 60, 50) # Should hit the Engineering cluster
recommend_career_for_student(50, 45, 88, 92) # Should hit the Life Sciences cluster
