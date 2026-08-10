import pandas as pd
import numpy as np

def synthesize_multilabel_targets(df):
    """
    Since the raw S3 dataset does not contain actual longitudinal S6 career choices, 
    we synthesize realistic Multi-Label ground truth based on typical academic 
    requirements for Rwandan health sector programs to serve as the training targets 
    for the Multi-Label Classifier Chains model.
    """
    print("Synthesizing Multi-Label targets...")
    
    # Extract normalized scores
    bio = df['Biology_Score_Normalized']
    chem = df['Chemistry_Score_Normalized']
    math = df['Math_Score_Normalized']
    phys = df['Physics_Score_Normalized']
    
    # 1. Medicine: Extremely high Bio & Chem, solid Math/Phys
    df['Target_Medicine'] = ((bio >= 0.75) & (chem >= 0.75) & (math >= 0.6) & (phys >= 0.6)).astype(int)
    
    # 2. Pharmacy: Extremely high Chem & Math, solid Bio
    df['Target_Pharmacy'] = ((chem >= 0.75) & (math >= 0.75) & (bio >= 0.6)).astype(int)
    
    # 3. Nursing: Solid Bio & Chem, moderate Math
    df['Target_Nursing'] = ((bio >= 0.65) & (chem >= 0.6) & (math >= 0.5)).astype(int)
    
    # 4. Public Health: Moderate across all core STEM
    df['Target_PublicHealth'] = ((bio >= 0.55) & (chem >= 0.55) & (math >= 0.5) & (phys >= 0.5)).astype(int)
    
    # 5. Biomedical Engineering: High Math & Phys, moderate Bio/Chem
    df['Target_BiomedicalEng'] = ((math >= 0.75) & (phys >= 0.7) & (bio >= 0.5) & (chem >= 0.5)).astype(int)

    return df

def main():
    input_file = '../DATA/cleaned_skill_profiles.csv'
    output_file = '../DATA/processed_multilabel_data.csv'
    
    print(f"Loading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Original shape: {df.shape}")
    
    features = ['Math_Score_Normalized', 'Physics_Score_Normalized', 
                'Chemistry_Score_Normalized', 'Biology_Score_Normalized']
    
    # Drop rows missing the core STEM features
    df = df.dropna(subset=features).copy()
    print(f"Shape after dropping NaNs in core features: {df.shape}")
    
    # Filter out rows where scores might be incorrectly scaled (e.g. > 1.0 or < 0.0)
    for f in features:
        df = df[(df[f] >= 0.0) & (df[f] <= 1.0)]
    print(f"Shape after ensuring strict 0-1 normalization bounds: {df.shape}")
    
    # Synthesize the multi-label ground truth
    df = synthesize_multilabel_targets(df)
    
    # Summary of label distribution
    targets = ['Target_Medicine', 'Target_Pharmacy', 'Target_Nursing', 
               'Target_PublicHealth', 'Target_BiomedicalEng']
    
    print("\nTarget Label Distribution (Positive Cases):")
    for t in targets:
        positives = df[t].sum()
        percentage = (positives / len(df)) * 100
        print(f"  {t}: {positives} ({percentage:.2f}%)")
        
    # Check for overlapping labels (Multi-Label nature)
    df['Total_Labels'] = df[targets].sum(axis=1)
    print("\nMulti-Label Overlap Distribution:")
    print(df['Total_Labels'].value_counts().sort_index())
    
    # Save the processed dataset
    print(f"\nSaving processed multi-label dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Preprocessing Complete!")

if __name__ == "__main__":
    main()
