import pandas as pd
import numpy as np
import joblib
import itertools

def generate_counterfactual(model, original_x, target_index=0, feature_names=None):
    """
    A simplified gradient-free counterfactual generator (L2 optimization).
    It searches for the minimal positive perturbation to the original_x
    that flips the prediction of the target class to 1.
    """
    # Verify if it's already accepted
    pred = model.predict([original_x])[0]
    if pred[target_index] == 1:
        return "Already accepted for this target. No counterfactual needed."
        
    print(f"\nOriginal Profile: {original_x}")
    print(f"Original Prediction (All targets): {pred}")
    print("Generating counterfactual... (Searching for minimal positive improvements)")
    
    # We only allow positive improvements (students can only study harder, not time travel to score less)
    # We create a grid of possible improvements in steps of 0.05 (5%)
    step = 0.05
    max_steps = 10 # Up to 50% improvement per subject
    
    best_cf = None
    min_distance = float('inf')
    
    # Generate all combinations of steps for 4 features
    step_ranges = [range(max_steps + 1) for _ in range(4)]
    
    for steps in itertools.product(*step_ranges):
        # Skip no change
        if sum(steps) == 0:
            continue
            
        perturbation = np.array(steps) * step
        candidate_x = original_x + perturbation
        
        # Clip to 1.0 (max score)
        candidate_x = np.clip(candidate_x, 0, 1.0)
        
        # L2 Distance
        distance = np.sqrt(np.sum((candidate_x - original_x)**2))
        
        # If this distance is already worse than our best found, skip evaluating the model (optimization)
        if distance >= min_distance:
            continue
            
        # Check model prediction
        candidate_pred = model.predict([candidate_x])[0]
        
        if candidate_pred[target_index] == 1:
            best_cf = candidate_x
            min_distance = distance
            
    if best_cf is not None:
        print(f"Found Counterfactual! L2 Distance: {min_distance:.4f}")
        print(f"Required Profile: {best_cf}")
        
        # Calculate exact changes
        changes = best_cf - original_x
        print("\nActionable Advice:")
        for i, change in enumerate(changes):
            if change > 0.001:
                name = feature_names[i] if feature_names else f"Feature {i}"
                print(f"- Improve {name} by {change*100:.1f}%")
    else:
        print("Could not find a valid counterfactual within realistic bounds.")

def main():
    model_path = '../DATA/best_cc_model.pkl'
    data_path = '../DATA/processed_multilabel_data.csv'
    
    print(f"Loading best model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Model file not found. Run 02_modeling.py first.")
        return
        
    print(f"Loading a sample from dataset...")
    df = pd.read_csv(data_path).sample(1000, random_state=42)
    
    features = ['Math_Score_Normalized', 'Physics_Score_Normalized', 
                'Chemistry_Score_Normalized', 'Biology_Score_Normalized']
                
    # Find a student who was rejected for Medicine
    # Medicine is index 0 in the targets list from 02_modeling.py
    # Let's find someone with decent scores who barely missed it
    
    student = df[(df['Target_Medicine'] == 0) & (df['Biology_Score_Normalized'] > 0.6) & (df['Chemistry_Score_Normalized'] > 0.6)].iloc[0]
    
    original_x = student[features].values.astype(float)
    
    generate_counterfactual(model, original_x, target_index=0, feature_names=features)
    
if __name__ == "__main__":
    main()
