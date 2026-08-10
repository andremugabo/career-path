import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from xgboost import XGBClassifier
import joblib

def evaluate_model(y_true, y_pred, model_name):
    # Subset Accuracy: strict exact match of all labels
    subset_acc = accuracy_score(y_true, y_pred)
    # Hamming Loss: fraction of incorrectly predicted labels
    hl = hamming_loss(y_true, y_pred)
    # Micro F1: aggregates contributions of all classes
    f1 = f1_score(y_true, y_pred, average='micro')
    
    print(f"\n--- {model_name} Results ---")
    print(f"Subset Accuracy : {subset_acc:.4f}")
    print(f"Hamming Loss    : {hl:.4f}")
    print(f"Micro F1-Score  : {f1:.4f}")
    
    return {'Model': model_name, 'Subset_Accuracy': subset_acc, 'Hamming_Loss': hl, 'Micro_F1': f1}

def main():
    input_file = '../DATA/processed_multilabel_data.csv'
    
    print(f"Loading dataset from {input_file}...")
    try:
        # Subsample to 5,000 rows to speed up training for the proof-of-concept
        df = pd.read_csv(input_file).sample(5000, random_state=42)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Run 01_preprocessing.py first.")
        return
        
    # Define features and targets
    features = ['Math_Score_Normalized', 'Physics_Score_Normalized', 
                'Chemistry_Score_Normalized', 'Biology_Score_Normalized']
    
    targets = ['Target_Medicine', 'Target_Pharmacy', 'Target_Nursing', 
               'Target_PublicHealth', 'Target_BiomedicalEng']
               
    X = df[features].values
    Y = df[targets].values
    
    # Stratification in multi-label is complex, we will use a simple random split for this POC
    print("Splitting dataset into 80% train, 20% test...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples\n")
    
    # Define the base classifiers
    base_models = {
        'Decision Tree (DT)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest (RF)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'XGBoost (XGB)': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        'Multi-Layer Perceptron (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }
    
    results = []
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    for name, base_clf in base_models.items():
        print(f"Training Classifier Chain with {name}...")
        start_time = time.time()
        
        # Initialize Classifier Chain. order='random' could be used, or a specific order.
        # We will use the default natural order [0, 1, ..., K-1] which maps to [Medicine, Pharmacy, ...]
        chain = ClassifierChain(base_estimator=base_clf, order='random', random_state=42)
        
        try:
            chain.fit(X_train, Y_train)
            Y_pred = chain.predict(X_test)
            
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f} seconds.")
            
            res = evaluate_model(Y_test, Y_pred, name)
            results.append(res)
            
            # Keep track of the best model to save for counterfactual generation
            if res['Micro_F1'] > best_f1:
                best_f1 = res['Micro_F1']
                best_model = chain
                best_model_name = name
                
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    # Save the best model
    if best_model is not None:
        model_path = '../DATA/best_cc_model.pkl'
        print(f"\nSaving the best model ({best_model_name}) to {model_path}...")
        joblib.dump(best_model, model_path)
        
    # Output final summary
    print("\n=== Final Model Comparison ===")
    results_df = pd.DataFrame(results).sort_values(by='Micro_F1', ascending=False)
    print(results_df.to_string(index=False))
    
if __name__ == "__main__":
    main()
