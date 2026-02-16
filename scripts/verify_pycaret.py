import pandas as pd
from pycaret.classification import setup, compare_models, get_config
from pathlib import Path
import os

def verify_pycaret():
    print("Starting PyCaret verification...")
    
    # Define paths
    data_path = Path('data/processed/ravdess_mfcc_features.csv')
    output_dir = Path('data/processed')
    
    # Check if data exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Filter for MFCC columns and emotion
    feature_cols = [c for c in df.columns if c.startswith(('mfcc', 'delta_mfcc'))]
    if 'emotion' not in df.columns:
        raise KeyError("Expected target column 'emotion' in MFCC feature table")
        
    df_prepped = df[feature_cols + ['emotion']].copy()
    print(f"Prepared dataframe with {len(feature_cols)} features + emotion target")
    
    # Setup PyCaret
    print("Initializing PyCaret setup...")
    clf1 = setup(
        data=df_prepped,
        target='emotion',
        session_id=42,
        fold=5,
        train_size=0.7,
        normalize=True,
        transformation=False,
        fix_imbalance=False,
        feature_selection=False,
        remove_multicollinearity=False,
        low_variance_threshold=None,
        use_gpu=False,
        verbose=False, # Reduce output for script
        html=False # Disable HTML output for script
    )
    print("PyCaret setup complete.")
    
    # Compare models (turbo=True for speed)
    print("Running compare_models (turbo=True)...")
    best_model = compare_models(turbo=True, n_select=1, verbose=False)
    print(f"Best model found: {best_model}")
    
    # Export train/test splits
    print("Exporting train/test splits...")
    X_train = get_config('X_train')
    X_test = get_config('X_test')
    y_train = get_config('y_train')
    y_test = get_config('y_test')
    
    # Combine X and y for export
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_path = output_dir / 'ravdess_mfcc_train_for_dl.csv'
    test_path = output_dir / 'ravdess_mfcc_test_for_dl.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train split saved to {train_path}")
    print(f"Test split saved to {test_path}")
    print("Verification successful!")

if __name__ == "__main__":
    # Ensure we are in the project root
    # Assuming the script is run from project root as 'python scripts/verify_pycaret.py'
    # If run from scripts dir, adjust path
    if not Path('data').exists():
        # Try to move up one level if data not found
        os.chdir('..')
        
    if not Path('data').exists():
         raise FileNotFoundError("Could not find 'data' directory. Please run from project root.")

    verify_pycaret()
