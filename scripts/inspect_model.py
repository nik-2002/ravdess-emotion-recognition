import joblib
import sys

MODEL_PATH = "models/ravdess_mfcc_pycaret_best.pkl"

try:
    print(f"Loading model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    print("\nModel loaded successfully.")
    print(f"Type: {type(pipeline)}")
    
    if hasattr(pipeline, 'steps'):
        print("\nPipeline Steps:")
        for name, step in pipeline.steps:
            print(f" - {name}: {type(step)}")
            if name == 'trained_model':
                print(f"   -> ESTIMATOR: {step}")
    else:
        print("\nNot a standard Pipeline object. Inspecting directly:")
        print(pipeline)

except Exception as e:
    print(f"\nError loading model: {e}")
