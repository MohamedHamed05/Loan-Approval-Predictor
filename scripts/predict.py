import argparse
import pandas as pd
import pickle
import sys

def load_artifacts(model_path, preprocessor_path):
    """Load the trained model and preprocessor."""
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

def preprocess(df: pd.DataFrame):
    """Apply feature mapping and engineering consistent with training."""
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    df['loan_grade'] = df['loan_grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    df['int_rate_x_percent_income'] = df['loan_int_rate'] * df['loan_percent_income']
    return df

def main():
    parser = argparse.ArgumentParser(description="Predict loan approval status")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV file with loan applications")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV for predictions")
    parser.add_argument('--model_path', type=str, default='models/model.pkl',help='Path to the trained model pickle file')
    parser.add_argument('--preprocessor_path', type=str, default='models/preprocessor.pkl',help='Path to the preprocessor pickle file')
    args = parser.parse_args()

    # Load input data
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}", file=sys.stderr)
        sys.exit(1)
        
    df = preprocess(df)
    pred_df = df[['id']]
    df.drop(columns=['id'], inplace=True)

    # Load model and preprocessor
    preprocessor, model = load_artifacts(args.model_path, args.preprocessor_path)

    # Transform features and predict
    X_transformed = preprocessor.transform(df)
    predictions = model.predict(X_transformed)

    # Save predictions
    pred_df['loan_status'] = predictions
    pred_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()