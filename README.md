# Loan Approval Predictor

The Loan Approval Predictor is a machine learning project designed to predict whether a loan application will be approved based on borrower and loan characteristics. By leveraging historical loan data, this project helps financial institutions make informed lending decisions.

## Project Description

This project was motivated by the need to automate and improve loan approval processes in financial institutions. By predicting loan approval outcomes, it addresses the challenge of assessing credit risk efficiently and accurately. The project uses a dataset (`train.csv`) with features like borrower age, income, loan amount, and credit history to train machine learning models.

**Why this project?**
- **Problem Solved**: Reduces manual effort in loan assessments and minimizes approval errors by predicting outcomes based on data-driven insights.
- **Technologies Used**: Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, and imblearn were chosen for their robust data processing, modeling, and visualization capabilities.
- **Challenges Faced**: Handling skewed numerical features and class imbalance in the dataset required advanced preprocessing techniques like PowerTransformer and SMOTE.
- **Future Features**: Adding a web interface for user input, and exploring neural network models.

**What I Learned**:
- Advanced feature engineering techniques, such as creating interaction terms like `int_rate_x_percent_income`.
- Balancing precision and recall in classification tasks for financial applications.
- The importance of clear documentation for reproducibility and collaboration.

## Installation

To make predictions using pretrained model:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/loan-approval-predictor.git
   cd loan-approval-predictor
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed. Install required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
   
## How to Use
  **Make Predictions with the Saved Model**:
   Use the pre-trained model for new loan applications:
   ```bash
    python predict.py --input_csv path/to/applications.csv --output_csv path/to/predictions.csv \
        [--model_path path/to/model.pkl] [--preprocessor_path path/to/preprocessor.pkl]
   ```
**Screenshots**:
- *Feature Importance Plot*: See `notebooks/loan_approval_prediction.ipynb` for a plot showing key predictors like `loan_int_rate` and `loan_grade`.
- *ROC Curve*: Visualizes model performance in distinguishing approved vs. denied loans.  
