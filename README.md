# Loan Approval Predictor

The Loan Approval Predictor is a machine learning project designed to predict whether a loan application will be approved based on borrower and loan characteristics. By using historical loan data, this project helps financial institutions make informed lending decisions.

## Project Description

This project was motivated by the need to automate and improve loan approval processes in financial institutions. By predicting loan approval outcomes, it addresses the challenge of assessing credit risk efficiently and accurately. The project uses a dataset (`train.csv`) with features like borrower age, income, loan amount, and credit history to train machine learning models.

**Why this project?**
- **Problem Solved**: Reduces manual effort in loan assessments and minimizes approval errors by predicting outcomes based on data-driven insights.
- **Technologies Used**: Python, pandas, scikit-learn, XGBoost, matplotlib, and seaborn were chosen for their robust data processing, modeling, and visualization capabilities.
- **Challenges Faced**: Handling skewed numerical features and class imbalance in the dataset required advanced preprocessing techniques like PowerTransformer and SMOTE.
- **Future Features**: Adding a web interface for user input, and exploring neural network models.

**What I Learned**:
- Effective techniques for preprocessing financial datasets, including outlier capping and feature transformation to handle non-normal distributions and extreme outliers.
- The importance of model selection and hyperparameter tuning, particularly in optimizing XGBoost for high AUC-ROC and balanced F1-score.
- How to interpret and visualize feature importance to identify key drivers of loan approval, such as interest rate and loan-to-income ratio.
- Strategies for addressing class imbalance like SMOTE, improving model performance on minority classes.
- The value of modular code design and documentation in creating reproducible machine learning pipelines.

## Model Performance

The project evaluates seven machine learning models, with performance metrics on the test set summarized below:

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.8041   | 0.4086    | 0.8407 | 0.5499   | 0.8937  |
| Naive Bayes          | 0.8011   | 0.3970    | 0.7653 | 0.5228   | 0.8692  |
| K-Nearest Neighbors  | 0.9234   | 0.8379    | 0.5725 | 0.6802   | 0.8678  |
| Decision Tree        | 0.9150   | 0.6999    | 0.7054 | 0.7027   | 0.8276  |
| Random Forest        | 0.9513   | 0.9385    | 0.7042 | 0.8047   | 0.9340  |
| SVM                  | 0.8990   | 0.6097    | 0.8072 | 0.6947   | 0.0000  |
| XGBoost              | 0.9504   | 0.9029    | 0.7399 | 0.8073   | 0.9561  |

XGBoost was selected as the final model due to its superior F1-score and AUC-ROC

## Installation

To make predictions using the pretrained model:

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
   ```bash
    python predict.py --input_csv path/to/applications.csv --output_csv path/to/predictions.csv \
        [--model_path path/to/model.pkl] [--preprocessor_path path/to/preprocessor.pkl]
   ```
