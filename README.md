Credit Card Fraud Detection
Project Objective
The goal of this project is to develop a machine learning model that can accurately detect fraudulent credit card transactions and help prevent financial fraud.

Dataset Information
Dataset: creditcard.csv

Shape: (284807 rows × 31 columns)

Missing Values: None

Class Distribution:

0 → Normal Transactions

1 → Fraudulent Transactions (only 492 instances)

Project Workflow
Data Loading: Loaded the dataset using the Pandas library.

Class Imbalance Handling: Handled the class imbalance using techniques like oversampling or undersampling.

Model Training: Trained a classification model to predict frauds.

Model Evaluation: Evaluated the model’s performance using:

Accuracy Score

Confusion Matrix

Classification Report

ROC AUC Score

Results
Accuracy: 93%

ROC AUC Score: 93%

The model performed well even with imbalanced data, accurately detecting fraudulent transactions.

Technologies Used
Python:

Jupyter Notebook

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

imbalanced-learn

How to Run
Install Dependencies:
Install the required libraries by running:
bash
Copy
Edit
pip install -r requirements.txt

Run the Notebook:
Open the fraud_detection.ipynb Jupyter notebook and run the cells step-by-step.

Analyze Results:
Evaluate the model’s performance based on the provided metrics.

Thank you for visiting!
