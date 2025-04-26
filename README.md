Credit Card Fraud Detection Project
Project Objective:

To develop a machine learning model that can accurately detect fraudulent credit card transactions and help prevent financial fraud. The model identifies fraudulent transactions from a dataset with various features, including transaction amount, merchant details, and timestamps.

Dataset Information:
Dataset: creditcard.csv

Shape: (284807 rows × 31 columns)

Missing Values: None

Class Distribution:

0 → Normal Transactions

1 → Fraudulent Transactions (only 492)

Project Workflow

Data Exploration: Loaded the dataset and explored its shape and missing values.

Preprocessing: Standardized the Amount and Time columns using StandardScaler. Dropped the original Amount and Time columns to avoid redundancy.

Handling Class Imbalance: Used undersampling to balance the classes (fraudulent vs non-fraudulent transactions). Sampled the majority class (non-fraudulent transactions) to match the number of fraudulent transactions, ensuring a balanced dataset for training.

Model Building: Used the RandomForestClassifier to train the model on the balanced dataset.

Model Evaluation: Evaluated the model's performance using:
Accuracy Score
Confusion Matrix
Classification Report
ROC AUC Score

Visualization: Displayed the confusion matrix as a heatmap to visualize the true positives, false positives, true negatives, and false negatives.

Model Saving: Saved the trained model using joblib for future use without retraining.

Results:

Accuracy: 93%
ROC AUC Score: 93%
Model: Performed well despite the class imbalance and effectively detected fraudulent transactions.

Technologies Used:
Python
Jupyter Notebook
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn
imbalanced-learn
joblib (for saving the model)

How to Run:
Install the required libraries by running:
pip install -r requirements.txt

Download the dataset (creditcard.csv) and place it in the appropriate directory.

Open the Jupyter Notebook or Python file and run each step sequentially.

Analyze model performance based on the evaluation metrics (accuracy, ROC AUC, etc.).

Use the saved model (fraud_detection_model.pkl) for future predictions.

Thank you for visiting!
