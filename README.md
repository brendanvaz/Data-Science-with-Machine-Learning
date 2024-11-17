Exploratory Evaluation of Water Pumps in Tanzania
Overview
This group project analyzes the operational status of water pumps in Tanzania using machine learning techniques. The study was led by Brendan Ezekiel Agnelo Vaz, with team members collaboratively working on different aspects of the project. Leveraging the "Pump it Up: Data Mining the Water Table" dataset from the Taarifa waterpoints dashboard, this analysis provides actionable insights into pump functionality, maintenance needs, and factors influencing failures. The project evaluates multiple machine learning models to identify the best performer for predicting water pump functionality.

Objectives
To predict the functionality of water pumps (functional, functional but needs repair, non-functional) using machine learning models.
To analyze key factors influencing water pump performance.
To recommend data-driven strategies for maintenance and resource allocation.
Dataset
Source: Pump it Up: Data Mining the Water Table
Description: This dataset includes 38 features related to water pump functionality, such as:
Pump type
Management authority
Geolocation
Installation date
Water quality and quantity
Methodology
Leadership Role:

As the project leader, I (Brendan Ezekiel Agnelo Vaz) coordinated all aspects of the project, from defining objectives to final delivery. I facilitated team collaboration, assigned responsibilities, and ensured deadlines were met.
Personally oversaw critical components such as data preprocessing, model evaluation, and documentation.
Data Preprocessing

Imputed missing values using MICE (Multivariate Imputation by Chained Equations) for numerical features.
Removed redundant or highly correlated columns to improve model performance.
Encoded categorical variables using one-hot encoding.
Feature Engineering

Created derived features like pump age to improve predictive performance.
Model Training

Tested the following machine learning models:
Decision Trees
Random Forest
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Multilayer Perceptron (MLP)
Gaussian Naive Bayes (GNB)
Tuned hyperparameters using Bayesian Optimization via the Optuna library.
Model Evaluation

Metrics used: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
Visualizations: Confusion matrices and ROC curves for key models.
Results
Best Performing Model: Random Forest
Accuracy: 79%
Precision (weighted): 79%
Recall (weighted): 80%
Key insights:
Geographic location, pump age, and management practices are significant predictors.
Random Forests outperformed other models due to their ability to handle complex, multi-class classification tasks.
Technologies Used
Programming Language: Python
Libraries:
Pandas
NumPy
scikit-learn
LightGBM
Matplotlib
Seaborn
Optuna
How to Run
Prerequisites:
Python 3.x installed
Required libraries installed via:
bash
Copy code
pip install -r requirements.txt
Steps:
Clone the repository:
bash
Copy code
git clone https://github.com/brendanvaz/EXPLORATORY-EVALUATION-OF-WATER-PUMPS-IN-TANZANIA
Navigate to the project folder:
bash
Copy code
cd EXPLORATORY-EVALUATION-OF-WATER-PUMPS-IN-TANZANIA
Open and execute the notebook:
bash
Copy code
jupyter notebook "Pump it Up Study Enigma (1).ipynb"
Future Work
Explore more advanced models like Gradient Boosting and Deep Neural Networks.
Integrate external environmental and socio-economic factors to enhance model accuracy.
Implement real-time monitoring systems for predictive maintenance.
Authors
Brendan Ezekiel Agnelo Vaz (Project Leader) - MSc Data Science, University of Nottingham
Joel Jacob Thomas - MSc Cyber Physical Systems, University of Nottingham
Matthieu Blackler - MSc Computer Science, University of Nottingham
