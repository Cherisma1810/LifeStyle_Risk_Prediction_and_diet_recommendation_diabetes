# LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes
An AI-powered healthcare application that predicts diabetes risk using an Artificial Neural Network (ANN) and generates personalized diet and lifestyle recommendations using Fuzzy Logic. The system combines machine learning with expert knowledge-based reasoning to provide intelligent healthcare assistance.

📌 Project Overview

This project is a hybrid intelligent healthcare system developed using:

Artificial Neural Network (ANN) for diabetes risk prediction
Fuzzy Logic System for personalized diet and lifestyle recommendations
Data Visualization for performance analysis and medical insights

The model is trained using the famous PIMA Indians Diabetes Dataset and achieves high predictive performance through preprocessing, feature engineering, and ensemble-based ANN training.

The project not only predicts whether a person is likely to have diabetes but also provides customized health recommendations based on:

Diabetes risk level
Physical activity level
Sugar intake level
🚀 Features
Diabetes prediction using Deep Learning
Personalized nutrition and exercise recommendation
Hybrid ANN + Fuzzy Logic architecture
5-Fold Cross Validation training
Automatic threshold optimization
Feature engineering and preprocessing
Detailed performance metrics
Multiple medical and analytical graphs
Sample patient risk analysis
Explainable fuzzy rule-based system
🧠 Technologies Used
Technology	Purpose
Python	Core Programming
TensorFlow / Keras	ANN Model
Scikit-learn	Preprocessing & Metrics
NumPy & Pandas	Data Handling
Matplotlib	Data Visualization
Scikit-Fuzzy	Fuzzy Logic System
📂 Dataset

Dataset used: PIMA Indians Diabetes Dataset

Source: Kaggle Dataset
File Required: diabetes.csv

Place the dataset in the same directory as the Python file or update the dataset path manually in the code.

⚙️ System Architecture
1️⃣ Data Preprocessing
Missing value handling
Median imputation
Feature scaling using StandardScaler
Derived feature generation:
Physical Activity
Diet Score
Stress Level
2️⃣ Artificial Neural Network

The ANN model contains:

Multiple Dense layers
Batch Normalization
Dropout regularization
Adam optimizer
Binary Cross Entropy loss
3️⃣ Fuzzy Logic Recommendation System

The fuzzy system uses:

27 expert-defined rules
Membership functions
Centroid defuzzification

It generates recommendations such as:

Strict diet plans
Moderate lifestyle modifications
Maintenance health routines
📊 Generated Graphs

The system automatically generates:

Confusion Matrix
ROC Curve
Performance Metrics Graph
Training vs Validation Accuracy
Training vs Validation Loss
Feature Importance Graph
Fuzzy Membership Functions
Risk Distribution Analysis
Sample Patient Radar Chart

All graphs are stored inside the graphs/ folder.

📁 Project Structure
LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes/
│
├── diabetes_system.py
├── diabetes.csv
├── graphs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── performance_metrics.png
│   ├── training_history.png
│   ├── feature_importance.png
│   ├── fuzzy_membership_functions.png
│   ├── risk_distribution.png
│   └── sample_patient_radar.png
│
└── README.md
▶️ Installation & Execution
Step 1: Clone Repository
git clone https://github.com/Cherisma1810/LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes.git
cd LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes
Step 2: Install Required Libraries
pip install numpy pandas matplotlib scikit-learn tensorflow keras scikit-fuzzy
Step 3: Download Dataset

Download:
diabetes.csv

From:
PIMA Indians Diabetes Dataset

Step 4: Update Dataset Path

Modify:

DATASET_PATH = "path_to_diabetes.csv"
Step 5: Run the Project
python diabetes_system.py
📈 Model Evaluation Metrics

The project evaluates:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score

It also prints:

Classification Report
Confusion Matrix
🥗 Sample Recommendations

Depending on the patient risk level, the system may recommend:

STRICT
Eliminate refined sugars
Daily aerobic exercise
Regular glucose monitoring
MODERATE
Balanced diet
Reduce processed food intake
Regular walking routine
MAINTAIN
Continue healthy habits
Annual health screening
Maintain BMI and hydration
🔬 Key Concepts Implemented
Deep Learning
Artificial Neural Networks
Ensemble Learning
Fuzzy Inference System
Membership Functions
Medical Data Analysis
Healthcare Recommendation Systems
Data Visualization
🎯 Future Enhancements
Web application integration
Real-time patient monitoring
Mobile healthcare assistant
IoT health sensor integration
Multi-disease prediction
Explainable AI dashboard
👩‍💻 Author

Developed by Cherisma Anamala
B.Tech Student | Artificial Intelligence & Machine Learning Enthusiast

GitHub Repository:
LifeStyle Risk Prediction and Diet Recommendation Diabetes Project

📜 License

This project is developed for educational and research purposes.

humanize and the file name is diabetes.py
Intelligent Diabetes Risk Prediction and Personalized Nutrition System

This project is an intelligent healthcare system that predicts the possibility of diabetes using Artificial Neural Networks (ANN) and provides personalized diet and lifestyle recommendations using Fuzzy Logic. The main objective of the project is to combine machine learning with intelligent decision-making techniques to help users understand their diabetes risk and follow healthier lifestyle habits.

The system is trained using the PIMA Indians Diabetes Dataset and performs data preprocessing, feature engineering, deep learning-based prediction, and fuzzy rule-based recommendation generation. Along with prediction, the project also creates various graphs and visual reports to analyze model performance and patient health conditions.

🌟 Project Highlights
Predicts diabetes risk using Deep Learning
Provides personalized diet and exercise recommendations
Uses a hybrid ANN + Fuzzy Logic approach
Generates medical and analytical graphs automatically
Performs feature engineering and preprocessing
Uses 5-Fold Cross Validation for better model performance
Includes a sample patient prediction system
Easy to understand and beginner-friendly implementation
🛠️ Technologies Used
Python
TensorFlow / Keras
Scikit-learn
NumPy
Pandas
Matplotlib
Scikit-Fuzzy
📂 Dataset Used

The project uses the PIMA Indians Diabetes Dataset.

Dataset Source:
Kaggle - PIMA Indians Diabetes Dataset

Download the diabetes.csv file and place it in the same folder as the Python file.

📁 Project Structure
LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes/
│
├── diabetes.py
├── diabetes.csv
├── graphs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── performance_metrics.png
│   ├── training_history.png
│   ├── feature_importance.png
│   ├── fuzzy_membership_functions.png
│   ├── risk_distribution.png
│   └── sample_patient_radar.png
│
└── README.md
⚙️ How the System Works
1️⃣ Data Preprocessing

The dataset is first cleaned and prepared before training the model.

The preprocessing steps include:

Handling missing values
Replacing invalid zero values
Feature scaling using StandardScaler
Creating additional health-related features such as:
Physical Activity
Diet Score
Stress Level

These additional features help improve the prediction accuracy of the model.

2️⃣ Artificial Neural Network (ANN)

The ANN model is used to predict whether a person is likely to have diabetes or not.

The network includes:

Multiple Dense Layers
Dropout Layers
Batch Normalization
Adam Optimizer
Binary Cross Entropy Loss Function

The model is trained using 5-Fold Cross Validation to improve reliability and reduce overfitting.

3️⃣ Fuzzy Logic Recommendation System

After predicting the diabetes risk, the system uses Fuzzy Logic to generate personalized health recommendations.

The fuzzy system considers:

Diabetes Risk
Physical Activity Level
Sugar Intake Level

Based on these values, the system provides recommendations such as:

Strict Diet Plans
Moderate Lifestyle Changes
Health Maintenance Suggestions

The fuzzy system contains 27 intelligent rules designed using expert knowledge.

📊 Graphs Generated

The project automatically generates several graphs for better understanding and analysis.

These include:

Confusion Matrix
ROC Curve
Performance Metrics Graph
Training Accuracy & Loss Graphs
Feature Importance Graph
Fuzzy Membership Function Graphs
Risk Distribution Graph
Patient Radar Chart

All generated graphs are stored inside the graphs/ folder.

▶️ Installation and Execution
Step 1: Clone the Repository
git clone https://github.com/Cherisma1810/LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes.git
cd LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes
Step 2: Install Required Libraries
pip install numpy pandas matplotlib scikit-learn tensorflow keras scikit-fuzzy
Step 3: Download the Dataset

Download diabetes.csv from:

PIMA Indians Diabetes Dataset

Place the dataset file in the same folder as diabetes.py.

Step 4: Update Dataset Path

Inside the code, update:

DATASET_PATH = "path_to_diabetes.csv"
Step 5: Run the Project
python diabetes.py
📈 Performance Evaluation

The system evaluates model performance using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score

It also displays:

Classification Report
Confusion Matrix
🥗 Sample Recommendations

Depending on the patient’s diabetes risk level, the system may provide recommendations like:

STRICT
Avoid refined sugars
Follow a calorie-controlled diet
Perform regular exercise
Monitor blood glucose frequently
MODERATE
Reduce processed food intake
Maintain balanced nutrition
Walk regularly
Stay hydrated
MAINTAIN
Continue healthy habits
Exercise regularly
Maintain proper BMI
Attend annual health checkups
🔮 Future Improvements

Some future enhancements for this project include:

Web application integration
Mobile health monitoring
Real-time patient tracking
Multi-disease prediction system
IoT device integration
Explainable AI dashboard
👩‍💻 Author

Developed by Cherisma Anamala

This project was created as part of academic learning and research in the field of Artificial Intelligence, Machine Learning, and Healthcare Analytics.

GitHub Repository:
Project Repository

📜 License

This project is intended for educational and research purposes only.
