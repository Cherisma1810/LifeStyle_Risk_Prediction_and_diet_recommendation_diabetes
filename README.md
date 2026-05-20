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


📜 License

This project is intended for educational and research purposes only.
