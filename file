🩺 Intelligent Diabetes Risk Prediction & Personalized Nutrition System
Ever wondered if your lifestyle is quietly pushing you toward diabetes? This project was built with that question in mind.
It combines the predictive power of Artificial Neural Networks (ANN) with the nuanced decision-making of Fuzzy Logic to help people understand their diabetes risk — and more importantly, what they can do about it. Rather than just outputting a cold "yes/no" prediction, the system delivers personalized diet and lifestyle guidance tailored to each individual's health profile.
Trained on the well-known PIMA Indians Diabetes Dataset, the system handles everything from raw data cleaning to deep learning inference to fuzzy rule-based recommendations — and wraps it all up with meaningful visual reports.

🌟 What Makes This Project Special

Predicts diabetes risk using a deep learning model trained with 5-Fold Cross Validation
Goes beyond prediction — gives you actionable diet and exercise recommendations
Hybrid intelligence — blends ANN precision with Fuzzy Logic's human-like reasoning
Rich visualizations — auto-generates graphs to help you understand both the model and the patient's health
Beginner-friendly — clean, well-structured code that's easy to follow and learn from


🛠️ Technologies Used
CategoryToolsLanguagePythonDeep LearningTensorFlow / KerasML UtilitiesScikit-learnFuzzy LogicScikit-FuzzyData HandlingNumPy, PandasVisualizationMatplotlib

📂 Dataset
This project uses the PIMA Indians Diabetes Dataset, a classic benchmark in medical ML research.
Download it from: Kaggle — PIMA Indians Diabetes Dataset
Once downloaded, place diabetes.csv in the same folder as diabetes.py.

📁 Project Structure
LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes/
│
├── diabetes.py                        # Main script
├── diabetes.csv                       # Dataset (download separately)
│
├── graphs/                            # Auto-generated visualizations
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

⚙️ How It All Works
1️⃣ Data Preprocessing
Raw medical data is messy — zeroes where there shouldn't be any, missing values, features on wildly different scales. The pipeline takes care of all that:

Detects and handles missing or invalid zero values
Scales features using StandardScaler
Engineers three additional health features that improve prediction quality:

Physical Activity level
Diet Score
Stress Level



These engineered features give the model a more complete picture of a patient's lifestyle.

2️⃣ Artificial Neural Network (ANN)
The core of the prediction system. The model takes a patient's health data and estimates their diabetes risk.
The architecture includes:

Multiple Dense layers for learning complex patterns
Dropout layers to prevent overfitting
Batch Normalization for stable training
Adam optimizer with Binary Cross-Entropy loss

To make the model more reliable, it's trained using 5-Fold Cross Validation — meaning the dataset is split into 5 parts and the model is evaluated on each one in turn. This gives a much more honest picture of how the model generalizes.

3️⃣ Fuzzy Logic Recommendation System
Once the ANN gives a risk score, a fuzzy logic engine takes over to turn that number into practical advice.
It considers three inputs:

Diabetes risk level
Physical activity level
Sugar intake level

And it runs them through 27 expert-designed fuzzy rules to output one of three recommendation tiers:
Risk TierWhat It Means🔴 StrictHigh risk — significant dietary and lifestyle changes needed🟡 ModerateModerate risk — improve habits and monitor progress🟢 MaintainLow risk — keep up the good work
Fuzzy logic is ideal here because health isn't black and white. Someone with "somewhat elevated" glucose and "fairly low" activity doesn't fit neatly into a single bucket — and fuzzy logic handles that ambiguity gracefully.

📊 Graphs Generated
The system automatically saves these visualizations to the graphs/ folder:

Confusion Matrix — how often predictions are right vs. wrong
ROC Curve — the model's ability to distinguish between diabetic and non-diabetic cases
Performance Metrics — accuracy, precision, recall, F1 at a glance
Training History — how loss and accuracy evolved during training
Feature Importance — which health metrics the model relies on most
Fuzzy Membership Functions — a visual explanation of how fuzzy logic works in this system
Risk Distribution — how risk is spread across the patient population
Patient Radar Chart — a holistic view of a sample patient's health profile


▶️ Getting Started
Step 1 — Clone the repository
bashgit clone https://github.com/Cherisma1810/LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes.git
cd LifeStyle_Risk_Prediction_and_diet_recommendation_diabetes
Step 2 — Install dependencies
bashpip install numpy pandas matplotlib scikit-learn tensorflow keras scikit-fuzzy
Step 3 — Download the dataset
Grab diabetes.csv from Kaggle and place it in the project folder alongside diabetes.py.
Step 4 — Update the dataset path
Open diabetes.py and set the path to your dataset:
pythonDATASET_PATH = "path_to_diabetes.csv"
Step 5 — Run
bashpython diabetes.py
That's it. The model will train, evaluate, generate all graphs, and run a sample patient prediction.

📈 How Performance Is Evaluated
The system reports:

Accuracy — overall correctness
Precision — of all predicted positives, how many were actually positive
Recall — of all actual positives, how many were caught
F1-Score — the balance between precision and recall
ROC-AUC Score — the model's discriminative ability

You'll also get a full Classification Report and Confusion Matrix in the terminal output.

🥗 Sample Recommendations
Here's a taste of what the recommendation engine might suggest:
🔴 Strict (High Risk)

Cut out refined sugars entirely
Follow a calorie-controlled meal plan
Exercise at least 30 minutes daily
Monitor blood glucose levels regularly

🟡 Moderate (Elevated Risk)

Reduce processed food consumption
Maintain balanced macronutrient intake
Build a daily walking habit
Stay well-hydrated throughout the day

🟢 Maintain (Low Risk)

Keep up your current healthy habits
Exercise consistently
Maintain a healthy BMI
Schedule annual health checkups


🔮 What's Next
There's a lot of room to grow this project. Some ideas on the roadmap:

A full web application with a user-friendly interface
Mobile health monitoring integration
Real-time patient tracking across visits
Expansion to multi-disease prediction (hypertension, cardiovascular risk, etc.)
IoT device integration for live health data input
An Explainable AI dashboard so users can understand why they got a particular risk score


👩‍💻 About the Author
Built by Cherisma Anamala as part of academic research exploring the intersection of Artificial Intelligence, Machine Learning, and Healthcare Analytics.
📎 GitHub Repository

📜 License
This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
