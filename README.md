🌸 Iris Flower Classification – ML Exploration Project
Welcome to one of the classic problems in the machine learning world! In this project, we dive into the beautiful Iris dataset 🌼 and use multiple ML models to classify flowers into their species — based on their petal and sepal measurements.

📁 Dataset Overview
The dataset IRIS.csv contains 150 samples of iris flowers with the following features:

sepal_length

sepal_width

petal_length

petal_width

species (target variable)

It’s clean, balanced, and perfect for beginners to explore classification tasks.

🔍 What This Project Covers
From basic exploration to real model training — this project walks through a complete machine learning pipeline. Here’s what you’ll find:

🔎 1. Data Exploration & Visualization
Summary statistics using .describe() and .info()

Distribution plots (hist, distplot) to understand value ranges

Class distribution using value_counts()

Beautiful scatter plots of petal and sepal dimensions colored by species

Correlation heatmap to detect feature relationships

⚙️ 2. Preprocessing
Label encoding the target column (species) for model training

Train-test split (70/30) using train_test_split

🤖 3. Model Building
The following ML models were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree Classifier

📊 4. Model Evaluation
Accuracy Score: For all models

Confusion Matrix: To understand classification performance per class

Classification Report: Precision, Recall, F1-score metrics

🧪 Sample Output
Here’s a sneak peek at what the model evaluation looks like:

lua
Copy
Edit
accuracy_score: 0.95
confusion_matrix:
[[16  0  0]
 [ 0 13  1]
 [ 0  1 14]]

classification_report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        16
           1       0.93      0.93      0.93        14
           2       0.93      0.93      0.93        15
🛠️ Tools & Libraries
Python

Pandas, NumPy

Seaborn, Matplotlib

scikit-learn (LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier)

▶️ How to Run
Make sure the dataset IRIS.csv is in the correct path.

Open and run iris_dataset.py in your Python environment or Google Colab.

View the visualizations and model evaluation results.
