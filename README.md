# Titanic-Survival-Prediction-
A beginner machine learning project that predicts whether a passenger 
survived the Titanic disaster using a Random Forest classifier.

---

##  Problem Statement
Given passenger information (age, gender, ticket class, etc.), 
predict whether they survived the Titanic disaster (1 = survived, 0 = did not survive).

---

##  Dataset
- **Source:** [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Size:** 891 passengers, 12 columns
- **Target variable:** `Survived` (0 or 1)

---

##  Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML model and evaluation |
| Matplotlib / Seaborn | Data visualization |

---

##  Project Workflow

1. **Data Exploration** — understand the dataset, check missing values
2. **Data Cleaning** — fill missing ages with median, fill missing ports with mode
3. **Feature Engineering** — created FamilySize, IsAlone, encoded Sex and Embarked
4. **Model Training** — Random Forest with 100 trees, max_depth=5
5. **Evaluation** — accuracy score, confusion matrix, feature importance

---

##  Results

| Metric | Value |
|--------|-------|
| Accuracy | 81.01% |
| True Positives | 52 |
| True Negatives | 93 |
| False Positives | 12 |
| False Negatives | 22 |

---

##  Key Findings

- **Gender was the most important feature** (importance = 0.444) — 
  confirms the "women and children first" evacuation policy
- **Fare and Pclass** were the next most important — 
  wealthier/higher class passengers had better survival odds
- The model was **better at predicting deaths** than survivals
- Removing the `Sex` column dropped accuracy by ~8% — 
  showing how critical that single feature was

---

##  File Structure
titanic-survival-prediction/
│
├── titanic.ipynb          # main Jupyter notebook with all code
├── train.csv              # dataset from Kaggle
├── titanic_results.png    # confusion matrix + feature importance chart
└── README.md              # this file

---

## How to Run

1. Clone this repo
2. git clone https://github.com/saniya-malik/titanic-survival-prediction-
3. 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
3. Open the notebook
jupyter notebook titanic.ipynb
4. Run all cells top to bottom

---

##  What I Learned

- How to handle missing data using median and mode filling
- How to encode categorical variables (text → numbers) for ML models
- How Random Forest works — ensemble of trees voting on a result
- How to read a confusion matrix and feature importance chart
- The difference between hyperparameters (things I set) vs parameters (things the model learns)

---

##  Hyperparameters Used

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| n_estimators | 100 | 100 trees gives stable predictions |
| max_depth | 5 | prevents overfitting |
| random_state | 42 | makes results reproducible |

---

##  Future Improvements

- Try other models like XGBoost or SVM and compare accuracy
- Do more feature engineering (extract titles from names like Mr, Mrs, Miss)
- Tune hyperparameters using GridSearchCV
- Submit predictions to Kaggle leaderboard

---

Built as part of my ML learning journey — 
