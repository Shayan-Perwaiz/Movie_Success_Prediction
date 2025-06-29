# 🎬 Movie Success Classification using Machine Learning

This project classifies movies as **Flop**, **Average**, or **Hit** based on their features like duration, number of reviews, Facebook likes, and more. It uses multiple machine learning models, feature engineering, and ensemble techniques to find the most accurate predictor.

---

## 🚀 Project Overview

**Goal:**  
Predict the success level of a movie using features from the `movie_metadata.csv` dataset.

**GitHub Repository:**  
[Shayan-Perwaiz/Movie Success Prediction](https://github.com/Shayan-Perwaiz/Movie-Success-Prediction)

---

## 📁 Dataset

The dataset used includes:

- `duration`
- `num_critic_for_reviews`
- `num_voted_users`
- `cast_total_facebook_likes`
- `genres`
- `imdb_score`
- And more movie metadata

---

## 🧪 ML Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualized feature relationships with IMDb score using pairplots
- Correlation matrix for understanding feature interdependencies

### 2. Data Preprocessing
- Removed missing values
- Label encoded categorical data (e.g., genres)
- Binned `imdb_score` into `Flop`, `Average`, `Hit`
- Standardized numerical columns
- Checked multicollinearity using Variance Inflation Factor (VIF)

### 3. Model Training & Evaluation
Trained and tested the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors

Metrics used:
- Accuracy Score
- Confusion Matrix
- Classification Report

### 4. Hyperparameter Tuning
- Used GridSearchCV on Random Forest to improve performance

### 5. Ensemble Model
- Combined models using Voting Classifier (soft voting)

### 6. Feature Importance
- Extracted and visualized important features from the best model

---

## ✅ Best Model

The script outputs the best model based on accuracy, for example:

**🔝 Random Forest with Accuracy: 0.86**

---

## 📊 Visualizations

- Pairplot of feature vs IMDb score
- Correlation heatmap
- Feature importance bar chart

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `statsmodels`

---

## 🧠 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Shayan-Perwaiz/Movie-Success-Prediction.git
cd Movie-Success-Prediction

# 2. (Optional) Create virtual environment and install dependencies
pip install -r requirements.txt

# 3. Run the script
python movie_success_classifier.py
# Or open the notebook in Jupyter
