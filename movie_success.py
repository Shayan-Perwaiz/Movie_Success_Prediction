# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv('movie_metadata.csv')

## Data Exploration
# Explore feature distributions and relationships with IMDB scores
sns.pairplot(df, x_vars=['duration', 'num_critic_for_reviews', 'num_voted_users', 'cast_total_facebook_likes'],
             y_vars='imdb_score', height=4, aspect=.8)
plt.show()

## Data Preprocessing
# Handle missing values
df = df.dropna()

# Perform label encoding for categorical variables
le = LabelEncoder()
df.loc[:, 'genres'] = le.fit_transform(df['genres'])

# Categorization of IMDB Scores
# Create a new column 'Classify' for categorization
bins = [0, 3, 6, 10]
labels = ['Flop', 'Average', 'Hit']
df['Classify'] = pd.cut(df['imdb_score'], bins=bins, labels=labels)

# Ensure only numeric columns are used for VIF calculation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
vif = pd.DataFrame()
vif['features'] = numeric_df.columns
vif['VIF'] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
print(vif)

## Feature Engineering
# Apply scaling techniques for numerical variables
scaler = StandardScaler()
df[['duration', 'num_critic_for_reviews', 'num_voted_users', 'cast_total_facebook_likes']] = scaler.fit_transform(df[['duration', 'num_critic_for_reviews', 'num_voted_users', 'cast_total_facebook_likes']])

# Prepare features and target variable
X = df.drop(['imdb_score', 'Classify'], axis=1, errors='ignore')  # Use errors='ignore' to avoid issues if 'Classify' is not present
y = df['Classify']

# Check data types and convert any non-numeric columns to numeric if possible
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
X = X.fillna(0)  # Fill NaN values with 0

# Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models for comparison
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Dictionary to store accuracy scores
accuracy_scores = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
best_rf_accuracy = grid_search.best_score_
print(f'\nBest Random Forest Model Accuracy: {best_rf_accuracy:.4f}')

# Ensemble Method: Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', best_rf_model),
    ('svc', SVC(probability=True))
], voting='soft')

voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
print(f'\nVoting Classifier Accuracy: {voting_accuracy:.4f}')

# Select the best model based on accuracy
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model_accuracy = accuracy_scores[best_model_name]
print(f'\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}')

# Final prediction using the best model
best_model = models[best_model_name]
best_model.fit(X, y)  # Fit on the entire dataset for final predictions

# Feature importances for Random Forest
if best_model_name == 'Random Forest':
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)

    # Visualization of Feature Importances
    feature_importances.plot(kind='bar', figsize=(10, 6))
    plt.title('Feature Importances')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

# Create a correlation matrix using only numeric columns
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd')
plt.title('Correlation Matrix')
plt.show()
