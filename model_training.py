import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("insurance.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

# ================================
# STEP 2: CHECK MISSING VALUES
# ================================
print("\nMissing Values:")
print(df.isnull().sum())

# ================================
# STEP 3: FEATURE ENGINEERING
# ================================
# Create BMI category (feature engineering)
def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

df["bmi_category"] = df["bmi"].apply(bmi_category)

# ================================
# STEP 4: OUTLIER HANDLING (IQR METHOD)
# ================================
def remove_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])

# apply to numeric columns
remove_outliers("bmi")
remove_outliers("charges")

# ================================
# STEP 5: SEPARATE FEATURES
# ================================
X = df.drop("charges", axis=1)
y = df["charges"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ================================
# STEP 6: IDENTIFY COLUMN TYPES
# ================================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("\nNumeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# ================================
# STEP 7: PREPROCESSING PIPELINES
# ================================

# Numerical pipeline
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ]
)

# ================================
# STEP 8: TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

#CODE (PART 2)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ================================
# BASE MODELS
# ================================
reg_lr = LinearRegression()
reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
reg_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

# ================================
# ENSEMBLE MODELS
# ================================
voting_reg = VotingRegressor([
    ('lr', reg_lr),
    ('rf', reg_rf),
    ('gb', reg_gb)
])

stacking_reg = StackingRegressor(
    estimators=[
        ('rf', reg_rf),
        ('gb', reg_gb)
    ],
    final_estimator=LinearRegression()
)

# ================================
# MODEL DICTIONARY
# ================================
models = {
    "Linear Regression": reg_lr,
    "Random Forest": reg_rf,
    "Gradient Boosting": reg_gb,
    "Voting Ensemble": voting_reg,
    "Stacking Ensemble": stacking_reg
}

#CODE (PART 3)
from sklearn.model_selection import cross_val_score

results = []

for name, model in models.items():

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # train
    pipe.fit(X_train, y_train)

    # predict
    y_pred = pipe.predict(X_test)

    # metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Model": name,
        "R2 Score": r2,
        "RMSE": rmse,
        "MAE": mae
    })

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print(results_df)

# ================================
# CROSS VALIDATION (BEST MODEL CHECK)
# ================================
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

cv_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

cv_rmse = np.sqrt(-cv_scores)

print("\nCV RMSE:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())
print("Std CV RMSE:", cv_rmse.std())

#CODE (PART 4)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

# ================================
# GRID SEARCH
# ================================
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest GridSearch Score:", -grid_search.best_score_)
print("Best Params:", grid_search.best_params_)

# ================================
# RANDOM SEARCH
# ================================
param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest RandomSearch Score:", -random_search.best_score_)
print("Best Params:", random_search.best_params_)

#CODE (PART 5)
import pickle

# ================================
# FINAL MODEL SELECTION
# ================================
best_model = grid_search.best_estimator_

# ================================
# FINAL PREDICTION
# ================================
y_final_pred = best_model.predict(X_test)

final_r2 = r2_score(y_test, y_final_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
final_mae = mean_absolute_error(y_test, y_final_pred)

print("\nFINAL MODEL PERFORMANCE")
print("R2 Score:", final_r2)
print("RMSE:", final_rmse)
print("MAE:", final_mae)

# ================================
# SAVE MODEL
# ================================
with open("insurance_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nModel saved as insurance_model.pkl")