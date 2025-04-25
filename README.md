
# KNN Regressor for Podcast Listening Time Prediction

This project predicts the **Listening Time (in minutes)** of podcast episodes using a K-Nearest Neighbors (KNN) Regression model. The dataset used is from Kaggle's Playground Series - Season 5, Episode 4.

## Dataset Files
- `train.csv`: Training data with features and target variable (`Listening_Time_minutes`)
- `test.csv`: Test data with features (target not provided)

## Steps Followed

### 1. **Data Loading & Inspection**
- Loaded the datasets using pandas.
- Checked for missing values, outliers, and dataset structure.

### 2. **Data Preprocessing**
- Dropped irrelevant columns: `id`, `Podcast_Name`, `Episode_Title`
- Handled missing values:
  - `Episode_Length_minutes`: filled with median.
  - `Guest_Popularity_percentage`: filled with mean.
- Removed outliers using IQR for `Episode_Length_minutes` and `Number_of_Ads`.

### 3. **Feature Engineering**
- One-hot encoded categorical features:
  - `Genre`, `Publication_Day`, `Publication_Time`, `Episode_Sentiment`
- Standardized numerical features using `StandardScaler`.

### 4. **Model Training & Evaluation**
- Split the training set (80/20) using stratified sampling.
- Used `KNeighborsRegressor` to train and evaluate the model.
- Plotted MSE vs. different values of K to find the optimal `k`.
- Evaluated model using:
  - Mean Squared Error (MSE)
  - R-squared (R²)

### 5. **Hyperparameter Tuning**
- Performed `GridSearchCV` to find best `k` and distance metric (`euclidean`, `manhattan`).
- Used cross-validation scoring based on negative MSE.

### 6. **Predictions & Submission**
- Final predictions made on the test dataset.
- Saved predictions to `submission_knn.csv`.

## Results
- Best K and metric selected from GridSearchCV.
- Achieved performance metrics (MSE, R²) on validation/test set.

## Output Files
- `submission_knn.csv`: Submission file with predicted listening times.
- `knn_mse_vs_k.png`: Visual showing performance of different `k` values.

## Libraries Used
- `numpy`, `pandas`
- `matplotlib`, `plotly`
- `sklearn` (KNeighborsRegressor, GridSearchCV, StandardScaler)

