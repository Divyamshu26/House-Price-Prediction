# House Price Prediction – End-to-End ML Project

This project builds an end-to-end **House Price Prediction** pipeline using the **California Housing** dataset.  
It demonstrates a complete **Data Science workflow** from data exploration to model selection, hyperparameter tuning, evaluation, and visualization.

## 🎯 Project Objectives

- Predict median house prices based on location and neighborhood features  
- Compare multiple regression models (Linear, Regularized, Tree-based, Boosting)  
- Apply **hyperparameter tuning** and **cross-validation**  
- Analyze **feature importance** and **residuals**  
- Export results for a **Power BI dashboard**

---

## 🧰 Tech Stack

- **Language:** Python (3.x)
- **Libraries:**  
  - Data: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - ML: `scikit-learn`, `xgboost`  
- **Tools:** Jupyter / Google Colab, GitHub, Power BI

---

## 📊 Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`  
- **Target Variable:** `MedHouseVal` (Median house value in units of 100,000 USD)  
- **Features include:**
  - `MedInc` – Median income in block  
  - `HouseAge` – Median house age  
  - `AveRooms`, `AveBedrms` – Average rooms/bedrooms per household  
  - `Population`, `AveOccup` – Population & avg occupancy  
  - `Latitude`, `Longitude`

---

## 🔄 Workflow

1. **Data Loading & Cleaning**
   - Load dataset from `sklearn`
   - Inspect types, missing values, summary statistics

2. **Exploratory Data Analysis (EDA)**
   - Target distribution & correlations
   - Relationships between key features and house prices

3. **Feature Scaling & Splitting**
   - Train–test split
   - Standardization for linear models

4. **Baseline & Advanced Models**
   - Linear Regression  
   - Ridge & Lasso Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  
   - XGBoost Regressor  

5. **Hyperparameter Tuning**
   - `RandomizedSearchCV` on Random Forest  
   - Best parameters selection using RMSE (negative root mean squared error)

6. **Model Evaluation**
   - Metrics: MAE, RMSE, R²  
   - Cross-validation on the best model  
   - Residual analysis

7. **Feature Importance**
   - Tree-based feature importance plot  
   - Insights into which features most influence house prices

8. **Export for Power BI**
   - Save actual vs predicted values and residuals to  
     `house_price_predictions_powerbi.csv`  

---

## 📈 Results

- Multiple models compared in a single table (MAE, RMSE, R²)
- Tuned Random Forest / XGBoost typically outperform baseline linear models
- `MedInc`, `Latitude`, and `Longitude` often emerge as the most important features

*(You can add your actual numbers and screenshots here once you run the notebook).*

---

## 📊 Power BI Dashboard (Optional)

This project ships with an export file: `house_price_predictions_powerbi.csv`.

You can:
- Load it into **Power BI Desktop**
- Build:
  - Scatter plot: Actual vs Predicted  
  - Bar chart: Feature importance  
  - Card visuals: Average price, RMSE, MAE  
  - Slicers: Filter by binned income, house age, or location

---

## 🚀 How to Run

### Option 1 – Google Colab
1. Open the notebook in Google Colab  
2. Run all cells (top → bottom)  
3. Download `house_price_predictions_powerbi.csv` to use in Power BI

### Option 2 – Local
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
jupyter notebook
