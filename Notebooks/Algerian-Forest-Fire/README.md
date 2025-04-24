# Algerian Forest Fires – ML Regression Project

This project involves the end-to-end application of regression models to predict the likelihood of forest fires in Algeria based on environmental attributes. The dataset contains meteorological features recorded across two regions and has been used to demonstrate key machine learning steps: **data cleaning**, **EDA**, **feature engineering**, and **model building** with **Simple Linear Regression**, **Lasso**, and **Ridge Regression**.

---

## 📂 Dataset

- **Source**: [Kaggle – Algerian Forest Fires Dataset](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset)
- **Format**: CSV
- **Target Variable**: `Classes` (Fire / Not Fire)
- **Features**: Includes temperature, relative humidity (RH), wind speed (WS), rain, and other environmental indicators recorded daily from June to September 2012.

--

## 🧰 Tools & Libraries Used

- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn (LinearRegression, Lasso, Ridge, train_test_split, metrics)
- Jupyter Notebook

---

## 📊 Project Workflow

### 1. Data Cleaning
- Handled missing and inconsistent values
- Unified structure for two regions into a single dataset
- Removed redundant columns

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of temperature, RH, wind speed, etc.
- Studied relationships between variables and fire occurrence

### 3. Feature Engineering
- Encoded categorical features if any
- Normalized numerical features
- Split into training and testing sets

### 4. Model Building
- Applied **Simple Linear Regression**, **Lasso**, and **Ridge Regression**
- Compared performance using **R² Score** and **RMSE**
- Performed hyperparameter tuning for Lasso & Ridge

---

## 🧪 How to Use This Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/algerian-forest-fire-ml.git
   cd algerian-forest-fire-ml
