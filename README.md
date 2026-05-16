# 🏠 House Price Prediction Using Linear Regression

A machine learning project to predict house sale prices based on key property features using Linear Regression.

---

## 🔍 Project Overview

This project builds a house price predictor using:
- **Linear Regression** to predict continuous sale prices
- **Feature Selection** to pick the most relevant columns
- **Scikit-learn** for model training and evaluation
- **Matplotlib** for visualizations

**R2 Score achieved: 70.6%** on test data.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── house_price_prediction.py   # Main project code
├── train.csv                   # Training dataset (from Kaggle)
├── actual_vs_predicted.png     # Model performance chart
├── feature_importance.png      # Feature coefficient chart
├── price_distribution.png      # Price distribution chart
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| Scikit-learn | Linear Regression model |
| Matplotlib | Data visualization |
| NumPy | Numerical operations |

---

## ⚙️ How It Works

1. **Load Dataset** — House Prices CSV from Kaggle
2. **Select Features** — GrLivArea, Bedrooms, Bathrooms, YearBuilt
3. **Clean Data** — Drop missing values
4. **Train Test Split** — 80% train, 20% test
5. **Train Model** — Linear Regression
6. **Evaluate** — R2 Score + Mean Absolute Error
7. **Visualize** — Actual vs Predicted scatter plot
8. **Predict** — Custom house price predictions

---

## 📊 Results

| Metric | Value |
|--------|-------|
| R2 Score | 0.706 (70.6%) |
| Mean Absolute Error | $32,044 |

---

## 🖼️ Visualizations

### Actual vs Predicted Prices
![Actual vs Predicted](actual_vs_predicted.png)

### Feature Importance
![Feature Importance](feature_importance.png)

### Price Distribution
![Price Distribution](price_distribution.png)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/shadowsming/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn numpy

# 3. Download dataset from Kaggle
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# 4. Run the project
python house_price_prediction.py
```

---

## 🧪 Custom Predictions

The project predicts prices for custom houses at the bottom:

```python
[2000 sqft | 3 bed | 2 bath | Built 2005] → ~$180,000
[1200 sqft | 2 bed | 1 bath | Built 1990] → ~$130,000
[3500 sqft | 5 bed | 3 bath | Built 2015] → ~$280,000
```

---

## 🧠 What I Learned

- How to select relevant features from a large dataset
- How Linear Regression works for predicting continuous values
- What R2 Score and Mean Absolute Error mean
- How to visualize actual vs predicted results
- How to make custom predictions using a trained model

---

## 👤 Author

**Faiz Ahmed Khan**
- GitHub: [github.com/shadowsming](https://github.com/shadowsming)
- LinkedIn: [linkedin.com/in/faiz-khan-40553026b](https://linkedin.com/in/faiz-khan-40553026b)
- Email: khanfaiz0119@gmail.com

---

## 📚 Dataset Source

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — Kaggle

---

⭐ **If you found this project helpful, give it a star!**
