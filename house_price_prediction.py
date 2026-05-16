# ============================================================
#   HOUSE PRICE PREDICTION USING LINEAR REGRESSION
#   Author  : Faiz Ahmed Khan
#   GitHub  : github.com/shadowsming
#   Tools   : Python, Scikit-learn, Pandas, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ── STEP 1: LOAD DATASET ─────────────────────────────────
print("=" * 55)
print("  HOUSE PRICE PREDICTION - ML PROJECT")
print("=" * 55)

df = pd.read_csv('train.csv')

print(f"\n✅ Dataset Loaded")
print(f"   Total Rows    : {df.shape[0]}")
print(f"   Total Columns : {df.shape[1]}")

# ── STEP 2: SELECT IMPORTANT COLUMNS ─────────────────────
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'SalePrice']]
df = df.dropna()

print(f"\n✅ Selected Features")
print(f"   GrLivArea    - Above ground living area (sq ft)")
print(f"   BedroomAbvGr - Number of bedrooms")
print(f"   FullBath     - Number of full bathrooms")
print(f"   YearBuilt    - Year house was built")
print(f"   SalePrice    - Target variable (what we predict)")
print(f"\nSample Data:")
print(df.head())

# ── STEP 3: CHECK MISSING VALUES ─────────────────────────
print(f"\n✅ Missing Values Check")
print(df.isnull().sum())

# ── STEP 4: SPLIT X AND Y ────────────────────────────────
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

print(f"\n✅ Features and Target Split")
print(f"   X shape : {X.shape}")
print(f"   y shape : {y.shape}")

# ── STEP 5: TRAIN TEST SPLIT ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n✅ Train Test Split")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# ── STEP 6: TRAIN MODEL ──────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✅ Model Trained - Linear Regression")

# ── STEP 7: EVALUATE MODEL ───────────────────────────────
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ Model Evaluation")
print(f"   R2 Score             : {r2:.4f} ({r2*100:.2f}%)")
print(f"   Mean Absolute Error  : ${mae:,.2f}")

# ── STEP 8: VISUALIZATIONS ───────────────────────────────

# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.5, edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Sale Price ($)", fontsize=11)
plt.ylabel("Predicted Sale Price ($)", fontsize=11)
plt.title("Actual vs Predicted House Prices", fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✅ actual_vs_predicted.png saved")

# Plot 2: Feature Importance
coefficients = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(7, 4))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefficients]
coefficients.plot(kind='bar', color=colors, edgecolor='white')
plt.title("Feature Coefficients (Impact on Price)", fontsize=13, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ feature_importance.png saved")

# Plot 3: Price Distribution
plt.figure(figsize=(7, 4))
plt.hist(df['SalePrice'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
plt.xlabel("Sale Price ($)")
plt.ylabel("Count")
plt.title("Distribution of House Sale Prices", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("✅ price_distribution.png saved")

# ── STEP 9: PREDICT YOUR OWN HOUSE ───────────────────────
print("\n── Predict Custom House Price ──")

custom_houses = [
    [2000, 3, 2, 2005],   # 2000 sqft, 3 bed, 2 bath, built 2005
    [1200, 2, 1, 1990],   # 1200 sqft, 2 bed, 1 bath, built 1990
    [3500, 5, 3, 2015],   # 3500 sqft, 5 bed, 3 bath, built 2015
]

for house in custom_houses:
    price = model.predict([house])[0]
    print(f"   {house[0]} sqft | {house[1]} bed | {house[2]} bath | Built {house[3]} → Predicted: ${price:,.0f}")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE! Upload this to GitHub 🚀")
print("=" * 55)
