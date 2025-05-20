# 📉 Linear Regression on CO₂ Emissions Dataset

This project explores **Linear Regression** using the _CO2 Emissions_Canada.csv_ dataset in three different implementations:

1. ✅ **Pure Python (No Libraries)**
2. ✅ **NumPy + Pandas**
3. ✅ **Scikit-learn (ML Libraries)**

---

## 📂 Dataset

- **Source**: `CO2 Emissions_Canada.csv`
- **Features Used**:
  - `Engine Size (L)` → input (X)
  - `CO2 Emissions (g/km)` → target (Y)

---

## 🎯 Project Objectives

- Predict CO₂ emissions based on engine size.
- Understand and compare how linear regression is implemented from scratch vs. with libraries.
- Reinforce understanding of machine learning foundations like gradient descent and standardization.

---

## 🧠 Model Implementations

### 1. 💡 Linear Regression with Scikit-learn  
**File**: `libraries_linear_regression.py`  
**Output**: `plot_with_libraries.png`

- Uses: `pandas`, `numpy`, `scikit-learn`
- Scales features using `StandardScaler`
- Trains using `LinearRegression()` from scikit-learn
- ✅ No manual loop, very fast to implement

---

### 2. 🔧 Linear Regression with NumPy and Pandas  
**File**: `numpy_linear_regression.py`  
**Output**: `plot_only_numpy.png`

- Uses: `NumPy`, `Pandas`
- Implements **gradient descent manually**
- Standardization and MSE calculated from scratch
- Uses **3000 epochs**, adjustable
- ✅ Great for understanding the math behind the model

---

### 3. 🛠️ Linear Regression with Pure Python (No Libraries)  
**File**: `no_numpy_linear_regression.py`  
**Output**: `plot_no_numpy.png`

- No external libraries (except `csv` and `math`)
- Manual CSV parsing, data scaling, prediction, and gradient calculation
- Loop-based training with 3000 epochs
- ✅ Best for reinforcing fundamentals

---

## 🧮 Standardization Explained

Before training, features are **standardized** using **Z-score normalization** to help models converge faster and avoid bias due to scale.

### 📊 Why Standardize?

- Makes **mean = 0** and **std = 1**
- Ensures features are on the same scale
- Essential for gradient descent-based models

### 🧾 Standard Deviation  

Steps:
1. Calculate the **mean**
2. Subtract the mean from each value
3. Square the results
4. Sum them up
5. Divide by number of data points
6. Take the square root

### 📈 Z-Score Normalization  

1. Subtract the mean from x
2. Divide by the standard deviation
3. This becomes x scaled

## 🧠 Model Training & Loss (Step-by-Step)

The goal of the linear regression models is to minimize the error between predicted and actual values using **gradient descent** and the **Mean Squared Error (MSE)** loss function.

---

### 📉 Mean Squared Error (MSE)

The MSE is a common loss function for regression problems.


**Steps to Calculate MSE:**

1. Use the current model to predict values:  
   `ŷ = w * x + b`

2. Subtract the predicted value from the actual value:  
   `error = y - ŷ`

3. Square each of the errors:  
   `(y - ŷ)²`

4. Sum all the squared errors.

5. Divide by the number of values `n`:  
   `MSE = (1/n) * Σ(y - ŷ)²`

---

### 🔁 Gradient Descent Update (Step-by-Step)

To reduce the loss, we update the model’s parameters (`w` and `b`) using the gradients of the loss function.

**Steps to Update Parameters:**

1. Compute the gradients:

   - Gradient with respect to weight `w`:  
     `∂L/∂w = -(2/n) * Σ(x * (y - ŷ))`

   - Gradient with respect to bias `b`:  
     `∂L/∂b = -(2/n) * Σ(y - ŷ)`

2. Multiply each gradient by the learning rate `α` (e.g., 0.001).

3. Update the parameters:

   - `w = w - α * ∂L/∂w`  
   - `b = b - α * ∂L/∂b`

4. Repeat this
