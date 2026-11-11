# Module 3: Logistic Regression (Final)

Sigmoid function, decision boundary aur scikit-learn se logistic regression, ek hi folder me: notebook (outputs ke saath), script, helper `utils.py` aur plots.

## Files

| File | Description |
|------|-------------|
| `Module_3_Logistic_Regression_SOLVED.ipynb` | Solved notebook, code aur saved outputs (prints + plots) ek hi jagah |
| `logistic_regression.py` | Same solution ka standalone Python script |
| `utils.py` | Original helper `plot_data(X, y, ax)` (green `x` = y=1, blue `.` = y=0) |
| `sigmoid.png`, `decision_boundary.png` | Script se bane plots |

## Setup

```bash
pip install numpy matplotlib scikit-learn jupyter
```

`utils.py` notebook ke same folder me honi chahiye (`from utils import plot_data`).

```bash
jupyter notebook Module_3_Logistic_Regression_SOLVED.ipynb
# ya
python logistic_regression.py
```

## Solution summary

**1. Sigmoid:** $g(z) = \frac{1}{1+e^{-z}}$

```python
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
```

**2. Decision boundary:** given model `b = -3, w0 = 1, w1 = 1`. `h(x) >= 0.5` <=> `x0 + x1 - 3 >= 0`, yaani boundary line `x1 = 3 - x0`.

**3. scikit-learn:**
```python
lr_model = LogisticRegression()
lr_model.fit(X, y)
lr_model.predict(X)    # [0 0 0 1 1 1]
lr_model.score(X, y)   # 1.0
lr_model.coef_, lr_model.intercept_
```

## Results

| Model | w0 | w1 | b | Accuracy |
|-------|----|----|---|----------|
| Given (manual) | 1 | 1 | -3 | - |
| sklearn (default, `C=1.0`) | 0.904 | 0.736 | -2.334 | 1.0 |

## Notes
- Notebook me `y` column-vector `(6, 1)` jaata hai, isliye sklearn `DataConversionWarning` deta hai. Result par asar nahi hota. Script me `y.ravel()` use hua hai.
- Default `C=1.0` L2 regularisation lagata hai, isliye sklearn ke weights manual wale se chhote hain aur boundary thodi alag hai. Dono model saare 6 training examples sahi classify karte hain.
