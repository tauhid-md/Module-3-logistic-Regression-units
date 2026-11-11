"""Module 3: Logistic Regression - solved notebook ka script version (utils.py aapki original wali use hoti hai).

Notebook jaisa hi logic: sigmoid, decision boundary plot, aur
scikit-learn LogisticRegression (default settings, C=1.0).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # plots PNG files me save hote hain
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from utils import plot_data


def sigmoid(z):
    """Compute the sigmoid of z (scalar ya numpy array)."""
    g = 1 / (1 + np.exp(-z))
    return g


def main():
    # ---- exp() demo
    input_array = np.array([1, 2, 3])
    print("Input to exp:", input_array)
    print("Output of exp:", np.exp(input_array))
    print("Input to exp:", 1)
    print("Output of exp:", np.exp(1))

    # ---- sigmoid table + plot
    z_tmp = np.arange(-10, 11)
    y_sig = sigmoid(z_tmp)
    np.set_printoptions(precision=3)
    print("\nInput (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y_sig])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(z_tmp, y_sig, c="b")
    ax.set_title("Sigmoid function")
    ax.set_ylabel("sigmoid(z)")
    ax.set_xlabel("z")
    fig.savefig("sigmoid.png", dpi=120, bbox_inches="tight")

    # ---- data
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

    # ---- decision boundary for b=-3, w0=1, w1=1
    x0 = np.arange(0, 6)
    x1 = 3 - x0
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x0, x1, c="b")
    ax.axis([0, 4, 0, 3.5])
    ax.fill_between(x0, x1, alpha=0.2)
    plot_data(X, y, ax)
    ax.set_ylabel(r"$x_1$")
    ax.set_xlabel(r"$x_0$")
    fig.savefig("decision_boundary.png", dpi=120, bbox_inches="tight")

    # ---- scikit-learn
    lr_model = LogisticRegression()
    lr_model.fit(X, y.ravel())  # ravel(): column-vector warning se bachne ke liye

    y_pred = lr_model.predict(X)
    print("\nPrediction on training set:", y_pred)
    print("Accuracy on training set:", lr_model.score(X, y.ravel()))
    print(f"Theta/Parameters of the model: w0 = {lr_model.coef_[0][0]}, "
          f"w1 = {lr_model.coef_[0][1]}, b = {lr_model.intercept_[0]}")


if __name__ == "__main__":
    main()
