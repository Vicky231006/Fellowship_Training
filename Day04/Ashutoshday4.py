import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold


# YOUR ORIGINAL FUNCTIONS

def forward(x, a, b):
    y = a*x + b
    return y


def loss(yp, ya):
    return yp - ya


def update(errors, a, b, n, x, lamda):
    grada = np.mean(2*errors*x) + 2*lamda*a
    gradb = np.mean(2*errors) + 2*lamda*b

    a = a - grada*n
    b = b - gradb*n

    return a, b


# WRAPPER FOR GRIDSEARCHCV

class MyRegression(RegressorMixin, BaseEstimator):

    def __init__(self, lamda=0.0, N=100, n=0.01):
        self.lamda = lamda
        self.N = N
        self.n = n

    def fit(self, X, Y):

        X = np.asarray(X).ravel()
        Y = np.asarray(Y).ravel()

        self.a_, self.b_ = 0.0, 0.0
        self.graph_mean_ = []

        for epoch in range(1, self.N):

            yp = forward(X, self.a_, self.b_)
            errors = loss(yp, Y)

            cost = np.mean(errors**2) + self.lamda*(
                self.a_**2 + self.b_**2
            )

            self.graph_mean_.append(cost)

            self.a_, self.b_ = update(
                errors, self.a_, self.b_,
                self.n, X, self.lamda
            )

        return self

    def predict(self, X):
        X = np.asarray(X).ravel()
        return forward(X, self.a_, self.b_)


# DATASET

X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    dtype=float
)

Y = np.array(
    [25, 30, 38, 42, 51, 57, 63, 70, 76, 85],
    dtype=float
)


# GRIDSEARCHCV

param_grid = {
    "lamda": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

model = MyRegression(N=100, n=0.01)

grid = GridSearchCV(
    model,
    param_grid,
    scoring="neg_mean_squared_error",
    cv=cv
)

grid.fit(X.reshape(-1, 1), Y)


# BEST MODEL

best_model = grid.best_estimator_

print("Best lambda:", grid.best_params_["lamda"])
print("Best CV MSE:", -grid.best_score_)
print("Slope (a):", best_model.a_)
print("Intercept (b):", best_model.b_)


# PLOT

plt.plot(best_model.graph_mean_)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Gradient Descent Loss")
plt.savefig("linreg.png")
plt.show()
