import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

import regression_models.linear_models.linear_regression
from gradient_descent.optimizers import SGD


if __name__ == '__main__':
    np.random.seed(42)
    X, y = load_diabetes(return_X_y=True)
    selfImplementedLR = (regression_models.linear_models.linear_regression
    .LinearRegression(
        optimizer=SGD(learning_rate=0.1),
        max_epochs=1000,
        verbose=False
    ))

    selfImplementedLR.fit(X, y)
    coeffs1 = selfImplementedLR.get_training_metadata().get('coefficients', None)

    libaryLR = LinearRegression()
    libaryLR.fit(X, y)

    coeffs2 = libaryLR.coef_

    predictions1 = selfImplementedLR.predict(X)
    predictions2 = libaryLR.predict(X)

    data1, data2 = [], []
    for x1, x2, x3 in zip(predictions1, predictions2, y):
        data1.append(x1 - x3)
        data2.append(x2 - x3)
