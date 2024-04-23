import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

X = np.array([1,4,6,7,10,17]).reshape((-1, 1))
Y = np.array([4, 6, 8, 10, 5, 12])

X_with_intercept = sm.add_constant(X)

model = sm.OLS(Y, X_with_intercept).fit()

predictions = model.predict(X_with_intercept)
Residuals = Y- predictions

std_dev = np.std(Residuals)

print("Confidence intervals: \n ", conf_intervals)

plt.scatter(X,Y, color = 'blue')

params = model.params
plt.plot(X, params[0] + params[1]*X, color='red', label='Regression Line')

plt.plot(X, (params[0] + params[1]*X) + std_dev, color='green', label='STD')
plt.plot(X, (params[0] + params[1]*X) - std_dev, color='blue', label='STD')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Statsmodels')
plt.legend()

# Show plot
plt.show()

