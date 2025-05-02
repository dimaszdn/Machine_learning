import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

# Получаем набор данных
diabetes = datasets.load_diabetes()
data_frame = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data_frame['target'] = diabetes.target

selected_column = 'bmi'
x = data_frame[[selected_column]]
y = data_frame['target']

# Линейная регрессия sklearn
print("-" * 10 + "Линейная регрессия sklearn" + "-" * 10)
sklearn_model = LinearRegression()
sklearn_model.fit(x, y)
sklearn_slope = sklearn_model.coef_[0]
sklearn_intercept = sklearn_model.intercept_
sklearn_predictions = sklearn_model.predict(x)

print(f"sklearn_slope: {sklearn_slope:.2f}")
print(f"sklearn_intercept: {sklearn_intercept:.2f}")

# Считаем метрики для регрессии sklearn
sklearn_MAE = mean_absolute_error(y, sklearn_predictions)
sklearn_R2 = r2_score(y, sklearn_predictions)
sklearn_MAPE = mean_absolute_percentage_error(y, sklearn_predictions)
print("\nМетрики: ")
print(f"sklearn_MAE: {sklearn_MAE}")
print(f"sklearn_R2: {sklearn_R2}")
print(f"sklearn_MAPE: {sklearn_MAPE}")

# Реализация линейной регрессии
print("\n")
print("-" * 10 + "Линейная регрессия" + "-" * 10)

def regression(x, y):
    n = len(x)
    s1, s2 = 0, 0
    M_x, M_y = np.mean(x), np.mean(y)
    for i in range(n):
        s1 += (x[i] - M_x) * (y[i] - M_y)
        s2 += (x[i] - M_x) ** 2
    slope = s1 / s2
    intercept = M_y - slope * M_x
    return slope, intercept

my_slope, my_intercept = regression(x[selected_column].tolist(), y)
print(f"my_slope: {my_slope:.2f}")
print(f"my_intercept: {my_intercept:.2f}")

# Считаем метрики для моей регрессии
my_predictions = my_intercept + my_slope * np.array(x[selected_column].tolist())
my_MAE = mean_absolute_error(y, my_predictions)
my_R2 = r2_score(y, my_predictions)
my_MAPE = mean_absolute_percentage_error(y, my_predictions)
print("\nМетрики: ")
print(f"my_MAE: {my_MAE}")
print(f"my_R2: {my_R2}")
print(f"my_MAPE: {my_MAPE}")