import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# 1. Загрузка набора данных diabetes
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names
print("Набор данных diabetes успешно загружен.")
print(f"Форма данных (количество образцов, количество признаков): {x.shape}")
print(f"Форма целевой переменной: {y.shape}")
print(f"Доступные признаки: {feature_names}")

# 2. Исследование данных и выбор подходящего столбца.

# Посмотрим на корреляции между признаками и целевой переменной.
diabetes_df = pd.DataFrame(x, columns=feature_names)
diabetes_df['target'] = y

# Вычислим матрицу корреляций
correlation_matrix = diabetes_df.corr()

# Выведем корреляцию каждого признака с целевой переменной, отсортированную по убыванию
print("\nКорреляция признаков с целевой переменной:")
print(correlation_matrix['target'].sort_values(ascending=False))

chosen_feature_name = 'bmi'
chosen_feature_index = feature_names.index(chosen_feature_name)
print(f"\nДля линейной регрессии выбран признак: '{chosen_feature_name}'")
X_selected_feature = x[:, chosen_feature_index].reshape(-1, 1)

# 3. Реализация метода линейной регрессии с использованием Scikit-Learn
print("\n" + "-" * 30 + "sklearn_linear_model" + "-" * 30)

model = LinearRegression()
model.fit(X_selected_feature, y)
sklearn_slope = model.coef_[0]
sklearn_intercept = model.intercept_
print(f"\nsklearn_slope: {sklearn_slope:.2f}")
print(f"sklearn_intercept: {sklearn_intercept:.2f}")

# 4. Вывод таблицы с результатами предсказаний sklearn
sklearn_predictions = model.predict(X_selected_feature)
results_df = pd.DataFrame({
    f'{chosen_feature_name}': X_selected_feature.flatten(),
    'Фактическое значение (y)': y,
    'Предсказанное значение (y_pred)': sklearn_predictions
})

print("\nТаблица сравнения фактических и предсказанных значений:")
print(results_df.head())
print("...")
print(results_df.tail())

# 5. Своя реализация линейной регрессии
print("\n" + "-" * 30 + "my_linear_model" + "-" * 30)

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

my_regression_slope, my_regression_intercept = regression(X_selected_feature.flatten(), y)
print(f"\nmy_regression_slope: {my_regression_slope:.2f}")
print(f"my_regression_intercept: {my_regression_intercept:.2f}")

# 6. Вывод таблицы с результатами предсказаний моей линейной регрессии
my_predictions = my_regression_intercept + my_regression_slope * X_selected_feature.flatten()
results_df_manual = pd.DataFrame({
    f'{chosen_feature_name}': X_selected_feature.flatten(),
    'Фактическое значение (y)': y,
    'Предсказанное значение (y_pred)': my_predictions
})

print("\nТаблица сравнения фактических и предсказанных значений (ваша реализация):")
print(results_df_manual.head())
print("...")
print(results_df_manual.tail())

# 7. Вывод графиков
# Можно также визуализировать результаты (опционально)
plt.scatter(X_selected_feature, y, color='blue', label='Исходные данные')
plt.plot(X_selected_feature, sklearn_predictions, color='red', linewidth=2, label=f'sklearn: y = {sklearn_slope:.2f}x + {sklearn_intercept:.2f}')
plt.plot(X_selected_feature, my_predictions, color='green', linewidth=1, label=f'my_regres: y = {my_regression_slope:.2f}x + {my_regression_intercept:.2f}')
plt.xlabel(f'Значение признака: {chosen_feature_name}')
plt.ylabel('Прогрессия диабета')
plt.legend()
plt.grid(True)
plt.show()