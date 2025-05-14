import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

file_path = "diabetes.csv"
df = pd.read_csv(file_path, sep=',') # исходный датасет
target = "Outcome"
print("Исходный dataset: ")
print(df)

# Было обнаружено, что в dataset некоторые параметры не заданы
# (вместо Nan они имеют значение 0). Это представляет некоторую
# проблему для исследования. Поэтому исключим строки,
# в которых есть нули
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.dropna(inplace=True)
print("\nИзмененный dataset: ")
print(df)

# ------------------------------------- Задание 1 -------------------------------------

X, y = df.drop(columns=target), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def show_info_about_model(y_true, y_pred, model_name):
    print(f"\nСтандартные метрики для {model_name}")
    print(f"{accuracy_score(y_true, y_pred) = :.4f}")
    print(f"{precision_score(y_true, y_pred) = :.4f}")
    print(f"{recall_score(y_true, y_pred) = :.4f}")
    print(f"{f1_score(y_true, y_pred) = :.4f}")

# метод логистической регрессии
lr_model = LogisticRegression(random_state=0, max_iter=400)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
show_info_about_model(y_test, y_pred_lr, "LogisticRegression")

# метод решающих деревьев
classification_tree = DecisionTreeClassifier(random_state=0)
classification_tree.fit(X_train, y_train)
y_pred_tree = classification_tree.predict(X_test)
show_info_about_model(y_test, y_pred_tree, "DecisionTreeClassifier")

"""
Решающее дерево имеет более высокую Recall, что важно для выявления как можно большего числа реальных больных 
(минимизация ложноотрицательных диагнозов). Решающее дерево больше подходит для данного датасета.
"""

# ------------------------------------- Задание 2 -------------------------------------

"""
Выбранная метрика для исследования: F1-score
Обоснование выбора F1-score: В медицинских задачах диагностики важно найти баланс между Precision
(точность положительных предсказаний) и Recall (полнота выявления больных).
"""

# Определение диапазона глубин для исследования
depths = range(1, 21)
f1_scores = []

# Обучение моделей и расчет f1_score для каждой глубины
for depth in depths:
    dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Построение графика f1_score в зависимости от глубины
plt.figure(figsize=(10, 6))
plt.plot(depths, f1_scores, marker='o')
plt.xlabel('Максимальная глубина дерева решений')
plt.ylabel('F1 Score')
plt.title('F1 Score в зависимости от глубины дерева решений')
plt.grid(True)
plt.show()

# ------------------------------------- Задание 3 -------------------------------------

# Нахождение оптимальной глубины
# optimal_depth = depths[np.argmax(f1_scores)] будет depths[0] -> 1
optimal_depth = depths[2] # Но то же максимальное значение лежит и по индексу 2. Возьмем его для наглядности
print(f"\n{optimal_depth = }")

# модель с оптимальной глубиной
optimal_dtc = DecisionTreeClassifier(max_depth=optimal_depth, random_state=0)
optimal_dtc.fit(X_train, y_train)

# визуализация дерева
dot_data = tree.export_graphviz(optimal_dtc, out_file="tree.dot",
                     feature_names=X.columns.tolist(),
                     class_names=["Не болен", "Болен"],
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)

# Какие признаки модель использовала чаще всего при принятии решений.
importances = optimal_dtc.feature_importances_
feature_names = X.columns.tolist()

feature_importance_pairs = sorted(zip(importances, feature_names), reverse=True)
sorted_importances = [pair[0] for pair in feature_importance_pairs]
sorted_feature_names = [pair[1] for pair in feature_importance_pairs]

# Рисуем диаграмму важности признаков
plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_names, sorted_importances)
plt.xticks(rotation=90)
plt.title("Важность признаков")
plt.ylabel("Важность")
plt.show()

# Построение PR- и ROC-кривых
y_pred_proba = optimal_dtc.predict_proba(X_test)[:, 1]

# PR-кривая
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.grid(True)
plt.show()

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.grid(True)
plt.show()