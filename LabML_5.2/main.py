import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

file_path = "diabetes.csv"
df = pd.read_csv(file_path, sep=',')
target = "Outcome"

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.dropna(inplace=True)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Исследование качества модели от глубины используемых деревьев ---
depths = range(1, 21)
accuracy_by_depth = []
f1_by_depth = []

for depth in depths:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_by_depth.append(accuracy_score(y_test, y_pred))
    f1_by_depth.append(f1_score(y_test, y_pred))

# Отрисовка зависимости качества от глубины деревьев
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracy_by_depth, marker='o', label='Accuracy')
plt.plot(depths, f1_by_depth, marker='o', label='F1-score')
plt.title('Зависимость качества Random Forest от глубины деревьев')
plt.xlabel('Максимальная глубина дерева (max_depth)')
plt.ylabel('Метрика качества')
plt.xticks(depths)
plt.grid(True)
plt.legend()
plt.show()

# --- Исследование качества модели от количества подаваемых на дерево признаков ---
n_features_options = range(1, X_train.shape[1] + 1)
accuracy_by_features = []
f1_by_features = []

for n_features in n_features_options:
    model = RandomForestClassifier(max_features=n_features, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_by_features.append(accuracy_score(y_test, y_pred))
    f1_by_features.append(f1_score(y_test, y_pred))

# Отрисовка зависимости качества от количества признаков
plt.figure(figsize=(10, 6))
plt.plot(n_features_options, accuracy_by_features, marker='o', label='Accuracy')
plt.plot(n_features_options, f1_by_features, marker='o', label='F1-score')
plt.title('Зависимость качества Random Forest от количества признаков')
plt.xlabel('Количество признаков на дерево (max_features)')
plt.ylabel('Метрика качества')
plt.xticks(n_features_options)
plt.grid(True)
plt.legend()
plt.show()

# --- Исследование качества модели от числа деревьев ---
n_estimators_options = range(10, 201, 10)
accuracy_by_estimators = []
f1_by_estimators = []
training_times = []

for n_estimators in n_estimators_options:
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    end_time = time.time()
    training_times.append(end_time - start_time)

    y_pred = model.predict(X_test)
    accuracy_by_estimators.append(accuracy_score(y_test, y_pred))
    f1_by_estimators.append(f1_score(y_test, y_pred))

# Отрисовка зависимости качества и времени обучения от числа деревьев
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(n_estimators_options, accuracy_by_estimators, marker='o', label='Accuracy', color='tab:blue')
ax1.plot(n_estimators_options, f1_by_estimators, marker='o', label='F1-score', color='tab:orange')
ax1.set_xlabel('Число деревьев (n_estimators)')
ax1.set_ylabel('Метрика качества', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)
ax1.legend(loc='upper left')

ax2 = ax1.twinx() # Создаем вторую ось Y
ax2.plot(n_estimators_options, training_times, marker='x', label='Время обучения (сек)', color='tab:green')
ax2.set_ylabel('Время обучения (сек)', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.legend(loc='upper right')

plt.title('Зависимость качества и времени обучения Random Forest от числа деревьев')
plt.xticks(n_estimators_options)
plt.show()

# --- Пункт 2 ---
print("Сравнение: ")

def show_info_about_model(y_true, y_pred, dt, model_name):
    print(f"\n--- {model_name} ---")
    print(f"{accuracy_score(y_true, y_pred) = :.4f}")
    print(f"{precision_score(y_true, y_pred) = :.4f}")
    print(f"{recall_score(y_true, y_pred) = :.4f}")
    print(f"{f1_score(y_true, y_pred) = :.4f}")
    print(f"Время обучения: {dt:.4f}")

model_rf_std = RandomForestClassifier(random_state=0)
start_time_rf_std = time.time()
model_rf_std.fit(X_train, y_train)
end_time_rf_std = time.time()
y_pred_rf_std = model_rf_std.predict(X_test)
show_info_about_model(y_test, y_pred_rf_std, end_time_rf_std - start_time_rf_std, "RandomForestClassifier")

hyperparameters = {
    'objective': 'binary:logistic',
    'n_estimators': 200,
    'max_depth': 6,
    'subsample': 0.8
}

model_xgb = xgb.XGBClassifier(**hyperparameters)
start_time_xgb = time.time()
model_xgb.fit(X_train, y_train)
end_time_xgb = time.time()
y_pred_xgb = model_xgb.predict(X_test)
show_info_about_model(y_test, y_pred_xgb, end_time_xgb - start_time_xgb, "XGBClassifier")
print(f"\nВыбранные параметры: {hyperparameters}")