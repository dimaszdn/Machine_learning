from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
data_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
data_frame['target'] = iris.target

data_frame['species'] = data_frame['target'].map({0: iris.target_names[0],
                                                  1: iris.target_names[1],
                                                  2: iris.target_names[2]})
print("Исходный DataFrame (первые 5 строк):")
print(data_frame.head())
print("-" * 100)

# --- Задача 1: Отрисовка зависимостей с помощью Matplotlib ---

colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
species_list = data_frame['species'].unique()

def show_addiction(first_property, second_property):
    plt.figure(figsize=(10, 6))  # Размер графика
    for species_name in species_list:
        subset = data_frame[data_frame['species'] == species_name]
        plt.scatter(subset[first_property], subset[second_property],
                    c=colors[species_name], label=species_name, alpha=0.7)

    plt.xlabel(first_property)
    plt.ylabel(second_property)
    plt.title(f"{first_property} vs {second_property}")
    plt.legend()
    plt.grid(True)
    plt.show()

show_addiction('sepal length (cm)', 'sepal width (cm)')
show_addiction('petal length (cm)', 'petal width (cm)')

# --- Задача 2: Использование seaborn.pairplot ---

sns.pairplot(data_frame, hue='species', vars=iris.feature_names)
plt.suptitle('Pairplot для датасета Iris (раскраска по сортам)', y=1.02) # Добавляем общий заголовок над всеми графиками
plt.show()

# --- Задача 3: Подготавливаем 2 датасета  ---

setosa_versicolor_df = data_frame[data_frame['species'].isin(['setosa', 'versicolor'])].copy()
versicolor_virginica_df = data_frame[data_frame['species'].isin(['versicolor', 'virginica'])].copy()

# --- Задачи 4 - 8 ---

# ------------------------- первый датасет -------------------------
X1 = setosa_versicolor_df[iris.feature_names]
y1 = setosa_versicolor_df['target']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=42, stratify=y1)

clf1 = LogisticRegression(random_state=0)
clf1.fit(X1_train, y1_train)
y1_pred = clf1.predict(X1_test)
accuracy1 = clf1.score(X1_test, y1_test)
print(f"Точность модели (setosa_versicolor): {accuracy1:.4f}")

# ------------------------- второй датасет -------------------------
X2 = versicolor_virginica_df[iris.feature_names]
y2 = versicolor_virginica_df['target']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=42, stratify=y2)

clf2 = LogisticRegression(random_state=0)
clf2.fit(X2_train, y2_train)
y2_pred = clf2.predict(X2_test)
accuracy2 = clf2.score(X2_test, y2_test)
print(f"Точность модели (versicolor_virginica): {accuracy2:.4f}")

# --- Задача 9 ---
X_generated, y_generated = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
plt.figure(figsize=(6, 6))
plt.scatter(X_generated[:, 0], X_generated[:, 1], c=y_generated, cmap='viridis', alpha=0.7)
plt.title("Сгенерированный датасет")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.grid(True)
plt.show()

X_gener_train, X_gener_test, y_gener_train, y_gener_test = train_test_split(X_generated, y_generated, test_size=0.25, random_state=42, stratify=y_generated)

clf3 = LogisticRegression(random_state=0)
clf3.fit(X_gener_train, y_gener_train)
y_gener_pred = clf3.predict(X_gener_test)
accuracy3 = clf3.score(X_gener_test, y_gener_test)
print(f"Точность модели (сгенерированный датасет): {accuracy3:.4f}")