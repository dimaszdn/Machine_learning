import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# <--- Задание 1: Предобработка данных --->

file_path = "Titanic.csv"
df = pd.read_csv(file_path, sep=',') # исходный датасет
target = "Survived"
print("Исходный датасет: ")
print(df)

df_cleaned = df.dropna() # удаляем пропуски
df_prepared = df_cleaned.select_dtypes(include='number') # оставляем только столбцы с числовыми значениями
df_prepared['Sex'] = df_cleaned['Sex'].map({'male': 0, 'female': 1}) # перекодируем столбец Sex и сразу добавим
df_prepared['Embarked'] = df_cleaned['Embarked'].map({'C': 1, 'Q': 2, 'S': 3}) # аналогично с Embarked
df_prepared = df_prepared.drop(columns='PassengerId')
print("\nПреобразованный датасет: ")
print(df_prepared)

print(f"\nПроцент потерянных данных: {((len(df) - len(df_prepared)) / len(df) * 100):.2f} %")

# <--- Задание 2: Машинное обучение --->
print("-" * 100)

def get_accuracy_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return score

X, y = df_prepared.drop(columns=target), df_prepared[target]
accuracy = get_accuracy_logistic_regression(X, y)
print(f"\nТочность модели: {accuracy:.2f}")

# Оценим теперь влияние признака Embarked
X_, y_ = df_prepared.drop(columns=[target, 'Embarked']), df_prepared[target]
accuracy_with_embarked = get_accuracy_logistic_regression(X_, y_)
print(f"\nТочность модели без признака Embarked: {accuracy_with_embarked:.2f}")

print(f"\nРазница в точности: {abs(accuracy - accuracy_with_embarked)}")