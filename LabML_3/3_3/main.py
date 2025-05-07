import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def show_information_about_model(y_test, y_pred, y_pred_proba, model_name, save_path):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open(f'{save_path}/output.txt', 'w', encoding='utf-8') as f:
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")

    # матрица ошибок и тепловая карта
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={'fontsize': 14})
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.title('Матрица ошибок', pad=15)
    plt.gcf().savefig(save_path + "Тепловая карта")
    # plt.show()

    # кривая PR
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.grid(True)
    plt.gcf().savefig(save_path + "PR")
    # plt.show()

    # кривая ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.grid(True)
    plt.gcf().savefig(save_path + "ROC")
    # plt.show()

file_path = "Titanic.csv"
df = pd.read_csv(file_path, sep=',') # исходный датасет
target = "Survived"

df_cleaned = df.dropna() # удаляем пропуски
df_prepared = df_cleaned.select_dtypes(include='number') # оставляем только столбцы с числовыми значениями
df_prepared['Sex'] = df_cleaned['Sex'].map({'male': 0, 'female': 1}) # перекодируем столбец Sex и сразу добавим
df_prepared['Embarked'] = df_cleaned['Embarked'].map({'C': 1, 'Q': 2, 'S': 3}) # аналогично с Embarked
df_prepared = df_prepared.drop(columns='PassengerId')

X, y = df_prepared.drop(columns=target), df_prepared[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# модель логистической регрессии
logisticRegression_model = LogisticRegression(random_state=0)
logisticRegression_model.fit(X_train, y_train)
y_pred_lr = logisticRegression_model.predict(X_test)
y_pred_proba_lr = logisticRegression_model.predict_proba(X_test)[:, 1]
# show_information_about_model(y_test, y_pred_lr, y_pred_proba_lr, "LogisticRegression", "LogisticRegression_info/")

# модель опорных векторов
svm_model = svm.SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
# show_information_about_model(y_test, y_pred_svm, y_pred_proba_svm, "SVM", "SVM_info/")

# модель ближайших соседей
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
# show_information_about_model(y_test, y_pred_knn, y_pred_proba_knn, "KNN", "KNN_info/")