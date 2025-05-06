import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
data_frame['target'] = iris.target

data_frame['species'] = data_frame['target'].map({0: iris.target_names[0],
                                                  1: iris.target_names[1],
                                                  2: iris.target_names[2]})

X = data_frame[['petal length (cm)', 'petal width (cm)']]
y = data_frame['target']

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(X, y)

# Визуализация
x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5  # Для petal length
y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5  # Для petal width

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(xx, yy, Z, cmap="plasma", alpha=0.8)
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="plasma", edgecolors='k', s=20)

plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title("Многоклассовая логистическая регрессия")
handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names, title="Species")

plt.show()