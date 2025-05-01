import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import types

# выводим статистическую информацию
def show_info(value):
    print("\n")
    print(f"Информация для: {value.name}")
    print(f"Количество: {len(value.data)}")
    print(f"Max: {np.max(value.data)}")
    print(f"Min: {np.min(value.data)}")
    print(f"Mean: {np.mean(value.data)}")

# метод наименьших квадратов
def least_squares_method(x, y):
    n = len(x.data)
    s1, s2 = 0, 0
    M_x, M_y = np.mean(x.data), np.mean(y.data)
    for i in range(n):
        s1 += (x.data[i] - M_x) * (y.data[i] - M_y)
        s2 += (x.data[i] - M_x) ** 2
    slope = s1 / s2
    intercept = M_y - slope * M_x
    return slope, intercept

# изображение полученных результатов
def draw(x, y):
    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1) # Верхний левый
    ax2 = fig.add_subplot(2, 2, 2) # Верхний правый
    ax3 = fig.add_subplot(2, 2, 3) # Нижний левый

    slope, intercept = least_squares_method(x, y)
    Y = intercept + slope * x.data  # Предсказанные значения

    # построение первого графика
    ax1.scatter(x.data, y.data, label='Исходные точки', color='blue')
    ax1.set_title('График 1: Исходные точки')
    ax1.set_xlabel(x.name)
    ax1.set_ylabel(y.name)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # построение второго графика
    ax2.scatter(x.data, y.data, label='Исходные точки', color='blue')
    ax2.plot(x.data, Y, label=f'y = {slope:.2f}x + {intercept:.2f}', color='red')
    ax2.set_title('График 2: Регрессионная прямая')
    ax2.set_xlabel(x.name)
    ax2.set_ylabel(y.name)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # построение третьего графика
    ax3.scatter(x.data, y.data, label='Исходные точки', color='blue', zorder=5)
    ax3.plot(x.data, Y, label=f'y = {slope:.2f}x + {intercept:.2f}', color='red', zorder=4)

    # Добавляем квадраты ошибок
    for i in range(len(x.data)):
        xi = x.data[i]
        yi = y.data[i]
        # Убедимся, что Y имеет ту же длину, что и x.data/y.data
        if i < len(Y):
            y_pred_i = Y[i]

            side_len = abs(yi - y_pred_i)
            bottom_left_y = min(yi, y_pred_i)
            bottom_left_x = xi

            error_square = patches.Rectangle(
                (bottom_left_x, bottom_left_y), # Нижний левый угол
                side_len,                       # Ширина
                side_len,                       # Высота
                linewidth=1, edgecolor='gray', facecolor='lightcoral',
                alpha=0.5, linestyle='--', zorder=1 # zorder ниже точек/линии
            )
            ax3.add_patch(error_square)

    error_patch_legend = patches.Patch(facecolor='lightcoral', alpha=0.5, edgecolor='gray', linestyle='--',
                                       label='Квадрат ошибки')
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(error_patch_legend)
    ax3.legend(handles=handles) # Показываем полную легенду

    ax3.set_xlabel(x.name)
    ax3.set_ylabel(y.name)
    ax3.set_title("График 3: Квадраты ошибок")
    ax3.grid(True, linestyle='--', alpha=0.6)

    # отображение
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# чтение файла
x = types.SimpleNamespace(name="", data= np.array([]))
y = types.SimpleNamespace(name="", data=np.array([]))
# file_path = "student_scores.csv" # заменить на input
file_path = input("Введите путь к файлу: ")
with open(file_path) as file:
    reader = csv.DictReader(file, delimiter=",")
    headers = reader.fieldnames
    print(f"Доступные столбцы в файле: {", ".join(headers)}")
    x.name = input(f"Укажите название для столбца X из доступных: ")
    y.name = input(f"Укажите название для столбца Y  из доступных: ")
    x_vals, y_vals = [], []
    for row in reader:
        x_vals.append(float(row.get(x.name)))
        y_vals.append(float(row.get(y.name)))

x.data = np.array(x_vals)
y.data = np.array(y_vals)

show_info(x)
show_info(y)
draw(x, y)