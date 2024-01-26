import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Правая часть уравнения d2U/dx2 + d2U/dy2 = f
def f(x_: float, y_: float) -> float:
    return 0.


# Внешнее граничное условие (на границе исследуемой области).
def u_outer(x_: float, y_: float) -> float:
    r = np.sqrt((x_ - sat_pos[0]) ** 2 + (y_ - sat_pos[1]) ** 2)
    return R_0 * U_0 / r


# Внутреннее граничное условие (на границе/поверхности спутника).
def u_inner(x_: float, y_: float) -> float:
    return U_0


def step(U):
    h = step_size[0]
    for i in range(1, grid_size[0] - 1):
        for j in range(1, grid_size[1] - 1):
            if (sat_shape_start_x_index <= i <= sat_shape_end_x_index
                    and sat_shape_start_y_index <= j <= sat_shape_end_y_index):
                continue
            U[i, j] = (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1]) / 4 + (h ** 2) * f(x[i], y[j])
    return U


def solve() -> tuple[object, int]:
    # Инициализируем всё нулями.
    U = np.zeros(shape=(grid_size[0], grid_size[1]), dtype=float)

    # Внешнее граничное условие (на границе исследуемой области).
    for i in range(grid_size[0]):
        U[i, 0] = u_outer(x[i], y[0])
        U[i, -1] = u_outer(x[i], y[-1])

    for j in range(grid_size[1]):
        U[0, j] = u_outer(x[0], y[j])
        U[-1, j] = u_outer(x[-1], y[j])

    # Внутреннее граничное условие (на границе/поверхности спутника).
    sat_shape_start_x_real = sat_pos[0] - sat_size[0] / 2.
    sat_shape_end_x_real = sat_pos[0] + sat_size[0] / 2.
    sat_shape_start_y_real = sat_pos[1] - sat_size[1] / 2.
    sat_shape_end_y_real = sat_pos[1] + sat_size[1] / 2.

    global sat_shape_start_x_index,  sat_shape_end_x_index, sat_shape_start_y_index, sat_shape_end_y_index
    sat_shape_start_x_index = None
    sat_shape_end_x_index = None
    sat_shape_start_y_index = None
    sat_shape_end_y_index = None

    for i, xi in enumerate(x):
        if sat_shape_start_x_index is None and xi >= sat_shape_start_x_real:
            sat_shape_start_x_index = i
        if sat_shape_end_x_index is None and xi >= sat_shape_end_x_real:
            sat_shape_end_x_index = i

    for j, yj in enumerate(y):
        if sat_shape_start_y_index is None and yj >= sat_shape_start_y_real:
            sat_shape_start_y_index = j
        if sat_shape_end_y_index is None and yj >= sat_shape_end_y_real:
            sat_shape_end_y_index = j

    for i in range(sat_shape_start_x_index, sat_shape_end_x_index + 1):
        for j in range(sat_shape_start_y_index, sat_shape_end_y_index + 1):
            U[i, j] = u_inner(x[i], y[j])

    U1 = step(U.copy())
    k = 1

    while np.max(np.abs(U - U1)) > eps:
        U = U1
        U1 = step(U.copy())
        k += 1

    return U1, k


###################################################################################################


sat_pos = (0., 0.)  # координаты центра спутника [см;см]
sat_size = (10., 10.)  # размеры спутника [см;см]
area_size = (100., 100.)  # размеры исследуемой области около спутника [см;см]

R_0 = 5.  # радиус шара (круга), которым мы "приближаем" исследуемую квадратную область
U_0 = 10.  # электрический потенциал на поверхности спутника

precision = 4  # точность решения (число знаков после запятой)
eps = 10.0 ** (-precision)  # погрешность решения
step_size = (1., 1.)  # величина шага (по X, по Y) [см]
grid_size: tuple[int, int] = (
    int(area_size[0] // step_size[0]),
    int(area_size[1] // step_size[1])
)  # размер сетки (по X, по Y) = длина стороны / величина шага

x = np.linspace(sat_pos[0] - area_size[0] / 2., sat_pos[0] + area_size[1] / 2., grid_size[0])
y = np.linspace(sat_pos[1] - area_size[1] / 2., sat_pos[1] + area_size[1] / 2., grid_size[1])


def main():
    plots_color_theme = "plasma"

    # Для красивого вывода чисел в массивах numpy.
    np.set_printoptions(linewidth=100, precision=precision, suppress=True, floatmode="fixed")

    # Для интерактивных графиков в matplotlib.
    # Ещё в настройках PyCharm (Settings -> Tools -> Python Scientific) нужно отключить Show plots in tool window.
    matplotlib.use("TkAgg")

    print("...")

    # Вывод ответа.
    U, k = solve()
    title = f"Результат. Размер сетки: {grid_size[0]}x{grid_size[1]}. Итераций: {k}"
    print(title)
    print(U)

    X, Y = np.meshgrid(x, y)
    plt.figure()
    plt.pcolormesh(X, Y, U, cmap=plots_color_theme)
    plt.colorbar()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
