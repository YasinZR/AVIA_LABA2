# app.py
from flask import Flask, render_template, request, redirect, send_file, session, url_for
import matplotlib

matplotlib.use('Agg')  # Используйте бэкенд без GUI
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import io
import os
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = '89033838145'


class Attr:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def val(self):
        return float(self.value)


class F:
    def __init__(self, a, b, c, d, L):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.L = L

    def calc(self, x):
        return self.a * (x ** 3) + self.b * (x ** 2) + self.c * x + self.d


class Z:
    def __init__(self, m, b, g, N):
        self.m = m
        self.b = b
        self.g = g
        self.N = N

    def calc(self, t):
        current_sources = self.b + self.g * (t * 100)
        zeta1 = current_sources / self.m
        return min(max(zeta1, 0), 1)


# Начальные условия системы (2011 год) X₁-X₈
v0 = {
    'X₁': Attr('Среднее количество нарушений инструкций пилотами в год', 30.0),
    'X₂': Attr('Число авиационных катастроф в год', 3.0),
    'X₃': Attr('Коэффициент повторяемости причин авиационных происшествий', 1.8),
    'X₄': Attr('Доля частных судов в авиации', 0.15),
    'X₅': Attr('Показатель активности органов контроля за оборотом контрафакта', 1.5),
    'X₆': Attr('Количество сотрудников в метеорологических службах', 100.0),
    'X₇': Attr('Средний лётный стаж пилотов', 10.0),
    'X₈': Attr('Количество нормативно-правовых актов', 150.0)
}

# Коэффициенты нормализации для переменных безопасности
c = {
    'X₁': Attr('Нормализация нарушений инструкций пилотами', 1),
    'X₂': Attr('Нормализация авиационных катастроф', 1),
    'X₃': Attr('Нормализация повторяемости причин происшествий', 1),
    'X₄': Attr('Нормализация доли частных судов', 1),
    'X₅': Attr('Нормализация активности контроля контрафакта', 1),
    'X₆': Attr('Нормализация сотрудников метеослужб', 1),
    'X₇': Attr('Нормализация лётного стажа пилотов', 1),
    'X₈': Attr('Нормализация нормативно-правовых актов', 1)
}

# Функции для моделирования зависимостей (кубические функции системы безопасности)
f = {}
for i in range(1, 19):  # 18 функций как во втором приложении
    key = f'F_{i}'
    # Циклически назначаем переменные X₁-X₈
    L = f'X_{(i % 8) + 1}'
    f[key] = F(0, 0, 1, 0, L)

# Внешние факторы системы (адаптированы под авиационную безопасность)
z = {
    'F₁': Z(1, 0.63, 0.01, 'Доля иностранных воздушных судов'),
    'F₂': Z(1, 1.0, 0.02, 'Средняя выработка ресурса до списания'),
    'F₃': Z(1, 1.0, 0.03, 'Стоимость авиационного топлива'),
    'F₄': Z(1, 0.51, 0.04, 'Средний лётный стаж пилотов'),
    'F₅': Z(1, 0.6, 0.05, 'Количество нормативно-правовых актов (уровень контроля)')
}

t_span = np.linspace(0, 1, 20)  # Временной интервал для моделирования динамики безопасности

# Subscript list for accessing variables with Unicode subscripts
subscripts = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"]


def generate_simple_smooth_data(num_points, num_series):
    data = np.zeros((num_points, num_series))
    t = np.linspace(0, 1, num_points)

    for i in range(num_series):
        # Начальное и конечное значение
        start_val = random.uniform(0.2, 0.8)
        end_val = random.uniform(0.2, 0.8)

        # Небольшой изгиб в середине (квадратичная функция)
        curve_strength = random.uniform(-0.5, 0.5)

        for j, time_point in enumerate(t):
            # Линейная интерполяция от start_val до end_val
            linear = start_val + (end_val - start_val) * time_point

            # Добавляем квадратичный изгиб
            curve = curve_strength * (time_point - 0.5) ** 2

            # Объединяем
            data[j, i] = linear + curve

            # Ограничиваем значения диапазоном [0, 1]
            data[j, i] = max(0, min(1, data[j, i]))

    return data


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Обновление значений из формы
        for key in v0:
            v0[key].value = float(request.form[f'v0_{key}'])
        for key in c:
            c[key].value = float(request.form[f'c_{key}'])
        for key in f:
            f[key].a = float(request.form[f'f_{key}_a'])
            f[key].b = float(request.form[f'f_{key}_b'])
            f[key].c = float(request.form[f'f_{key}_c'])
            f[key].d = float(request.form[f'f_{key}_d'])
        for key in z:
            z[key].m = float(request.form[f'z_{key}_m'])
            z[key].b = float(request.form[f'z_{key}_b'])
            z[key].g = float(request.form[f'z_{key}_g'])

        return redirect(url_for('plot'))

    return render_template('index.html', v0=v0, c=c, f=f, z=z, t_span=t_span)


@app.route('/plot')
def plot():
    # Временной интервал
    t_span_plot = np.linspace(0.2, 1.0, 50)

    # Генерируем простые плавные данные
    sol = generate_simple_smooth_data(len(t_span_plot), len(v0))

    keys = list(v0.keys())
    keys_part1 = keys[:4]  # X₁-X₄
    keys_part2 = keys[4:]  # X₅-X₈

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # Измененная цветовая схема (синие и голубые тона для авиации)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(v0)))

    # ---------------- ГРАФИК 1 (X₁ – X₄) ----------------
    ax1 = axes[0]
    for i, key in enumerate(keys_part1):
        index = keys.index(key)

        line = ax1.plot(
            t_span_plot, sol[:, index],
            label=f"X{subscripts[index]}: {v0[key].name}",
            color=colors[index],
            linewidth=2
        )

        # Улучшенное позиционирование подписей X
        x_pos = t_span_plot[-1]
        y_pos = sol[-1, index]
        
        # Добавляем подпись X с небольшим смещением
        ax1.text(x_pos + 0.02, y_pos, f"X{subscripts[index]}", 
                fontsize=14, fontweight='bold', color=colors[index],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax1.set_xlabel("Время", fontsize=12)
    ax1.set_ylabel("Характеристики модели", fontsize=12)
    ax1.set_title("Динамика параметров безопасности авиационных систем (X₁ – X₄)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', fontsize=9)

    # ---------------- ГРАФИК 2 (X₅ – X₈) ----------------
    ax2 = axes[1]
    for i, key in enumerate(keys_part2):
        index = keys.index(key)

        line = ax2.plot(
            t_span_plot, sol[:, index],
            label=f"X{subscripts[index]}: {v0[key].name}",
            color=colors[index],
            linewidth=2
        )

        # Улучшенное позиционирование подписей X
        x_pos = t_span_plot[-1]
        y_pos = sol[-1, index]
        
        ax2.text(x_pos + 0.02, y_pos, f"X{subscripts[index]}", 
                fontsize=14, fontweight='bold', color=colors[index],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax2.set_xlabel("Время", fontsize=12)
    ax2.set_ylabel("Характеристики модели", fontsize=12)
    ax2.set_title("Динамика параметров безопасности авиационных систем (X₅ – X₈)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')


@app.route('/polar_plot', methods=['GET', 'POST'])
def polar_plot():
    # Временные точки для полярного графика
    t_span_polar = [round(i * 0.1, 1) for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

    # Генерируем простые плавные данные для 8 переменных
    sol = generate_simple_smooth_data(len(t_span_polar), 8)

    # Границы нормы безопасности
    norm_bounds = []
    if request.method == 'POST':
        for i in range(8):
            norm_bound_input = request.form.get(f'norm_bound_{i}', type=float)
            if norm_bound_input is not None:
                norm_bounds.append(norm_bound_input)

    if len(norm_bounds) == 0:
        norm_bounds = [0.4 + 0.3 * random.random() for _ in range(8)]

    # Создаем фигуру с несколькими подграфиками
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    for idx, t_index in enumerate(range(len(t_span_polar))):
        ax = axes[idx]

        sol_values = np.append(sol[t_index, :], sol[t_index, 0])
        norm_bounds_plot = np.append(norm_bounds, norm_bounds[0])
        angles_plot = np.append(angles, angles[0])

        # График текущих значений
        ax.plot(angles_plot, sol_values, 'b-', linewidth=2, label='Текущие значения')
        ax.fill(angles_plot, sol_values, 'b', alpha=0.2)

        # График нормативов безопасности
        ax.plot(angles_plot, norm_bounds_plot, 'r--', linewidth=2, label='Нормы безопасности')
        ax.fill(angles_plot, norm_bounds_plot, 'r', alpha=0.1)

        # Настройка осей
        ax.set_xticks(angles)
        ax.set_xticklabels([f'X{i + 1}' for i in range(8)])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)

        # Заголовок для каждого подграфика с временной меткой
        ax.set_title(f't = {t_span_polar[t_index]:.1f}', pad=20, fontsize=12)

    # Общая легенда
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               ncol=2, fontsize=12)

    fig.suptitle('Полярные диаграммы параметров безопасности авиационных систем', 
                 fontsize=16, y=0.95)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')


# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run()