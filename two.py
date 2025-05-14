import mujoco
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
import mujoco.viewer

# Загрузка модели MuJoCo
model_path = r"D:\simrobs-main\roms_project\unitree_a1\scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Параметры симуляции
T = 12.0  # Общее время (с)
T_delay = 1.0  # Время задержки для стабилизации (с)
N = 400  # Количество временных шагов
dt = T / N
target_distance = 0.1  # Целевое перемещение (м)
z_threshold = 0.15  # Порог высоты (м)
initial_qpos = model.key_qpos[0].copy()
initial_ctrl = np.array([0, 0.9, -1.8] * 4)  # [отведение, бедро, колено] x 4

# Индексы актуаторов
foot_names = ['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']
foot_ids = [model.body(name).id for name in foot_names]
thigh_indices = [1, 4, 7, 10]  # Бедра: FR, FL, RR, RL
knee_indices = [2, 5, 8, 11]   # Колени: FR, FL, RR, RL

# Функция управления для походки "трот" с задержкой и плавной остановкой
def get_control(t, A, B, stop=False, stop_time=None):
    if stop and stop_time is not None:
        fade_time = 0.3  # Время затухания (с)
        if t < stop_time + fade_time:
            fade_factor = 1 - (t - stop_time) / fade_time
            return get_control(t, A * fade_factor, B * fade_factor, stop=False)
        return initial_ctrl.copy()
    if t < T_delay:
        return initial_ctrl.copy()
    ctrl = initial_ctrl.copy()
    T_step = 2.0  # Период шага
    t_adjusted = t - T_delay
    for i in range(4):
        phase_offset = 0 if i in [0, 3] else np.pi
        sin_term = np.sin(2 * np.pi * t_adjusted / T_step + phase_offset)
        ctrl[thigh_indices[i]] = initial_ctrl[thigh_indices[i]] - B * sin_term
        ctrl[knee_indices[i]] = initial_ctrl[knee_indices[i]] - A * sin_term
    return ctrl

# Функция стоимости для оптимизации с акцентом на энергоэффективность
def compute_cost(params):
    A, B = params
    if A < 0.1 or A > 1.5 or B < 0.1 or B > 1.5:
        return 1e6
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    
    energy = 0.0
    height_penalty = 0.0
    orientation_penalty = 0.0
    p0 = data.body('trunk').xpos.copy()
    z0 = p0[2]
    stop_triggered = False
    stop_time = None
    
    for step in range(N):
        t = step * dt
        x_pos = data.body('trunk').xpos[0]
        distance_moved = x_pos - p0[0]
        if distance_moved >= target_distance and not stop_triggered:
            stop_triggered = True
            stop_time = t
        
        data.ctrl[:] = get_control(t, A, B, stop=stop_triggered, stop_time=stop_time)
        mujoco.mj_step(model, data)
        
        tau = data.qfrc_actuator
        energy += np.sum(tau**2) * dt
        
        z_body = data.body('trunk').xpos[2]
        height_penalty += abs(z_body - z0) * dt
        
        quat = data.body('trunk').xquat
        roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1]**2 + quat[2]**2))
        pitch = np.arcsin(2 * (quat[0] * quat[2] - quat[3] * quat[1]))
        orientation_penalty += (roll**2 + pitch**2) * dt
        
        if z_body < z_threshold:
            return 1e6
    
    p_final = data.body('trunk').xpos
    distance_moved = p_final[0] - p0[0]
    distance_error = (distance_moved - target_distance)**2
    if distance_moved < 0:
        distance_error += 1000 * abs(distance_moved)
    if distance_moved < 0.05:
        distance_error += 2000 * (0.05 - distance_moved)
    
    cost = 500 * energy + 100 * distance_error + 200 * height_penalty + 200 * orientation_penalty
    return cost

# Функция для симуляции и визуализации
def run_simulation(A, B, label="Simulation"):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    p0 = data.body('trunk').xpos.copy()
    z0 = p0[2]
    print(f"\n{label}:")
    print(f"Начальная высота тела: {z0:.3f} м")
    print(f"Начальная позиция по X: {p0[0]:.3f} м")
    
    energy = 0.0
    energy_per_step = []
    height_penalty = 0.0
    orientation_penalty = 0.0
    stop_triggered = False
    stop_step = None
    stop_time = None
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(N):
            t = step * dt
            x_pos = data.body('trunk').xpos[0]
            distance_moved = x_pos - p0[0]
            if distance_moved >= target_distance and not stop_triggered:
                stop_triggered = True
                stop_step = step
                stop_time = t
                print(f"Достигнуто перемещение {distance_moved:.3f} м на шаге {step}, время {t:.2f} с. Остановка.")
            
            data.ctrl[:] = get_control(t, A, B, stop=stop_triggered, stop_time=stop_time)
            mujoco.mj_step(model, data)
            
            tau = data.qfrc_actuator
            step_energy = np.sum(tau**2) * dt
            energy += step_energy
            energy_per_step.append(step_energy)
            
            z_body = data.body('trunk').xpos[2]
            quat = data.body('trunk').xquat
            roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1]**2 + quat[2]**2))
            pitch = np.arcsin(2 * (quat[0] * quat[2] - quat[3] * quat[1]))
            height_penalty += abs(z_body - z0) * dt
            orientation_penalty += (roll**2 + pitch**2) * dt
            
            if step % 50 == 0:
                print(f"Шаг {step}, Время {t:.2f} с: Высота = {z_body:.3f} м, Позиция X = {x_pos:.3f} м, Энергия на шаге = {step_energy:.6f}")
            
            viewer.sync()
    
    p_final = data.body('trunk').xpos
    distance_moved = p_final[0] - p0[0]
    final_height = p_final[2]
    
    distance_error = (distance_moved - target_distance)**2
    if distance_moved < 0:
        distance_error += 1000 * abs(distance_moved)
    if distance_moved < 0.05:
        distance_error += 2000 * (0.05 - distance_moved)
    cost = 500 * energy + 100 * distance_error + 200 * height_penalty + 200 * orientation_penalty
    
    print(f"\n{label} результаты:")
    print(f"Конечное перемещение по X: {distance_moved:.3f} м")
    print(f"Конечная высота тела: {final_height:.3f} м")
    print(f"Общая энергия: {energy:.3f}")
    print(f"Полная стоимость: {cost:.3f}")
    print(f"Вклад штрафа за перемещение: {500 * distance_error:.3f}")
    print(f"Вклад штрафа за высоту: {200 * height_penalty:.3f}")
    print(f"Вклад штрафа за ориентацию: {200 * orientation_penalty:.3f}")
    if stop_triggered:
        print(f"Остановка произошла на шаге {stop_step}, время {stop_step * dt:.2f} с")
    
    return energy, energy_per_step, distance_moved, final_height

# Оптимизация с dual_annealing
print("Запуск оптимизации...")
bounds = [(0.1, 1.5), (0.1, 1.5)]  # Диапазоны для A и B
result = dual_annealing(compute_cost, bounds, maxiter=400)
optimal_A, optimal_B = result.x
print(f"Оптимальная амплитуда колена A: {optimal_A:.3f}")
print(f"Оптимальный угол бедра B: {optimal_B:.3f}")

# Симуляция с оптимальными параметрами
energy_opt, energy_per_step_opt, dist_opt, height_opt = run_simulation(
    optimal_A, optimal_B, label="Оптимизированные параметры"
)

# Симуляция с тестовыми параметрами для сравнения
energy_test, energy_per_step_test, dist_test, height_test = run_simulation(
    A=0.5, B=0.3, label="Тестовые параметры (A=0.5, B=0.3)"
)

# Построение графика энергозатрат
time_steps = np.arange(N) * dt
plt.figure(figsize=(10, 6))
plt.plot(time_steps, energy_per_step_opt, label="Оптимизированные параметры", color='blue')
plt.plot(time_steps, energy_per_step_test, label="Тестовые параметры (A=0.5, B=0.3)", color='orange')
plt.xlabel("Время (с)")
plt.ylabel("Энергия на шаге")
plt.title("Энергозатраты по времени")
plt.legend()
plt.grid()
plt.show()