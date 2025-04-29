# 題目：
# (1) Euler method + Taylor method order 2 解 y'(t) = 1 + (y/t) + (y/t)^2, y(1) = 0
# (2) Runge-Kutta method 解 u1', u2' 的系統

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def f(t, y):
    return 1 + (y / t) + (y / t)**2

def f_t(t, y):
    return (-y / t**2) + (-2 * y**2 / t**3)

def f_y(t, y):
    return (1 / t) + (2 * y / t**2)

def y_exact(t):
    return t * math.tan(math.log(t))

pd.set_option('display.float_format', '{:>12.6f}'.format)
pd.set_option('display.expand_frame_repr', False)

# 題目一
print("\n(1.a) Euler Method Results:\n")
t0 = 1.0
y0 = 0.0
h = 0.1
n_steps = int((2.0 - 1.0) / h)

# Euler method
t_values = [t0]
y_euler = [y0]
y_real = [y_exact(t0)]
t = t0
y = y0
for _ in range(n_steps):
    t_new = t + h
    y_new = y + h * f(t, y)
    t_values.append(t_new)
    y_euler.append(y_new)
    y_real.append(y_exact(t_new))
    t = t_new
    y = y_new

df_euler = pd.DataFrame({
    't': t_values,
    'Euler y': y_euler,
    'Exact y': y_real,
    'Error': np.abs(np.array(y_real) - np.array(y_euler))
})
print(df_euler.to_string(index=False))

# Taylor method order 2
print("\n(1.b) Taylor Method Results:\n")
t_values_taylor = [t0]
y_taylor2 = [y0]
y_real_taylor = [y_exact(t0)]
t = t0
y = y0
for _ in range(n_steps):
    t_new = t + h
    fy = f(t, y)
    ft = f_t(t, y)
    fy_prime = f_y(t, y)
    y_new = y + h * fy + (h**2 / 2) * (ft + fy_prime * fy)
    t_values_taylor.append(t_new)
    y_taylor2.append(y_new)
    y_real_taylor.append(y_exact(t_new))
    t = t_new
    y = y_new

df_taylor2 = pd.DataFrame({
    't': t_values_taylor,
    'Taylor2 y': y_taylor2,
    'Exact y': y_real_taylor,
    'Error': np.abs(np.array(y_real_taylor) - np.array(y_taylor2))
})
print(df_taylor2.to_string(index=False))

plt.figure(figsize=(8,5))
plt.plot(t_values, y_euler, marker='o', linestyle='-', label='Euler Approximation')
plt.plot(t_values, y_real, marker='x', linestyle='--', label='Exact Solution')
plt.title('(1.a) Euler Method vs Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(t_values_taylor, y_taylor2, marker='s', linestyle='-', color='orange', label='Taylor2 Approximation')
plt.plot(t_values_taylor, y_real_taylor, marker='x', linestyle='--', color='green', label='Exact Solution')
plt.title('(1.b) Taylor2 Method vs Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

# 題目二
print("\n=== 題目二結果 ===")

def f(t, u1, u2):
    du1_dt = 9 * u1 + 24 * u2 + 5 * np.cos(t) - (1/3) * np.sin(t)
    du2_dt = -24 * u1 - 52 * u2 - 9 * np.cos(t) + (1/3) * np.sin(t)
    return np.array([du1_dt, du2_dt])

def exact_u1(t):
    return 2 * np.exp(-3*t) - np.exp(-39*t) + (1/3) * np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2 * np.exp(-39*t) - (1/3) * np.cos(t)

def rk4_system(h, T):
    n_steps = int(T / h)
    t_values = [0]
    u1_values = [4/3]
    u2_values = [2/3]
    u1_exact = [exact_u1(0)]
    u2_exact = [exact_u2(0)]

    t = 0
    u = np.array([4/3, 2/3])

    for _ in range(n_steps):
        k1 = f(t, u[0], u[1])
        k2 = f(t + h/2, u[0] + h/2 * k1[0], u[1] + h/2 * k1[1])
        k3 = f(t + h/2, u[0] + h/2 * k2[0], u[1] + h/2 * k2[1])
        k4 = f(t + h, u[0] + h * k3[0], u[1] + h * k3[1])

        u = u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_values.append(t)
        u1_values.append(u[0])
        u2_values.append(u[1])
        u1_exact.append(exact_u1(t))
        u2_exact.append(exact_u2(t))

    df = pd.DataFrame({
        't': t_values,
        'u1 (RK4)': u1_values,
        'u1 (Exact)': u1_exact,
        'Error u1': np.abs(np.array(u1_values) - np.array(u1_exact)),
        'u2 (RK4)': u2_values,
        'u2 (Exact)': u2_exact,
        'Error u2': np.abs(np.array(u2_values) - np.array(u2_exact)),
    })
    return df

def plot_results(df, h_value):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['t'], df['u1 (RK4)'], 'bo-', label='u1 RK4', markersize=4)
    plt.plot(df['t'], df['u1 (Exact)'], 'r--', label='u1 Exact')
    plt.xlabel('t')
    plt.ylabel('u1')
    plt.title(f'u1: RK4 vs Exact (h={h_value})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df['t'], df['u2 (RK4)'], 'go-', label='u2 RK4', markersize=4)
    plt.plot(df['t'], df['u2 (Exact)'], 'r--', label='u2 Exact')
    plt.xlabel('t')
    plt.ylabel('u2')
    plt.title(f'u2: RK4 vs Exact (h={h_value})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("\nRunge-Kutta with h = 0.1\n")
df_h01 = rk4_system(0.1, 1.0)
print(df_h01.to_string(index=False))
plot_results(df_h01, 0.1)

print("\nRunge-Kutta with h = 0.05\n")
df_h005 = rk4_system(0.05, 1.0)
print(df_h005.to_string(index=False))
plot_results(df_h005, 0.05)
