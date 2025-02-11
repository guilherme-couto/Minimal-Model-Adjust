import matplotlib.pyplot as plt
import numpy as np
from auxfuncs import *

# Read file
def read_file_to_array(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    lines = [float(x) for x in lines]
    return lines

# Plot array
def plot_AP(potential_values, title):
    time = np.linspace(0, 600, num=len(potential_values))
    plt.plot(time, potential_values)
    plt.title(title)
    #plt,ylim(-80, 50)
    plt.xlabel('t (ms)')
    plt.ylabel('V (mV)')

    plt.savefig(f'{title}.png')
    plt.close()

# Minimal model
# Stimulus
def I_stim(t):
    if ((1.0 < t < 2.0)):
        return 1.0
    else:
        return 0.0
        
def minimal_model(parameters_values):
    # Set parameters
    u_o = parameters_values[0]
    u_u = parameters_values[1]
    theta_v = parameters_values[2]
    theta_w = parameters_values[3]
    theta_v_minus = parameters_values[4]
    theta_o = parameters_values[5]
    tau_v1_minus = parameters_values[6]
    tau_v2_minus = parameters_values[7]
    tau_v_plus = parameters_values[8]
    tau_w1_minus = parameters_values[9]
    tau_w2_minus = parameters_values[10]
    k_w_minus = parameters_values[11]
    u_w_minus = parameters_values[12]
    tau_w_plus = parameters_values[13]
    tau_fi = parameters_values[14]
    tau_o1 = parameters_values[15]
    tau_o2 = parameters_values[16]
    tau_so1 = parameters_values[17]
    tau_so2 = parameters_values[18]
    k_so = parameters_values[19]
    u_so = parameters_values[20]
    tau_s1 = parameters_values[21]
    tau_s2 = parameters_values[22]
    k_s = parameters_values[23]
    u_s = parameters_values[24]
    tau_si = parameters_values[25]
    tau_w_inf = parameters_values[26]
    w_inf_star = parameters_values[27]

    # Time setup
    dt = 10**-2
    t0 = 0
    tf = 600
    Num_pts = (int)((tf - t0) / dt)
    t = np.linspace(0, tf, Num_pts+1)

    # Variables

    u = np.zeros(Num_pts+1)
    v = np.zeros(Num_pts+1)
    w = np.zeros(Num_pts+1)
    s = np.zeros(Num_pts+1)

    # Initial conditions
    u[0] = 0.0
    v[0] = 1.0
    w[0] = 1.0
    s[0] = 0.0

    # Loop
    for i in range(0, Num_pts):
        u[i+1] = u[i] + dt * (- (J_fi(u[i], v[i], theta_v, u_u, tau_fi) + J_so(u[i], u_o, theta_w, theta_o, tau_o1, tau_o2, tau_so1, tau_so2, k_so, u_so) + J_si(u[i], w[i], s[i], theta_w, tau_si)) + I_stim(t[i]))
        v[i+1] = v[i] + dt * ((1.0 - H(u[i] - theta_v)) * (v_inf(u[i], theta_v_minus) - v[i]) / tau_v_minus(u[i], theta_v_minus, tau_v1_minus, tau_v2_minus) - H(u[i] - theta_v) * v[i] / tau_v_plus)
        w[i+1] = w[i] + dt * ((1.0 - H(u[i] - theta_w)) * (w_inf(u[i], theta_o, tau_w_inf, w_inf_star) - w[i]) / tau_w_minus(u[i], tau_w1_minus, tau_w2_minus, k_w_minus, u_w_minus) - H(u[i] - theta_w) * w[i] / tau_w_plus)
        s[i+1] = s[i] + dt * (((1.0 + np.tanh(k_s * (u[i] - u_s))) / 2 - s[i]) / tau_s(u[i], theta_w, tau_s1, tau_s2))

    return u