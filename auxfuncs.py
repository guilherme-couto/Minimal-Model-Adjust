import numpy as np

# Standard Heaviside function
def H(x):
  if (x > 0.0):
    return 1.0
  else:
    return 0.0

# Functions for infinity values
def J_fi(u, v, parameters):
    return -v * H(u - parameters['theta_v']) * (u - parameters['theta_v']) * (parameters['u_u'] - u) / parameters['tau_fi']

def J_so(u, parameters):
    return (u - parameters['u_o']) * (1.0 - H(u - parameters['theta_w'])) / tau_o(u, parameters) + H(u - parameters['theta_w']) / tau_so(u, parameters)

def J_si(u, w, s, parameters):
    return -H(u - parameters['theta_w']) * w * s / parameters['tau_si']

def v_inf(u, parameters):
    if u < parameters['theta_v_minus']:
        return 1.0
    else:
        return 0.0

def w_inf(u, parameters):
    return (1.0 - H(u - parameters['theta_o'])) * (1.0 - u/parameters['tau_w_inf']) + H(u - parameters['theta_o']) * parameters['w_inf_star']

def tau_v_minus(u, parameters):
    return (1.0 - H(u - parameters['theta_v_minus'])) * parameters['tau_v1_minus'] + H(u - parameters['theta_v_minus']) * parameters['tau_v2_minus']

def tau_w_minus(u, parameters):
    return parameters['tau_w1_minus'] + (parameters['tau_w2_minus'] - parameters['tau_w1_minus']) * (1.0 + np.tanh(parameters['k_w_minus'] * (u - parameters['u_w_minus']))) / 2

def tau_so(u, parameters):
    return parameters['tau_so1'] + (parameters['tau_so2'] - parameters['tau_so1']) * (1.0 + np.tanh(parameters['k_so'] * (u - parameters['u_so']))) / 2

def tau_s(u, parameters):
    return (1.0 - H(u - parameters['theta_w'])) * parameters['tau_s1'] + H(u - parameters['theta_w']) * parameters['tau_s2']

def tau_o(u, parameters):
    return (1.0 - H(u - parameters['theta_o'])) * parameters['tau_o1'] + H(u - parameters['theta_o']) * parameters['tau_o2']

# Parameters

param_names = ['u_o', 'u_u', 'theta_v', 'theta_w', 'theta_v_minus', 'theta_o', 'tau_v1_minus', 'tau_v2_minus',
                'tau_v_plus', 'tau_w1_minus', 'tau_w2_minus', 'k_w_minus', 'u_w_minus', 'tau_w_plus', 'tau_fi',
                'tau_o1', 'tau_o2', 'tau_so1', 'tau_so2', 'k_so', 'u_so', 'tau_so1', 'tau_so2', 'k_so', 'u_so',
                'tau_s1', 'tau_s2', 'k_s', 'u_s', 'tau_si', 'tau_w_inf', 'w_inf_star']

# Parameters for ENDO

#u_o = 0.0
#u_u = 1.56
#theta_v = 0.3
#theta_w = 0.13
#theta_v_minus = 0.2
#theta_o = 0.006
#tau_v1_minus = 75.0
#tau_v2_minus = 10.0
#tau_v_plus = 1.4506
#tau_w1_minus = 6.0
#tau_w2_minus = 140.0
#k_w_minus = 200.0
#u_w_minus = 0.016
#tau_w_plus = 280.0
#tau_fi = 0.1
#tau_o1 = 470.0
#tau_o2 = 6.0
#tau_so1 = 40.0
#tau_so2 = 1.2
#k_so = 2.0
#u_so = 0.65
#tau_s1 = 2.7342
#tau_s2 = 2.0
#k_s = 2.0994
#u_s = 0.9087
#tau_si = 2.9013
#tau_w_inf = 0.0273
#w_inf_star = 0.78

# Parameters for EPI

#u_o = 0.0
#u_u = 1.55
#theta_v = 0.3
#theta_w = 0.13
#theta_v_minus = 0.006
#theta_o = 0.006
#tau_v1_minus = 60.0
#tau_v2_minus = 1150.0
#tau_v_plus = 1.4506
#tau_w1_minus = 60.0
#tau_w2_minus = 15.0
#k_w_minus = 65.0
#u_w_minus = 0.03
#tau_w_plus = 200.0
#tau_fi = 0.11
#tau_o1 = 400.0
#tau_o2 = 6.0
#tau_so1 = 30.0181
#tau_so2 = 0.9957
#k_so = 2.0458
#u_so = 0.65
#tau_s1 = 2.7342
#tau_s2 = 16.0
#k_s = 2.0994
#u_s = 0.9087
#tau_si = 1.8875
#tau_w_inf = 0.07
#w_inf_star = 0.94

# Parameters for M

# u_o = 0.0
# u_u = 1.61
# theta_v = 0.3
# theta_w = 0.13
# theta_v_minus = 0.1
# theta_o = 0.005
# tau_v1_minus = 80.0
# tau_v2_minus = 1.4506
# tau_v_plus = 1.4506
# tau_w1_minus = 70.0
# tau_w2_minus = 8.0
# k_w_minus = 200.0
# u_w_minus = 0.016
# tau_w_plus = 280.0
# tau_fi = 0.078
# tau_o1 = 410.0
# tau_o2 = 7.0
# tau_so1 = 91.0
# tau_so2 = 0.8
# k_so = 2.1
# u_so = 0.6
# tau_s1 = 2.7342
# tau_s2 = 4.0
# k_s = 2.0994
# u_s = 0.9087
# tau_si = 3.3849
# tau_w_inf = 0.01
# w_inf_star = 0.5