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
    # return (1.0 - H(u - parameters['theta_o'])) * (1.0 - u/parameters['tau_w_inf']) + H(u - parameters['theta_o']) * parameters['w_inf_star']
    return (1.0 - H(u - parameters['theta_o'])) * (parameters['u_u_2'] - u/parameters['tau_w_inf']) + H(u - parameters['theta_o']) * parameters['w_inf_star']

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
                'tau_s1', 'tau_s2', 'k_s', 'u_s', 'tau_si', 'tau_w_inf', 'w_inf_star', 'u_u_2']
