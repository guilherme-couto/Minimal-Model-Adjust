from functions import *

import numpy as np
import lmfit
import time as time_lib
import matplotlib.pyplot as plt
import io
import sys
import os

# ALTERACAO DE PARAMETROS:

# ESCOLHA DE MODELO QUE VAI FAZER O FIT
# model = 'epi'
# model = 'endo'
# model = 'm'
model = 'tnnp-epi-new-2'

# ESCOLHA DA FUNCAO DO LMFIT QUE VAI AJUSTAR (recomendado: leastsq e nelder)
# method = 'leastsq'
# method = 'differential_evolution'
# method = 'basinhopping'
# method = 'global_minimize'
method = 'nelder'

# ESCOLHA DO ARRAY DE NUMERO MAXIMO DE ITERACOES DA FUNCAO
# max_nfev_values = [50, 100, 250, 500]
max_nfev_values = [1]

# DEFINICAO DOS PARAMETROS DO MODELO EM USO (pode criar um novo e alterar seus parametros para o fit)
# Valores e intervalos dos parâmetros do minimal model
params = lmfit.Parameters()

if model == 'epi':
  params.add('u_o', 0, min=0*0.5, max=1.0)
  params.add('u_u', 1.56, min=1.56*0.5, max=1.56*2.0)
  params.add('theta_v', 0.3, min=0.3*0.5, max=0.3*2.0)
  params.add('theta_w', 0.13, min=0.13*0.5, max=0.13*2.0)
  params.add('theta_v_minus', 0.2, min=0.2*0.5, max=0.2*2.0)
  params.add('theta_o', 0.006, min=0.006*0.5, max=0.006*2.0)
  params.add('tau_v1_minus', 75, min=75.0*0.5, max=75.0*2.0)
  params.add('tau_v2_minus', 10, min=10.0*0.5, max=10.0*2.0)
  params.add('tau_v_plus', 1.4506, min=1.4506*0.5, max=1.4506*2.0)
  params.add('tau_w1_minus', 6.0, min=6.0*0.5, max=6.0*2.0)
  params.add('tau_w2_minus', 140, min=140.0*0.5, max=140.0*2.0)
  params.add('k_w_minus', 200, min=200.0*0.5, max=200.0*2.0)
  params.add('u_w_minus', 0.016, min=0.016*0.5, max=0.016*2.0)
  params.add('tau_w_plus', 280, min=280.0*0.5, max=280.0*2.0)
  params.add('tau_fi', 0.1, min=0.1*0.5, max=0.1*2.0)
  params.add('tau_o1', 470, min=470.0*0.5, max=470.0*2.0)
  params.add('tau_o2', 6.0, min=6.0*0.5, max=6.0*2.0)
  params.add('tau_so1', 40, min=40.0*0.5, max=40.0*2.0)
  params.add('tau_so2', 1.2, min=1.2*0.5, max=1.2*2.0)
  params.add('k_so', 2.0, min=2.0*0.5, max=2.0*2.0)
  params.add('u_so', 0.65, min=0.65*0.5, max=0.65*2.0)
  params.add('tau_s1', 2.7342, min=2.7342*0.5, max=2.7342*2.0)
  params.add('tau_s2', 2.0, min=2.0*0.5, max=2.0*2.0)
  params.add('k_s', 2.0994, min=2.0994*0.5, max=2.0994*2.0)
  params.add('u_s', 0.9087, min=0.9087*0.5, max=0.9087*2.0)
  params.add('tau_si', 2.9013, min=2.9013*0.5, max=2.9013*2.0)
  params.add('tau_w_inf', 0.0273, min=0.0273*0.5, max=0.0273*2.0)
  params.add('w_inf_star', 0.78, min=0.78*0.5, max=0.78*2.0)

elif model == 'endo':
  params.add('u_o', 0.0, min=0.0*0.5, max=1.0)
  params.add('u_u', 1.56, min=1.56*0.5, max=1.56*2.0)
  params.add('theta_v', 0.3, min=0.3*0.5, max=0.3*2.0)
  params.add('theta_w', 0.13, min=0.13*0.5, max=0.13*2.0)
  params.add('theta_v_minus', 0.2, min=0.2*0.5, max=0.2*2.0)
  params.add('theta_o', 0.006, min=0.006*0.5, max=0.006*2.0)
  params.add('tau_v1_minus', 75.0, min=75.0*0.5, max=75.0*2.0)
  params.add('tau_v2_minus', 10.0, min=10.0*0.5, max=10.0*2.0)
  params.add('tau_v_plus', 1.4506, min=1.4506*0.5, max=1.4506*2.0)
  params.add('tau_w1_minus', 6.0, min=6.0*0.5, max=6.0*2.0)
  params.add('tau_w2_minus', 140.0, min=140.0*0.5, max=140.0*2.0)
  params.add('k_w_minus', 200.0, min=200.0*0.5, max=200.0*2.0)
  params.add('u_w_minus', 0.016, min=0.016*0.5, max=0.016*2.0)
  params.add('tau_w_plus', 280.0, min=280.0*0.5, max=280.0*2.0)
  params.add('tau_fi', 0.1, min=0.1*0.5, max=0.1*2.0)
  params.add('tau_o1', 470.0, min=470.0*0.5, max=470.0*2.0)
  params.add('tau_o2', 6.0, min=6.0*0.5, max=6.0*2.0)
  params.add('tau_so1', 40.0, min=40.0*0.5, max=40.0*2.0)
  params.add('tau_so2', 1.2, min=1.2*0.5, max=1.2*2.0)
  params.add('k_so', 2.0, min=2.0*0.5, max=2.0*2.0)
  params.add('u_so', 0.65, min=0.65*0.5, max=0.65*2.0)
  params.add('tau_s1', 2.7342, min=2.7342*0.5, max=2.7342*2.0)
  params.add('tau_s2', 2.0, min=2.0*0.5, max=2.0*2.0)
  params.add('k_s', 2.0994, min=2.0994*0.5, max=2.0994*2.0)
  params.add('u_s', 0.9087, min=0.9087*0.5, max=0.9087*2.0)
  params.add('tau_si', 2.9013, min=2.9013*0.5, max=2.9013*2.0)
  params.add('tau_w_inf', 0.0273, min=0.0273*0.5, max=0.0273*2.0)
  params.add('w_inf_star', 0.78, min=0.78*0.5, max=0.78*2.0)
   
elif model == 'm':
  params.add('u_o', 0.0, min=0.0*0.5, max=1.0)
  params.add('u_u', 1.61, min=1.61*0.5, max=1.61*2.0)
  params.add('theta_v', 0.3, min=0.3*0.5, max=0.3*2.0)
  params.add('theta_w', 0.13, min=0.13*0.5, max=0.13*2.0)
  params.add('theta_v_minus', 0.1, min=0.1*0.5, max=0.1*2.0)
  params.add('theta_o', 0.005, min=0.005*0.5, max=0.005*2.0)
  params.add('tau_v1_minus', 80.0, min=80.0*0.5, max=80.0*2.0)
  params.add('tau_v2_minus', 1.4506, min=1.4506*0.5, max=1.4506*2.0)
  params.add('tau_v_plus', 1.4506, min=1.4506*0.5, max=1.4506*2.0)
  params.add('tau_w1_minus', 70.0, min=70.0*0.5, max=70.0*2.0)
  params.add('tau_w2_minus', 8.0, min=8.0*0.5, max=8.0*2.0)
  params.add('k_w_minus', 200.0, min=200.0*0.5, max=200.0*2.0)
  params.add('u_w_minus', 0.016, min=0.016*0.5, max=0.016*2.0)
  params.add('tau_w_plus', 280.0, min=280.0*0.5, max=280.0*2.0)
  params.add('tau_fi', 0.078, min=0.078*0.5, max=0.078*2.0)
  params.add('tau_o1', 410.0, min=410.0*0.5, max=410.0*2.0)
  params.add('tau_o2', 7.0, min=7.0*0.5, max=7.0*2.0)
  params.add('tau_so1', 91.0, min=91.0*0.5, max=91.0*2.0)
  params.add('tau_so2', 0.8, min=0.8*0.5, max=0.8*2.0)
  params.add('k_so', 2.1, min=2.1*0.5, max=2.1*2.0)
  params.add('u_so', 0.6, min=0.6*0.5, max=0.6*2.0)
  params.add('tau_s1', 2.7342, min=2.7342*0.5, max=2.7342*2.0)
  params.add('tau_s2', 4.0, min=4.0*0.5, max=4.0*2.0)
  params.add('k_s', 2.0994, min=2.0994*0.5, max=2.0994*2.0)
  params.add('u_s', 0.9087, min=0.9087*0.5, max=0.9087*2.0)
  params.add('tau_si', 3.3849, min=3.3849*0.5, max=3.3849*2.0)
  params.add('tau_w_inf', 0.01, min=0.01*0.5, max=0.01*2.0)
  params.add('w_inf_star', 0.5, min=0.5*0.5, max=0.5*2.0)

elif model == 'tnnp-epi-new-2':
  params.add('u_o', 0.39, min=0.39*0.7, max=0.39*1.5)
  params.add('u_u', 0.5, min=0.5*0.7, max=0.5*1.5)
  params.add('theta_v', 0.38, min=0.38*0.7, max=0.38*1.5)
  params.add('theta_w', 0.04, min=0.04*0.7, max=0.04*1.5)
  params.add('theta_v_minus', 0.15, min=0.15*0.7, max=0.15*1.5)
  params.add('theta_o', 0.002, min=0.002*0.7, max=0.002*1.5)
  params.add('tau_v1_minus', 130.0, min=130.0*0.7, max=130.0*1.5)
  params.add('tau_v2_minus', 16.0, min=16.0*0.7, max=16.0*1.5)
  params.add('tau_v_plus', 3.9, min=3.9*0.7, max=3.9*1.5)
  params.add('tau_w1_minus', 17.0, min=17.0*0.7, max=17.0*1.5)
  params.add('tau_w2_minus', 266.0, min=266.0*0.7, max=266.0*1.5)
  params.add('k_w_minus', 342.0, min=342.0*0.7, max=342.0*1.5)
  params.add('u_w_minus', 0.013, min=0.013*0.7, max=0.013*1.5)
  params.add('tau_w_plus', 620.0, min=620.0*0.7, max=620.0*1.5)
  params.add('tau_fi', 2.92, min=2.92*0.7, max=2.92*1.5)
  params.add('tau_o1', 29150.0, min=29150.0*0.7, max=29150.0*1.5)
  params.add('tau_o2', 294.0, min=294.0*0.7, max=294.0*1.5)
  params.add('tau_so1', 48.0, min=48.0*0.7, max=48.0*1.5)
  params.add('tau_so2', 0.98, min=0.98*0.7, max=0.98*1.5)
  params.add('k_so', 1.81, min=1.81*0.7, max=1.81*1.5)
  params.add('u_so', 0.7, min=0.7*0.7, max=0.7*1.5)
  params.add('tau_s1', 9.7, min=9.7*0.7, max=9.7*1.5)
  params.add('tau_s2', 0.8, min=0.8*0.7, max=0.8*1.5)
  params.add('k_s', 1.67, min=1.67*0.7, max=1.67*1.5)
  params.add('u_s', 3.4, min=3.4*0.7, max=3.4*1.5)
  params.add('tau_si', 1.47, min=1.47*0.7, max=1.47*1.5)
  params.add('tau_w_inf', 0.04, min=0.04*0.7, max=0.04*1.5)
  params.add('w_inf_star', 1.6, min=1.6*0.7, max=1.6*1.5)

# # Pegar dados de tempo e do ten Tusscher
# ref = read_file_to_array('minimal-model-cell-'+ model +'-0.010.txt')
ref = read_file_to_array('tnnp-cell-epi-norm-0.010.txt')
#plot_AP(a, 'TNNP 2006 EPI Norm')
dt = 10**-2
t0 = 0
tf = 600
Num_pts = (int)((tf - t0) / dt)
time = np.linspace(0, tf, Num_pts+1)

def calculate_max_derivative(data):
    derivative = np.diff(data)
    positive_derivative_max = np.max(derivative[derivative > 0])
    return positive_derivative_max

def resid(params, time, ydata):
    param_values = [params[p].value for p in param_names]
    y_model = minimal_model(param_values)

    # Cálculo dos resíduos padrão
    residual = y_model - ydata

    # Adicionando a diferença dos máximos
    max_diff = np.max(y_model) - np.max(ydata)
    residual += max_diff

    # Adicionando a diferença das derivadas positivas máximas
    der_max_diff = calculate_max_derivative(y_model) - calculate_max_derivative(ydata)
    residual += der_max_diff

    # Adicionando a diferença dos valores de repouso ao final
    rest_diff = params['theta_w'].value - ydata[-1]
    residual += rest_diff

    # Adicionando a diferença dos valores de repouso ao início
    rest_diff_init = y_model[0] - 0.01
    residual += rest_diff_init

    return residual


outer_folder = model
os.makedirs(outer_folder, exist_ok=True)
model_folder = os.path.join(outer_folder, method)
os.makedirs(model_folder, exist_ok=True)

for max_nfev in max_nfev_values:
  print("Using model =", model)
  print("Using method =", method)
  print("Using max_nfev =", max_nfev)
  start_time = time_lib.time()
  o1 = lmfit.minimize(resid, params, args=(time, ref), method=method)
  end_time = time_lib.time()
  execution_time = end_time - start_time
  print(f"Time: {execution_time:.6f} seconds")

  # Change stdout to capture report_fit output
  report_output = io.StringIO()
  sys.stdout = report_output

  print(f"\n# Fit using {method} (max_nfev={max_nfev}):")
  print(f"Time: {execution_time:.6f} seconds")
  # Print fit statistics
  lmfit.report_fit(o1)
  
  # Change stdout back to normal
  sys.stdout = sys.__stdout__

  # Save report to file
  report_filename = f'{model_folder}/report_max_nfev_{max_nfev}.txt'
  with open(report_filename, 'w') as report_file:
    report_file.write(report_output.getvalue())

  # Save result to file
  result_filename = f'{model_folder}/result_max_nfev_{max_nfev}.txt'
  with open(result_filename, 'w') as result_file:
    for residual in ref + o1.residual:
      result_file.write(f'{residual}\n')

  # Plot result
  plt.plot(time, ref, '.', label='data')
  plt.plot(time, ref + o1.residual, '-', label=f'{method} (max_nfev={max_nfev})')
  plt.legend()
  plt.savefig(f'{model_folder}/fit_{max_nfev}_max_nfev.png')
  plt.clf()