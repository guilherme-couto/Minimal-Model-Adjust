from functions import *

import numpy as np
import lmfit
import time as time_lib
import matplotlib.pyplot as plt
import io
import sys
import os

def resid(params, time, ydata):
    param_values = []
    for p in param_names:
        param_values.append(params[p].value)
    
    y_model = minimal_model(param_values)
    # print(f'len(y_model) = {len(y_model)}')
    
    return y_model - ydata

# Valores e intervalos dos parâmetros do minimal model
params = lmfit.Parameters()
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

# # Pegar dados de tempo e do ten Tusscher
ref = read_file_to_array('minimal-model-cell-epi-0.010.txt')
#plot_AP(a, 'TNNP 2006 EPI Norm')
dt = 10**-2
t0 = 0
tf = 600
Num_pts = (int)((tf - t0) / dt)
time = np.linspace(0, tf, Num_pts+1)


# max_nfev_values = [50, 100, 250, 500, 1000, 5000, 10000]
max_nfev_values = [50, 100, 250, 500]

# method = 'leastsq'
# method = 'differential_evolution'
method = 'basinhopping'
# method = 'global_minimize'
# method = 'brute'
# method = 'nelder'

model_folder = method
os.makedirs(model_folder, exist_ok=True)

for max_nfev in max_nfev_values:
  print("Using method =", method)
  print("Using max_nfev =", max_nfev)
  start_time = time_lib.time()
  o1 = lmfit.minimize(resid, params, args=(time, ref), method=method, max_nfev=max_nfev)
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




# import matplotlib.pyplot as plt
# import numpy as np

# # Defina uma lista de valores para max_nfev
# max_nfev_values = [50, 100, 250, 500, 1000, 5000, 10000]

# # Inicialize listas para armazenar os erros
# relative_errors = []

# # Loop sobre os valores de max_nfev
# for max_nfev in max_nfev_values:
#     # Carregar o arquivo de resultados
#     result_filename = f'result_max_nfev_{max_nfev}.txt'
#     with open(result_filename, 'r') as result_file:
#         # Ignorar linhas de comentário, se houver
#         while True:
#             line = result_file.readline()
#             if not line.startswith('#'):
#                 break

#         # Ler os resultados e ajustar o número de pontos
#         results = [float(line.strip()) for line in result_file][:len(time)]

#     # Calcular o erro relativo
#     relative_error = [abs((r - ref_val) / ref_val) if ref_val != 0 else abs(r - ref_val) for r, ref_val in zip(results, ref)]

#     # Calcular as barras de erro
#     error_bar = np.zeros(len(time))  # Substitua por valores reais de erro se disponíveis

#     # Plotar cada gráfico separadamente com barras de erro no erro relativo
#     plt.figure(figsize=(10, 6))

#     time = time[:len(results)]
#     error_bar = error_bar[:len(results)]
#     ref = ref[:len(results)]

#     # Plotar o resultado
#     plt.plot(time, ref, label='Reference', color='blue')
#     plt.plot(time, results, label=f'max_nfev={max_nfev}', color='green')

#     # Plotar o erro relativo com barras de erro
#     plt.errorbar(time, relative_error, yerr=error_bar, label='Relative Error', color='red', fmt='o', markersize=2)

#     plt.xlabel('Time')
#     plt.ylabel('Result')
#     plt.title(f'Results and Relative Error with Error Bars (max_nfev={max_nfev})')
#     plt.legend()
#     plt.savefig(f'fit_error_bar_max_nfev_{max_nfev}.png')
#     plt.close()  # Fechar o gráfico para liberar memória