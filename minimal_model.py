from functions import *

import lmfit

# Função de callback para imprimir progresso
def print_iter_callback(params, iter, resid, *args, **kwargs):
    print(f"Iteração {iter}: Residuals: {resid}")

def resid(params, x, ydata):
    param_values = []
    for p in param_names:
        param_values.append(params[p].value)
    
    y_model = minimal_model(param_values)
    #print(f'len(y_model) = {len(y_model)}')
    
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

# Pegar dados de tempo e do ten Tusscher
ref = read_file_to_array('tnnp-cell-epi-norm-0.010.txt')
#plot_AP(a, 'TNNP 2006 EPI Norm')
dt = 10**-2
t0 = 0
tf = 600
Num_pts = (int)((tf - t0) / dt)
time = np.linspace(0, tf, Num_pts+1)

# Ajuste do modelo
max_evals = 5000
o1 = lmfit.minimize(resid, params, args=(time, ref), method='leastsq', max_nfev=max_evals, iter_cb=print_iter_callback)

# Exibe o relatório do ajuste
print("# Fit using leastsq:")
lmfit.report_fit(o1)

# Salvar os valores dos parâmetros ajustados em um arquivo txt
with open('parametros_ajustados.txt', 'w') as f:
    f.write("# Valores ajustados dos parâmetros\n")
    for param_name in o1.params:
        param_value = o1.params[param_name].value
        f.write(f"{param_name}: {param_value}\n")

# Plot dos resultados
plt.plot(time, ref, '-', label='data')
plt.plot(time, ref + o1.residual, '-', label=f'leastsq - {max_evals} evals')
plt.title('Fit using leastsq')
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.legend()
plt.savefig('fit.png')