import matplotlib.pyplot as plt
import numpy as np

import lmfit

# Função de minimização (função objetivo do erro)
def resid(params, x, ydata):
    decay = params['decay'].value
    offset = params['offset'].value
    omega = params['omega'].value
    amp = params['amp'].value

    y_model = offset + amp * np.sin(x*omega) * np.exp(-x/decay)
    return y_model - ydata

# Valores do y data (ten Tusscher)
decay = 5
offset = 1.0
amp = 2.0
omega = 4.0

np.random.seed(2)
x = np.linspace(0, 10, 101)
y = offset + amp*np.sin(omega*x) * np.exp(-x/decay)
yn = y + np.random.normal(size=y.size, scale=0.450)

# Valores e intervalos dos parâmetros do minimal model
params = lmfit.Parameters()
params.add('offset', 2.0, min=0, max=10.0)
params.add('omega', 3.3, min=0, max=10.0)
params.add('amp', 2.5, min=0, max=10.0)
params.add('decay', 1.0, min=0, max=10.0)

# Chamadas da minimização
o1 = lmfit.minimize(resid, params, args=(x, yn), method='leastsq', max_nfev=1000)
print("# Fit using leastsq:")
lmfit.report_fit(o1)

o2 = lmfit.minimize(resid, params, args=(x, yn), method='differential_evolution', max_nfev=1000)
print("\n\n# Fit using differential_evolution:")
lmfit.report_fit(o2)

plt.plot(x, yn, 'o', label='data')
plt.plot(x, yn+o1.residual, '-', label='leastsq')
plt.plot(x, yn+o2.residual, '--', label='diffev')
plt.legend()
plt.savefig('fit.png')

print(o1.params)
