import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as file:
        data = np.array([float(line.strip()) for line in file if not line.startswith('#')])
    return data

def plot_relative_error(time, reference, data, max_nfev):
    relative_error = np.abs((reference - data) / np.where(reference == 0, 1, reference))

    plt.figure(figsize=(10, 6))
    plt.plot(time, relative_error, label=f'Erro Relativo {model_folder}/{method_folder} (max_nfev={max_nfev})')
    plt.xlabel('Tempo')
    plt.ylabel('Erro Relativo')
    plt.title(f'Erro Relativo do Ajuste (max_nfev={max_nfev})')
    plt.legend()
    # plt.show()
    plt.savefig(f'{model_folder}/{method_folder}/rel_{max_nfev}_max_nfev.png')
    plt.clf()

def plot_absolute_error(time, reference, data, max_nfev):
    relative_error = np.abs((reference - data))

    plt.figure(figsize=(10, 6))
    plt.plot(time, relative_error, label=f'Erro Absoluto {model_folder}/{method_folder} (max_nfev={max_nfev})')
    plt.xlabel('Tempo')
    plt.ylabel('Erro Absoluto')
    plt.title(f'Erro Absoluto do Ajuste (max_nfev={max_nfev})')
    plt.legend()
    # plt.show()
    plt.savefig(f'{model_folder}/{method_folder}/abs_{max_nfev}_max_nfev.png')
    plt.clf()

# Uso
model_folder = 'tnnp-epi-new'
method_folder = 'nelder'
max_nfev = 1
reference = read_data('tnnp-cell-epi-norm-0.010.txt')
data = read_data(f'{model_folder}/{method_folder}/result_max_nfev_{max_nfev}.txt')
time = np.linspace(0, 600, len(data))  # Assumindo que o tempo vai de 0 a 600 com o mesmo número de pontos que os dados
plot_relative_error(time, reference, data, max_nfev)
plot_absolute_error(time, reference, data, max_nfev)