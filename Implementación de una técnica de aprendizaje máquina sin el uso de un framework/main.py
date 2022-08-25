"""
Código que tiene la función de entrenar un perceptron
Autor: Erick Hernández Silva
Fecha de creación: 24/08/2022
Última actualización: 24/08/2022
"""

import numpy as np

# Lista con todos los vectores de resultados de cada x
x_samples = []
# Lista con todos los valores esperados t
t_expected = []
# Lista de los valores output o
outputs = []
# Lista con los valores de los pesos
weights = []
# Lista con los valores del cambio en los pesos
delta_weights = []
# Learning rate con default 0.5
learning_rate = 0.5

def read_txt():
    """
    Función que lee el archivo de inputs y regresa todas las entradas
    :return:
    """
    # Procesamos los datos del txt
    with open('./input.txt') as f:
        # Lista con todas las lineas del texto parseadas a numero
        entries = []
        for line in f.readlines():
            # Lista de numeros temporal para almacenar cada linea por separado
            numbers = []
            for number in line.split(','):
                if float(number) == 0.0:
                    numbers.append(-1.0)
                else:
                    numbers.append(float(number))
            entries.append(numbers)
    return entries
        #print(entries)



def calcular_outputs(x_samples, weights):
    outputs = []
    # Calculamos los outputs con nuestra función signo
    temp_outputs = []
    number_of_samples = len(x_samples)
    number_of_values = len(x_samples[0])
    sample_index = 0
    for iteration in range(0, len(x_samples[0])):
        temp_list_x = []
        temp_list_w = []
        for sample in range(0,number_of_samples):
            temp_list_x.append(x_samples[sample][sample_index])
            temp_list_w.append(weights[sample][sample_index])
        sample_index += 1
        # Sumamos y multiplicamos
        temp_sum = 0
        for i in range(0, len(temp_list_x)):
            temp_sum += temp_list_x[i] * temp_list_w[i]
        outputs.append(float(np.sign(temp_sum)))
    return outputs

def calcular_pesos(weights, x_samples, t_expected, outputs):
    import pandas as pd
    global learning_rate
    temp_weights = []
    sample_index = 0
    temp_weights_x = d = np.empty((len(x_samples), 0)).tolist()
    df = pd.DataFrame(x_samples)
    df = df.transpose()
    df_columns = ['x_' + str(i) for i in range(0,len(x_samples))]
    weights = pd.DataFrame(weights)
    weights = weights.transpose()
    weights_columns = ['w_' + str(i) for i in range(0,len(x_samples))]
    print(weights)
    df = pd.concat([df, weights, pd.DataFrame(t_expected), pd.DataFrame(outputs)], axis=1, ignore_index=True)
    df.columns = df_columns + weights_columns + ['t','o']
    print(df)

    print(df.shape[0])
    for i in range(0, df.shape[0]-1):
        temp_weights_x[i].append(df['w_'+str(i)][i] + (learning_rate * (df['t'][i] - df['o'][i]) * df['x_'+str(i)][i]))
    print('TEMP NEW WEIGHTS')
    print(temp_weights_x)

"""    for sample in range(0, len(x_samples[0])):

        operation = weights[sample][sample_index] + (learning_rate * (t_expected[sample] -
                            outputs[sample]) * x_samples[sample][sample_index])
        #operation = weights[i][j] + (learning_rate * (t_expected[i] - outputs[i]) * x_samples[i][j])
        sample_index += 1
        temp_weights_x.append(operation)
    print(temp_weights)
    weights = temp_weights
"""


def main():
    # Hacemos el setup inicial
    entries = read_txt()
    # Colocamos los valores en sus respectivas entradas
    x_samples = entries[0:-2]
    t_expected = entries[-2]
    learning_rate = entries[-1]
    # Inicializamos los pesos iniciales en 0.1
    num_muestras = len(x_samples[0])
    num_variables_x = len(x_samples)
    weights = [[0.1] * num_muestras] * (num_variables_x)
    # Inicializamos los cambios en los pesos en ceros
    delta_weights = [[0.0] * num_muestras] * (num_variables_x)
    outputs = calcular_outputs(x_samples, weights)
    print('Outputs')
    print(outputs)

    print('Muestras x')
    print(x_samples)
    print('Valores esperados')
    print(t_expected)
    print('Learning rate ', learning_rate)
    print('Pesos')
    print(weights)
    print('Variación en los pesos ')
    print(delta_weights)
    print('Outputs')
    print(outputs)
    print('Vlidar :', outputs != t_expected)
    # Comenzamos a hacer las corridas hasta que t = o
    if outputs != t_expected:
        while outputs != t_expected:
            calcular_pesos(weights, x_samples, t_expected, outputs)
            outputs = calcular_outputs(x_samples,weights)

main()