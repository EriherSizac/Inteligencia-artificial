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
# Learning rate con default 0.5
learning_rate = 0.5

def read_txt():
    """
    Función que lee el archivo de inputs y regresa todas las entradas
    :return: todas las entradas en una lista
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



def calcular_outputs(x_samples, weights):
    """
    Función que calcula los outputs con la función signo
    :param x_samples: las muestras de x
    :param weights: los pesos de las muestras
    :return:
    """
    outputs = []
    # Calculamos los outputs con nuestra función signo
    number_of_samples = len(x_samples)
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
    df = pd.concat([df, weights, pd.DataFrame(t_expected), pd.DataFrame(outputs)], axis=1, ignore_index=True)
    df.columns = df_columns + weights_columns + ['t','o']
    for i in range(0, df.shape[0]-1):
        for j in range(0, df.shape[0]):
            temp_weights_x[i].append(df['w_'+str(i)][j] + (learning_rate * (df['t'][j] - df['o'][j]) * df['x_'+str(i)][j]))
    return temp_weights_x



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
    print('=========== DATOS INICIALES ===========')
    print('Muestras x')
    print(x_samples)
    print('Valores esperados')
    print(t_expected)
    print('Pesos iniciales')
    print(weights)
    print('Outputs obtenidos con los pesos')
    print(outputs)
    print('Learning rate: ', learning_rate)
    print('========================================')
    # Comenzamos a hacer las corridas hasta que t = o
    if outputs != t_expected:
        while outputs != t_expected:
            weights = calcular_pesos(weights, x_samples, t_expected, outputs)
            outputs = calcular_outputs(x_samples,weights)
        print('=========== RESULTADOS ===========')
        print('Los pesos son: ')
        print(weights)
        print('Los outputs recibidos con estos pesos Son:')
        print(outputs)
        print('========================================')
main()