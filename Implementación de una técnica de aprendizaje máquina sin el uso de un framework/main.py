"""
Código que tiene la función de entrenar un perceptron
Autor: Erick Hernández Silva
Fecha de creación: 24/08/2022
Última actualización: 24/08/2022
"""
import random

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
learning_rate = 3
# Porcentaje de datos de entrenamiento
training_data = .7

def read_txt(file):
    """
    Función que lee el archivo de inputs y regresa todas las entradas
    :return: todas las entradas en una lista
    """
    # Procesamos los datos del txt
    with open(file+'.txt') as f:
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



def calcular_outputs(x, weights):
    """
    Función que calcula los outputs con la función signo
    :param x_samples: las muestras de x
    :param weights: los pesos de las muestras
    :return:
    """
    outputs = []
    # Calculamos los outputs con nuestra función signo
    number_of_samples = len(x)
    sample_index = 0
    for iteration in range(0, len(x[0])):
        temp_list_x = []
        temp_list_w = []
        for sample in range(0,number_of_samples):
            temp_list_x.append(x[sample][sample_index])
            temp_list_w.append(weights[sample][sample_index])
        sample_index += 1
        # Sumamos y multiplicamos
        temp_sum = 0
        for i in range(0, len(temp_list_x)):
            temp_sum += temp_list_x[i] * temp_list_w[i]
        outputs.append(float(np.sign(temp_sum)))
    return outputs

def calcular_pesos(weights, x, t_expected, outputs):
    import pandas as pd
    global learning_rate
    temp_weights_x = np.empty((len(x), 0)).tolist()
    for i in range(0, len(x)):
        #for j in range(0, df.shape[0]):
        for j in range(0, len(x[0])):
            # Python vainilla
            temp_weights_x[i].append(weights[i][j] + (learning_rate * (t_expected[j] - outputs[j]) * x[i][j]))
            # Usando dataframes
            #temp_weights_x[i].append(df['w_'+str(i)][j] + (learning_rate * (df['t'][j] - df['o'][j]) * df['x_'+str(i)][j]))
    return temp_weights_x



def main():
    import pandas as pd
    # Hacemos el setup inicial
    entries = read_txt(input('Nombre del archivo: '))
    # Colocamos los valores en sus respectivas entradas
    x_samples = entries[0:-2]
    t_expected = entries[-2]
    learning_rate = entries[-1]

    # Hacemos un shuffle a las listas
    df_t = pd.DataFrame(x_samples).transpose()
    df_t = pd.concat([df_t, pd.DataFrame(t_expected)], axis=1)
    # Se hace el shuffle y se resetean los indices
    df_t = df_t.sample(frac=1).reset_index(drop=True)

    # Lo enviamos a las listas de nuevo
    x_samples = df_t.iloc[:, 0:len(x_samples)].transpose().values.tolist()
    t_expected = df_t.iloc[:, len(x_samples):].values.tolist()
    temp = []
    # Cambiamos nuevamente la lista de valores de t
    for each in t_expected:
        temp.append(each[0])
    t_expected = temp

    # Separamos las muestras de entrenamiento y prueba
    lim_training = int(len(x_samples[0])*training_data)
    print(lim_training)
    x_training = np.empty((len(x_samples), 0)).tolist()
    x_test = np.empty((len(x_samples), 0)).tolist()
    t_training = []
    t_test = []
    for i in range(0,len(x_samples)):
        for j in range(0,lim_training):
            x_training[i].append(x_samples[i][j])
        for j in range(lim_training,len(x_samples[0])):
            x_test[i].append(x_samples[i][j])
    for k in range(0, lim_training):
        t_training.append(t_expected[k])
        t_test.append(t_expected[k])


    # Inicializamos los pesos iniciales en 0.1
    num_muestras = len(x_training[0])
    num_variables_x = len(x_training)
    weights = [[0.1] * num_muestras] * (num_variables_x)
    # Inicializamos los cambios en los pesos en ceros
    delta_weights = [[0.0] * num_muestras] * (num_variables_x)
    outputs = calcular_outputs(x_training, weights)
    # Comenzamos a hacer las corridas hasta que t = o
    if outputs != t_expected:
        while outputs != t_training:
            weights = calcular_pesos(weights, x_training, t_expected, outputs)
            outputs = calcular_outputs(x_training,weights)
    res = calcular_outputs(x_test, weights)
    resultados = []
    for resultado in res:
        if resultado == -1.0:
            resultados.append(0)
        else:
            resultados.append(resultado)
    print('======== RESULTADOS DEL TESTING =============')
    import pandas as pd
    df = pd.DataFrame((x_test)).transpose()
    df = df.replace([-1.0], 0.0)
    df = pd.concat([df, pd.DataFrame(resultados), pd.DataFrame(t_expected[lim_training:]).replace([-1.0], 0.0)], axis=1)
    columns = []
    for i in range(len(x_samples)):
        columns.append('x'+str(i))
    columns.append('o')
    columns.append('t')
    df.columns = columns
    print(df)

    correctas = 0
    incorrectas = 0
    temp = []
    for esperado in t_expected:
        if esperado == -1.0:
            temp.append(0)
        else:
            temp.append(esperado)
    t_expected = temp
    for i in range(len(resultados)):
        #print(f'{resultados[i]} == {t_expected[lim_training:][i]} = ')
        if resultados[i] == t_expected[lim_training:][i]:
            correctas += 1
        else:
            incorrectas += 1
    print(correctas, ' ', incorrectas, ' ', correctas/incorrectas)
main()