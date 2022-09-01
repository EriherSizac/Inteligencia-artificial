"""
Código que tiene la función de entrenar un perceptron
Autor: Erick Hernández Silva
Fecha de creación: 24/08/2022
<<<<<<< Updated upstream
Última actualización: 27/08/2022
=======
Última actualización: 29/08/2022
"""
import random

import numpy as np
import pandas as pd

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
training_data = .5

def read_csv(file):
    """
    Función que lee el archivo de inputs y regresa todas las entradas
    :return: todas las entradas en una lista
    """
    # Procesamos los datos del csv
    df = pd.read_csv(file+'.csv')
    return df



def calcular_outputs(df, weights):
    """
    Función que calcula los outputs
    :param df: dataframe sobre el que vamos a calcular los outputs
    :param weights: un array de pesos
    :return:
    """
    outputs = []
    # Calculamos los outputs con nuestra función signo
    number_of_samples = df.shape[1]
    sample_index = 0
    for iteration in range(0, df.shape[0]):
        temp_list_x = []
        temp_list_w = []
        for sample in range(0,number_of_samples):
            temp_list_x.append([sample][sample_index])
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
    df = read_csv(input('Nombre del archivo: '))
    learning_rate = input('Introduce el learning rate: ')
    training_percentage = int(input('Introduce el porcentaje de los datos que quieres usar como entrenamiento (Ejemplo: 70): '))
    training_percentage /= 100

    # Se hace el shuffle a los datos y se resetean los indices
    df = df.sample(frac=1).reset_index(drop=True)
    # Separamos las muestras de entrenamiento y prueba
    lim_training = int(df.shape[0] * training_percentage)

    df_training = df.iloc[:lim_training]
    df_test = df.iloc[lim_training:]


    # Inicialiamos los pesos iniciales en 0.1
    weights = pd.Series([0.1] * (df_training.shape[1]-1))
    print(weights)


    outputs = calcular_outputs(df_training, weights)
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
    accuracy = (correctas * 100 )/ len(x_test[0])
    print(correctas, ' ', incorrectas, ' ', accuracy)
main()