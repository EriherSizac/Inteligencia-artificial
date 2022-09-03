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
    cols = ['x_'+str(i) for i in range(0,df.shape[1]-1)]
    cols.append('t')
    df.columns = cols
    df.replace(0, -1, inplace=True)
    return df



"""def calcular_outputs(df, weights):
    """""""
    Función que calcula los outputs
    :param df: dataframe sobre el que vamos a calcular los outputs
    :param weights: un array de pesos
    :return:
    """"""
    outputs = []
    # Calculamos los outputs con nuestra función signo
    number_of_samples = df.shape[1]
    sample_index = 0
    for iteration in range(0, df.shape[0]):
        temp_list_x = []
        temp_list_w = []
        for sample in range(0,number_of_samples):
            temp_list_x.append(df[sample][sample_index])
            temp_list_w.append(weights[sample])
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
    return temp_weights_x"""
def calcular_outputs(x,w):
    result = []
    x= x.to_numpy()
    w = pd.DataFrame(w).transpose()
    w = w.loc[w.index.repeat(x.shape[0])]
    w = w.to_numpy()

    for i in range(w.shape[0]):
        res = np.sign(np.dot(x[i], w[i]))
        result.append(res)
    return pd.Series(result)

def calcular_pesos(w,learning_rate,t,o,x):
    """
    Funcion que calcula los pesos
    :param w: pesos actuales
    :param learning_rate: el learning rate dado por el usuario
    :param t: resultados esperados
    :param o: outputs obtenidos
    :param x: dataframe de x
    :return: un pandas series de pesos
    """
    x = x.to_numpy()
    for i in range(w.shape[0]):
        result = w[i]+learning_rate*(t[i]-o[i])*x[i]
        w = pd.Series(result)

    return w


def main():
    import pandas as pd
    # Hacemos el setup inicial
    df = read_csv(input('Nombre del archivo: '))
    learning_rate = float(input('Introduce el learning rate: '))
    training_percentage = int(input('Introduce el porcentaje de los datos que quieres usar como entrenamiento (Ejemplo: 70): '))
    """df = read_csv('test')
    learning_rate = 0.3
    training_percentage = 70"""
    training_percentage /= 100

    # Se hace el shuffle a los datos y se resetean los indices
    df = df.sample(frac=1).reset_index(drop=True)
    # Separamos las muestras de entrenamiento y prueba
    lim_training = int(df.shape[0] * training_percentage)

    df_training = df.iloc[:lim_training]
    df_test = df.iloc[lim_training:]

    # Sacamos las t esperadas
    t_training = df_training['t']
    t_test = df_test['t']
    # Quitamos del entrenamiento los resultados
    df_training.drop('t', axis=1, inplace=True)
    df_test.drop('t', axis=1, inplace=True)

    # Inicialiamos los pesos iniciales en 0.1
    weights = pd.Series([0.1] * (df_training.shape[1]))


    outputs = calcular_outputs(df_training, weights)
    # Comenzamos a hacer las corridas hasta que t = o
    if outputs.tolist() != t_training.tolist():
        accuracies = []
        accuracy = ((outputs == t_training).sum() * 100) / t_training.shape[0]
        cycle = True
        while cycle:
            weights = calcular_pesos(weights,learning_rate, t_training, outputs,df_training)
            outputs = calcular_outputs(df_training,weights)
            """print(outputs.head())
            print(t_training.head())"""
            accuracy = ((outputs == t_training).sum() * 100) / t_training.shape[0]
            accuracies.append(accuracy)
            for nac in accuracies:
                if accuracies.count(nac) > 6:
                    cycle = False
                    print("Max accuracy reached with training sample: ", nac)
                    break
    test = calcular_outputs(df_test, weights)
    print('======== RESULTADOS DEL TESTING =============')
    accuracy = ((test == t_test.reset_index(drop=True)).sum() * 100) / t_test.shape[0]
    print('Accuracy with test sample: ', accuracy)
main()