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
# Porcentaje de datos de entrenamiento
training_data = .6

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
        for sample in range(0,number_of_samples-1):
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
    #df = pd.DataFrame(x_samples)
    #df = df.transpose()
    #df_columns = ['x_' + str(i) for i in range(0,len(x_samples))]
    #weights = pd.DataFrame(weights)
    #weights = weights.transpose()
    #weights_columns = ['w_' + str(i) for i in range(0,len(x_samples))]
    #df = pd.concat([df, weights, pd.DataFrame(t_expected), pd.DataFrame(outputs)], axis=1, ignore_index=True)
    #df.columns = df_columns + weights_columns + ['t','o']
    for i in range(0, len(x)):
        #for j in range(0, df.shape[0]):
        for j in range(0, len(x[0])):
            # Python vainilla
            temp_weights_x[i].append(weights[i][j] + (learning_rate * (t_expected[j] - outputs[j]) * x[i][j]))
            # Usando dataframes
            #temp_weights_x[i].append(df['w_'+str(i)][j] + (learning_rate * (df['t'][j] - df['o'][j]) * df['x_'+str(i)][j]))
    return temp_weights_x



def main():
    # Hacemos el setup inicial
    entries = read_txt(input('Nombre del archivo: '))
    # Colocamos los valores en sus respectivas entradas
    x_samples = entries[0:-2]
    t_expected = entries[-2]
    learning_rate = entries[-1]

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
            t_training.append(t_expected[j])
        for j in range(lim_training,len(x_samples[0])):
            x_test[i].append(x_samples[i][j])
            t_test.append(t_expected[j])
    #x_training = x_samples[0:lim_training]
    #x_test = x_samples[lim_training:-1]
    #t_training = t_expected[0:lim_training]
    #t_test = t_expected[lim_training:-1]

    # Inicializamos los pesos iniciales en 0.1
    num_muestras = len(x_training[0])
    num_variables_x = len(x_training)
    weights = [[0.1] * num_muestras] * (num_variables_x)
    # Inicializamos los cambios en los pesos en ceros
    delta_weights = [[0.0] * num_muestras] * (num_variables_x)
    outputs = calcular_outputs(x_training, weights)
    print('=========== DATOS INICIALES ===========')
    print('Muestras x')
    print(x_training)
    print('Valores esperados')
    print(t_test)
    print('Pesos iniciales')
    print(weights)
    print('Outputs obtenidos con los pesos')
    print(outputs)
    print('Learning rate: ', learning_rate)
    print('========================================')
    # Comenzamos a hacer las corridas hasta que t = o
    if outputs != t_expected:
        while outputs != t_expected:
            weights = calcular_pesos(weights, x_training, t_expected, outputs)
            outputs = calcular_outputs(x_training,weights)
        print('=========== RESULTADOS ===========')
        print('Los pesos son: ')
        print(weights)
        print('Los outputs recibidos con estos pesos Son:')
        print(outputs)
        print('========================================')
    # Probamos las muestras para testing
    #x_test.insert(0,x_training[0])
    resultados = calcular_outputs(x_test, weights)
    print('======== RESULTADOS DEL TESTING =============')
    import pandas as pd
    df = pd.DataFrame((x_test)).transpose()
    #df= df.transpose()
    #df = pd.concat([df, pd.DataFrame(t_test)], axis=1)
    #df = pd.concat([df, pd.DataFrame(resultados)], axis=1)
    print(pd.concat([df, pd.DataFrame(resultados)], axis=1))
    #print(len(resultados))
main()