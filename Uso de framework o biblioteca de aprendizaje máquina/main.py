import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
def open(name):
    """
    Función que abre un archivo.
    :param name: nombre del archivo con extensión
    :return:
    """
    df = pd.read_csv(name)
    return df

def data_to_split(df, y_column, x_columns=None):
    if x_columns is None or x_columns == ['']:
        X = df.drop(y_column, axis=1)
    else:
        X = df[x_columns]
    y = df[y_column]

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state= 1)
    return xTrain, xTest, yTrain, yTest

if __name__ == '__main__':
    # Configuración del usuario
    # Nombre del dataset con extension a usar
    dataset_name = input('Ingresa el nombre del csv a usar: ')
    # Abrimos el csv y lo guardamos en un DataFrame
    df = open(dataset_name + '.csv')
    # Imprimimos las columnas
    print(f'Estas son las columnas del dataset: \n {df.columns}')
    # Columna de resultados Y
    y_column = input('Nombre de la columna de resultados del data set \n¿Qué es lo que quieres '
                     'predecir?: ')
    # Columnas de X que vamos a usar
    x_columns = input('¿Qué columnas de X quieres usar? Ingresalas separadas por comas o \n '
                      'Deja vacío para usar todas menos '
                      'la columna y: ')
    x_columns = x_columns.split(',')
    # Hacemos split de los datos de entrenamiento y de prueba
    xTrain, xTest, yTrain, yTest = data_to_split(df, y_column, x_columns)
    # Añadimos las constantes a las X de train y test
    xTrain = sm.add_constant(xTrain)
    xTest = sm.add_constant(xTest)
    # Hacemos un fit al model para entrenarlo
    model = sm.OLS(yTrain, xTrain).fit()
    # Hacemos un testeo con los datos de prueba
    results = model.predict(xTest)
    # Mostramos el resumen del modelo
    print(model.summary())
    # Mostramos el MSE de los resultados
    print('Mean squared error: ', mean_squared_error(yTest, results))
    # Grafica de predichos vs esperados
    plt.figure(figsize=(10, 10))
    plt.plot(yTest.reset_index(drop=True), label='Valores esperados')
    plt.plot(results.reset_index(drop=True), label='Valores predichos')
    plt.legend()
    plt.xlabel('Entrada')
    plt.ylabel('Resultado')
    plt.show()
