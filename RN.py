from time import process_time_ns
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt
import pandas as pd


def readData():
    
    data = pd.read_csv('./dataset.csv')
    aux = []
    aux2 = []
    for i in range(len(data['X1'])):
        aux.append(data['X1'][i])
        aux2.append(data['Y'][i])
    aux = numpy.array(aux, dtype=float)
    aux2 = numpy.array(aux2, dtype=float)

    return [aux, aux2]

def neurona():
    data = readData()
    
    # celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    # fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
    capa = keras.layers.Dense(units= 1, input_shape=[1], activation='selu')
    modelo = keras.Sequential([capa])

    modelo.compile(
        optimizer=keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    print('Entrenamiento')

    historial = modelo.fit(data[0], data[1], epochs=1000, verbose= False)

    print('Modelo entrenado')
  
    print('Historia:')
    print(historial.history['loss'])
    
    plt.xlabel('# Epoca')
    plt.ylabel('Magnitud de perdida')
    # plt.plot([1,2,3,4,5,6])
    plt.plot(historial.history['loss'])
 

    plt.show()


    print('Prediccion')

    resultado = modelo.predict([72])

    print(f'Resultado es {resultado}')

if __name__ == '__main__':
    neurona()
    # readData()