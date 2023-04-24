import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessing:
    def __init__(self):
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.input_steps = 18
        self.output_steps = 6
        self.input_steps_categoria = 90
        self.output_steps_categoria = 30
        self.fitxerModel = os.path.join('model', 'model.h5')
        self.hyperparameter_ranges = {
            "n_layers": [1, 30],
            "num_units_layer": [16, 200],
            "lr": [1e-4, 1e-2],
            "n_epochs": [100, 300],
            "batch_size": [16, 64]
        }

    @staticmethod
    def set_data():
        archivo_entrada = os.path.join(".", "..", "dades", "Dades_Per_entrenar.csv")
        pre_process = DataPreprocessing()
        data = pre_process.read_data(archivo_entrada)
        return data

    def read_data(self, nomFitxer=None):
        data = pd.read_csv(nomFitxer, sep=";", parse_dates=[0])
        return data

    def preprocess_data(self, data_prediccion=None):
        data_procesada = data_prediccion.copy()
        data_procesada[data_procesada.columns[2:]] = data_procesada.iloc[:, 2:].astype(float)
        data_procesada = pd.get_dummies(data_procesada, columns=['Gran Grup'], prefix='Gran Grup')
        data_procesada.iloc[:, 4:-5] = self.scaler_input.fit_transform(data_procesada.iloc[:, 4:-5])
        data_procesada.iloc[:, 1:4] = self.scaler_output.fit_transform(data_procesada.iloc[:, 1:4])
        data_procesada = data_procesada.drop(data_procesada.columns[0], axis=1)
        data_procesada = data_procesada.dropna()
        return data_procesada

    def split_data(self, data_procesada):
        train_size = int(len(data_procesada) * 0.7)
        train_data = data_procesada[:train_size]
        test_data = data_procesada[train_size:]
        x_train, y_train = self.create_sequences(train_data)
        x_test, y_test = self.create_sequences(test_data)
        return x_train, y_train, x_test, y_test

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.input_steps_categoria - self.output_steps_categoria + 1):
            input_data = data.iloc[i:i + self.input_steps_categoria, [0, *range(4, data.shape[1])]].values
            output_data = data.iloc[
                          i + self.input_steps_categoria:i + self.input_steps_categoria + self.output_steps_categoria,
                          0:3].values
            self.output_column_names = data.columns[3:6].tolist()
            X.append(input_data)
            y.append(output_data)
        return np.array(X), np.array(y)

    def create_sequences_for_prediction(self, data):
        """
        Crea secuencias de datos de entrada a partir del conjunto de datos procesado para realizar predicciones.
        """
        if len(data) < self.input_steps_categoria:
            raise ValueError(
                f"Se requieren al menos {self.input_steps_categoria} registros en los datos para la predicción.")
        input_data = data.iloc[-self.input_steps_categoria:, [0, *range(4, data.shape[1])]].values
        return np.array([input_data])

