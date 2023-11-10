import datetime
import os
import pickle
import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.sql_model import SQLModel
from src.model import RNNModel
from src.data_preprocessing import DataPreprocessing

class Opciones:
    def __init__(self):
        self.opcion = 0
        self.archivo = os.path.join(".", "..", "dades", "Dades_Per_entrenar.csv")
        self.hiperparametros = ''
        self.modelo = ''
        self.directorio = ''
        # self.model_sql = SQLModel()
        self.input_steps_categoria = 90
        self.output_steps_categoria = 30

    def prediccion_carpeta(self, directorio=None):
        archivo_csv, archivo_pkl, archivo_h5 = '', '', ''
        for archivo in os.listdir(directorio):
            ruta = os.path.join(directorio, archivo)
            if archivo.endswith('.csv') and not archivo_csv:
                archivo_csv = ruta
            elif archivo.endswith('.pkl') and not archivo_pkl:
                with open(ruta, 'rb') as f:
                    archivo_pkl = pickle.load(f)
            elif archivo.endswith('.h5') and not archivo_h5:
                archivo_h5 = ruta

        rnn = RNNModel(archivo_csv)
        rnn.best_model = load_model(archivo_h5)
        rnn.best_model.set_weights(archivo_pkl)
        rnn.split_data(rnn.preprocess_data())
        rnn.predict_last_rows()

    def guardar_modelo(self, final_model=None):
        sql = SQLModel()
        sql.guardar(final_model)
        print("Modelo guardado")

    def recuperar_modelo(self):
        sql = SQLModel()
        modelo = sql.recuperar()
        return modelo

    def utilizar_optuna(self, n_trials=1):

        # Creando la ruta al archivo 'Dades_Per_Prediccio.csv' en la carpeta 'dades'
        pre_process = DataPreprocessing()
        data = pre_process.read_data(self.archivo)
        data_procesada = pre_process.preprocess_data(data)
        # Creación de variables para los conjuntos de datos divididos
        x_train, y_train, x_test, y_test = pre_process.split_data(data_procesada)
        # Creación de una instancia de RNNModel con los conjuntos de datos divididos
        rnn = RNNModel(x_train, y_train, x_test, y_test)
        # Entrenamiento del modelo con los hiperparámetros óptimos
        best_params = rnn.optimize(n_trials=n_trials)
        print(f"Mejores hiperparámetros encontrados: {best_params}")
        print(best_params)
        final_model = rnn.best_model
        print("Guardando el modelo...")

        # Asegúrate de que la carpeta 'model' exista
        model_folder = os.path.join(".", "..", "model")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # Guarda el modelo en formato HDF5
        model_path = os.path.join(model_folder, "model.h5")
        final_model.save(model_path)
        print(f"Modelo guardado en {rnn.fitxerModel}")
        return final_model

    def predict(self,data_proces, model, input_sequence):
        # Realizar la predicción utilizando el modelo entrenado
        y_pred = model.predict(input_sequence)

        n_points = y_pred.shape[1] * y_pred.shape[2]
        y_pred_flat = y_pred.reshape((y_pred.shape[0], n_points))

        # Asegúrate de que el objeto scaler_output tenga la forma correcta
        if data_proces.scaler_output.n_features_in_ != n_points:
            data_proces.scaler_output.n_features_in_ = n_points
            data_proces.scaler_output.min_ = np.tile(data_proces.scaler_output.min_, self.output_steps_categoria)
            data_proces.scaler_output.scale_ = np.tile(data_proces.scaler_output.scale_, self.output_steps_categoria)

        y_pred_inv = data_proces.scaler_output.inverse_transform(y_pred_flat)

        # Reorganizar las columnas de y_pred_inv de acuerdo a las características originales
        y_pred_inv_rearranged = []
        for i in range(0, y_pred_inv.shape[1], self.output_steps_categoria):
            y_pred_inv_rearranged.append(y_pred_inv[:, i:i + self.output_steps_categoria])

        y_pred_inv = np.hstack(y_pred_inv_rearranged)

        return y_pred_inv

    def predict_last_rows(self, data_proces=None, model=None, menys_filas=0):
        """Ejecutar la predicción del mejor modelo con un fichero y seleccionar las últimas filas."""
        if model is None:
            model = self.best_model

        data = data_proces.read_data(self.archivo)
        data_copiada = data.copy()
        data_copiada = data_copiada.dropna()

        # Obtiene las últimas filas usando tail y elimina las últimas menys_filas si es necesario
        data_limitat = data_copiada.tail(self.input_steps_categoria+menys_filas).iloc[:-menys_filas if menys_filas > 0 else None]

        data_procesada = data_proces.preprocess_data(data_prediccion=data_limitat)

        x_prediccio = data_proces.create_sequences_for_prediction(data_procesada)

        n_features = x_prediccio.shape[2]

        y_pred = self.predict(data_proces, model, x_prediccio)

        # Organizar y guardar la predicción en un archivo
        column_names = ['Index'] + [f'Column_{i}' for i in range(1, 4)]
        repeated_numbers = np.tile(np.arange(1, 6), len(y_pred[0]) // (3 * 5) + 1)[:len(y_pred[0]) // 3]
        y_pred_reshaped = np.column_stack((repeated_numbers, y_pred[0][::3], y_pred[0][1::3], y_pred[0][2::3]))
        output_df = pd.DataFrame(y_pred_reshaped, columns=column_names)

        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_file = os.path.join(".", "..", "predicciones", f"predicciones_{current_time}.xlsx")
        output_df.to_excel(output_file, index=False)
        print(f"Predicción guardada en el archivo: {output_file}")

    def optuna_iterativo(self, n_busquedas=None, n_pruebas_por_busqueda=None, ruta_guardar_modelo=None):
        data_process = DataPreprocessing()
        data = data_process.read_data(self.archivo)
        data_procesada = data_process.preprocess_data(data)
        x_train, y_train, x_test, y_test = data_process.split_data(data_procesada)

        rnn = RNNModel(x_train, y_train, x_test, y_test)
        rnn.search_and_train_with_optuna(n_busquedas, n_pruebas_por_busqueda)


if __name__ == '__main__':
    opciones = Opciones()
    # ruta_carpeta = os.path.join("C:", "Users", "u1050", "PycharmProjects", "RNN_LLE", "for_model", "Cat1_17042023", "model", "18042023__10_18")
    #opciones.prediccion_carpeta(directorio=ruta_carpeta)
    # model = opciones.utilizar_optuna(n_trials=2)
    # data_proces = DataPreprocessing()
    # opciones.predict_last_rows(data_proces, model)
    opciones.optuna_iterativo(n_busquedas=1000, n_pruebas_por_busqueda=1000)

