import os
import pickle
import h5py
from tensorflow.keras.models import load_model
from src.sql_model import SQLModel
from src.model import RNNModel

class Opciones:
    def __init__(self):
        self.opcion = 0
        self.archivo = ''
        self.hiperparametros = ''
        self.modelo = ''
        self.directorio = ''
        self.model_sql = SQLModel()

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

    def utilizar_optuna(self):
        archivo_entrada = os.path.join("dades", "Dades_Per_entrenar.csv")
        rnn = RNNModel(archivo_entrada)
        rnn.split_data(rnn.preprocess_data())

        mejores_parametros = rnn.optimizar(n_trials=1)
        print(f"Mejores hiperparámetros encontrados: {mejores_parametros}")
        modelo_final = rnn.best_model
        modelo_final.save(rnn.archivo_modelo)
        print(f"Modelo guardado en {rnn.archivo_modelo}")

    def predecir_modelo(self):
        archivo_entrada = os.path.join("dades", "Dades_Per_entrenar.csv")
        rnn = RNNModel(archivo_entrada)
        rnn.split_data(rnn.preprocess_data())
        rnn.predict_last_rows()

    def optuna_iterativo(self, n_busquedas=None, n_pruebas_por_busqueda=None, ruta_guardar_modelo=None):
        archivo_entrada = os.path.join("dades", "Dades_Per_entrenar.csv")
        rnn = RNNModel(archivo_entrada)
        ruta_guardar_modelo = os.path.join("saved_models")
        rnn.buscar_y_entrenar_con_optuna(n_busquedas, n_pruebas_por_busqueda, ruta_guardar_modelo)


if __name__ == '__main__':
    opciones = Opciones()
    ruta_carpeta = os.path.join("C:", "Users", "u1050", "PycharmProjects", "RNN_LLE", "for_model", "Cat1_17042023", "model", "18042023__10_18")
    opciones.prediccion_carpeta(directorio=ruta_carpeta)

