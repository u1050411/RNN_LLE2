import datetime
import os
import pickle
import shutil

import h5py
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from optuna.pruners import MedianPruner
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

SEMILLA = 42

np.random.seed(SEMILLA)  # Establece la semilla aleatoria de NumPy
tf.random.set_seed(SEMILLA)  # Establece la semilla aleatoria de TensorFlow

# Asegurarse de que se utilice el separador de rutas adecuado para el sistema operativo
ruta_predicciones = os.path.join(".", "predicciones")
ruta_saved_models = os.path.join(".", "saved_models")


class RNNModel:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.input_steps = 18
        self.output_steps = 6
        self.input_steps_categoria = 90
        self.output_steps_categoria = 30
        self.best_model = None
        self.best_trial_value = None
        self.n_features = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.output_column_names = None
        self.input_steps_categoria = 90
        self.output_steps_categoria = 30
        self.fitxerModel = os.path.join("..", "model", "model.h5")
        self.input_file = os.path.join(".", "..", "dades", "Dades_Per_entrenar.csv")
        self.hyperparameter_ranges = {
            "n_layers": [1, 3],
            "num_units_layer": [1, 2],
            "lr": [1e-4, 1e-2],
            "n_epochs": [1, 3],
            "batch_size": [16, 64]
        }

    def create_model(self, n_layers, num_units_layer, lr):
        """
        Crea un modelo LSTM con los hiperparámetros proporcionados.
        """
        n_features = self.x_train.shape[2]
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(num_units_layer, input_shape=(self.input_steps_categoria, n_features), return_sequences=True)))
        model.add(Dropout(0.2))

        for i in range(n_layers - 1):
            model.add(Bidirectional(LSTM(num_units_layer, return_sequences=True)))
            model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(num_units_layer, return_sequences=False)))
        model.add(Dense(self.output_steps_categoria * self.y_train.shape[2], activation='relu'))
        model.add(Reshape((self.output_steps_categoria, self.y_train.shape[2])))

        # Compilar el modelo con el optimizador Adam y la función de pérdida mean_absolute_error
        model.compile(optimizer=Adam(learning_rate=lr), loss=tf.keras.losses.MeanAbsoluteError())

        # Devuelve solo el modelo
        return model

    def objective(self, trial):
        """
        Define el objetivo que Optuna debe minimizar. Este método entrena un modelo con hiperparámetros sugeridos
        por Optuna y devuelve el error absoluto medio ponderado.
        """
        n_layers = trial.suggest_int("n_layers", self.hyperparameter_ranges["n_layers"][0],
                                     self.hyperparameter_ranges["n_layers"][1])
        num_units_layer = trial.suggest_int("num_units_layer", self.hyperparameter_ranges["num_units_layer"][0],
                                            self.hyperparameter_ranges["num_units_layer"][1])
        lr = trial.suggest_float("lr", self.hyperparameter_ranges["lr"][0], self.hyperparameter_ranges["lr"][1],
                                 log=True)
        n_epochs = trial.suggest_int("n_epochs", self.hyperparameter_ranges["n_epochs"][0],
                                     self.hyperparameter_ranges["n_epochs"][1])
        batch_size = trial.suggest_int("batch_size", self.hyperparameter_ranges["batch_size"][0],
                                       self.hyperparameter_ranges["batch_size"][1])

        model = self.create_model(n_layers, num_units_layer, lr)

        # Entrenar el modelo

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
                            validation_data=(self.x_test, self.y_test), callbacks=[early_stopping])

        # Evaluar el modelo en el conjunto de prueba
        y_pred = model.predict(self.x_test)
        # Calcular el error absoluto medio ponderado
        weighted_mae = self.weighted_mean_absolute_error(self.y_test, y_pred)

        # Comprobar si este modelo es el mejor hasta ahora y, de ser así, guardarlo
        if self.best_trial_value is None or weighted_mae < self.best_trial_value:
            study = None
            self.guardar_model(model, study, self.input_file)
            self.best_trial_value = weighted_mae
            self.best_model = model

        return weighted_mae

    def get_error_weights(self, output_steps):
        weights = [1 / (i + 1) for i in range(output_steps)]
        return tf.constant(weights, dtype=tf.float64)

    @tf.function
    def weighted_mean_absolute_error(self, y_true, y_pred):
        """
        Calcula el error absoluto medio ponderado, dando más peso a los errores
        en las primeras etapas de la secuencia de salida.
        """
        y_pred = tf.cast(y_pred, tf.float64)  # Convertir y_pred a float64
        y_true = tf.cast(y_true, tf.float64)  # Convertir y_true a float64
        absolute_errors = tf.math.abs(y_true - y_pred)

        weights = self.get_error_weights(self.output_steps_categoria)
        weights = tf.reshape(weights, (-1, 1))  # Asegurar que weights tenga la forma (-1, 1)
        weighted_absolute_errors = absolute_errors * weights

        return tf.reduce_mean(weighted_absolute_errors)

    def optimize(self, n_trials=50, study=None, initial_params=None):
        """
        Optimiza los hiperparámetros del modelo LSTM utilizando Optuna.
        """
        pruner = MedianPruner()

        if study is None:
            study = optuna.create_study(direction='minimize', pruner=pruner)

        if initial_params is not None:
            # Utilizar los valores iniciales como sugerencias en el espacio de búsqueda de Optuna
            study.enqueue_trial(initial_params)

        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params

        return study.best_params, study

    def predict(self, model=None, input_sequence=None):
        # Realizar la predicción utilizando el modelo entrenado
        if model is None:
            model = load_model(self.fitxerModel)
        y_pred = model.predict(input_sequence)

        n_points = y_pred.shape[1] * y_pred.shape[2]
        y_pred_flat = y_pred.reshape((y_pred.shape[0], n_points))

        # Asegúrate de que el objeto scaler_output tenga la forma correcta
        if self.scaler_output.n_features_in_ != n_points:
            self.scaler_output.n_features_in_ = n_points
            self.scaler_output.min_ = np.tile(self.scaler_output.min_, self.output_steps_categoria)
            self.scaler_output.scale_ = np.tile(self.scaler_output.scale_, self.output_steps_categoria)

        y_pred_inv = self.scaler_output.inverse_transform(y_pred_flat)

        # Reorganizar las columnas de y_pred_inv de acuerdo a las características originales
        y_pred_inv_rearranged = []
        for i in range(0, y_pred_inv.shape[1], self.output_steps_categoria):
            y_pred_inv_rearranged.append(y_pred_inv[:, i:i + self.output_steps_categoria])

        y_pred_inv = np.hstack(y_pred_inv_rearranged)

        return y_pred_inv

    def predict_last_rows(self, model=None, data=None):
        """Ejecutar la predicción del mejor modelo con un fichero y seleccionar las últimas filas."""

        # Cargar los datos del fichero
        if data is None:
            data = self.data.copy()

        data_copiada = data.copy()
        data_copiada = data_copiada.dropna()

        data_limitat = data_copiada.tail(self.input_steps_categoria)

        data_procesada = self.preprocess_data(data_prediccion=data_limitat)

        x_prediccio = self.create_sequences_for_prediction(data_procesada)

        n_features = x_prediccio.shape[2]

        if model is None:
            model = self.best_model
        y_pred = self.predict(model, x_prediccio)

        # Organizar y guardar la predicción en un archivo
        column_names = ['Index'] + [f'Column_{i}' for i in range(1, 4)]
        repeated_numbers = np.tile(np.arange(1, 6), len(y_pred[0]) // (3 * 5) + 1)[:len(y_pred[0]) // 3]
        y_pred_reshaped = np.column_stack((repeated_numbers, y_pred[0][::3], y_pred[0][1::3], y_pred[0][2::3]))
        output_df = pd.DataFrame(y_pred_reshaped, columns=column_names)

        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_file = os.path.join("predicciones", f"primer_semestre{current_time}.xlsx")
        output_df.to_excel(output_file, index=False)
        print(f"Predicción guardada en el archivo: {output_file}")

    def search_and_train_with_optuna(self, n_searches, n_trials_per_search):
        """
        Realiza la búsqueda de los mejores modelos utilizando Optuna y entrena el mejor modelo.
        Guarda el modelo y repite el proceso n_searches veces.

        :param n_searches: Número de veces para repetir el proceso de búsqueda y entrenamiento.
        :param n_trials_per_search: Número de trials por búsqueda en Optuna.
        :param model_save_path: Ruta donde se guardarán los modelos entrenados.
        """
        study = None  # Inicializar el objeto de estudio como None
        best_params = None  # Inicializar los mejores parámetros como None
        model_save_path = os.path.join(".", "..", "save_models")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        for i in range(n_searches):
            print(f"\nBúsqueda y entrenamiento {i + 1} de {n_searches}")

            # Optimizar hiperparámetros con Optuna
            print("Optimizando hiperparámetros...")
            best_params, study = self.optimize(
                n_trials=n_trials_per_search, study=study, initial_params=best_params
            )
            print(f"Mejores hiperparámetros encontrados: {best_params}")



            # Guardar el mejor modelo
            save_path = os.path.join(model_save_path, f"best_model_{i + 1}.h5")
            self.guardar_model(self.best_model, study, self.input_file)
            print(f"Modelo guardado en {save_path}")

    def guardar_model(self, model_save, study, csv_path):
        # Obtenir la data i hora actual amb el format especificat
        data_hora = datetime.datetime.now().strftime('%d%m%Y__%H_%M')

        # Crear la subcarpeta amb el nom de la data i hora
        model_save_path = os.path.join(".", "..", "save_models")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        subcarpeta = os.path.join(os.path.join(model_save_path, data_hora))
        os.makedirs(subcarpeta, exist_ok=True)

        # Guardar el model en un fitxer .h5
        model_path = os.path.join(subcarpeta, "model.h5")
        save_model(model_save, model_path)

        # Extreure els hiperparàmetres del model
        hiperparametres = model_save.get_config()

        # Guardar els hiperparàmetres en un fitxer .pkl
        hiperparametres_path = os.path.join(subcarpeta, "hiperparametres.pkl")

        # Guardar el fitxer CSV amb els resultats de l'estudi
        if study:
            study_path = os.path.join(subcarpeta, "study.csv")
            study.trials_dataframe().to_csv(study_path)
            print(study.best_value)
            print(study.best_params)
            print(study.best_trial)

        with open(hiperparametres_path, 'wb') as f:
            pickle.dump(hiperparametres, f)

        # Copiar el fitxer CSV a la subcarpeta
        shutil.copy2(csv_path, subcarpeta)



