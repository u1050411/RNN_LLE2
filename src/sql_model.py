import datetime
import os
import pymysql


class SQLModel:

    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'root'
        self.database = 'bestmodels'
        self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password,
                                          database=self.database)
        self.nomFitxer = os.path.join('model', 'model.h5')

    def guardar(self, model):
        model.save(self.nomFitxer)
        if not os.path.isfile(self.nomFitxer):
            print("El archivo h5 proporcionado no es válido")
            return

        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        with open(self.nomFitxer, 'rb') as file:
            h5_data = file.read()

        try:
            with self.connection.cursor() as cursor:
                sql = "INSERT INTO bestmodels.models (data_model, fitxer_model) VALUES (%s, %s)"
                cursor.execute(sql, (current_datetime, h5_data))
                self.connection.commit()
        finally:
            self.connection.close()

    def recuperar(self):
        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT id, data_model, fitxer_model FROM bestmodels.models ORDER BY id DESC LIMIT 1"
                cursor.execute(sql)
                result = cursor.fetchone()

                if result is None:
                    print("No se encontraron registros en la base de datos.")
                    return

                id_model, model_name, model_file = result

                with open(self.nomFitxer, 'wb') as file:
                    file.write(model_file)

                print(f"El archivo h5 más reciente se ha guardado en: {self.nomFitxer}")
                return self.nomFitxer
        finally:
            self.connection.close()
