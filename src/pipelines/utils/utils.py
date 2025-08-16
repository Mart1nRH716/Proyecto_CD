import pickle
import pandas as pd


'''
Función que carga un objeto de tipo DataFrame desde un archivo CSV.
Recibe:
	filename (str): Ruta del archivo CSV a cargar.
Regresa:
	DataFrame: Un DataFrame de pandas cargado desde el archivo CSV.
'''	
def load_df(filename):
	return pd.read_csv(filename)


'''Función que guarda un objeto de tipo DataFrame en un archivo CSV.
Recibe:
	df (DataFrame): El DataFrame a guardar.
	filename (str): Ruta del archivo CSV donde se guardará el DataFrame.
Regresa:
	None
'''
def save_df(df, filename):
	df.to_csv(filename, index=False)


def funcion_prueba():
	print("Esta es una función de prueba en utils.py")

if __name__ == "__main__":
	funcion_prueba()
	