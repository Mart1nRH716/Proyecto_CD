from utils.utils import *
from utils.utils import load_df, save_df


'''
Función que ingesta un archivo CSV y devuelve un DataFrame con los títulos de las columnas normalizadas.
Recibe:
	file_path (str): Ruta del archivo CSV a ingestar.
Regresa:
	DataFrame: Un DataFrame de pandas cargado desde el archivo CSV.
'''
def ingest_file(file_path):
    df = load_df(file_path)
    df.columns = df.columns.str.strip().str.lower()
    return df


''''
Función que guarda un DataFrame en un archivo CSV después de la ingesta.
Recibe:
	df (DataFrame): El DataFrame a guardar.
	output_path (str): Ruta del archivo CSV donde se guardará el DataFrame.
Regresa:
	None
'''
def save_ingested_data(df, output_path):
	save_df(df, output_path)


'''Función que carga un DataFrame desde un archivo CSV después de la ingesta.
Recibe:
	file_path (str): Ruta del archivo CSV a cargar.
Regresa:
	DataFrame: Un DataFrame de pandas cargado desde el archivo CSV.
'''
def load_ingested_data(file_path):
	return load_df(file_path)


"""
Función que elimina columnas específicas de un DataFrame.
Recibe:
	df (DataFrame): El DataFrame del cual se eliminarán las columnas.
	cols (list): Lista de nombres de columnas a eliminar.
Regresa:
	DataFrame: El DataFrame sin las columnas especificadas.
"""
def drop_cols(df, cols):
	return df.drop(columns=cols, errors='ignore') if cols else df


# """
# 	Función que genera etiquetas a partir de un DataFrame.
# 	Recibe:
# 		df (DataFrame): El DataFrame del cual se generarán las etiquetas.
# 	Regresa:
# 		DataFrame: Un nuevo DataFrame con las etiquetas generadas.
# 	"""
# def generate_labels(df):
	
# 	return df
