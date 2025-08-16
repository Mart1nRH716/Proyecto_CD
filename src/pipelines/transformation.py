import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from ingestion import ingest_file

meses_es = {
    'ENERO': '01', 'FEBRERO': '02', 'MARZO': '03', 'ABRIL': '04', 'MAYO': '05',
    'JUNIO': '06', 'JULIO': '07', 'AGOSTO': '08', 'SEPTIEMBRE': '09',
    'OCTUBRE': '10', 'NOVIEMBRE': '11', 'DICIEMBRE': '12',
    'SEPT': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12',
    'SEP': '09', 'ENEROD': '01', 'NERO': '01', 'E ENERO': '01'
}

def limpiar_fecha(fecha):
    if pd.isna(fecha) or not isinstance(fecha, str):
        return np.nan
    f = fecha.upper()
    f = re.sub(r'[^A-Z0-9/ ]', ' ', f)  #simbolos
    f = re.sub(r'\s+', ' ', f).strip() #espacio
    for mes, num in meses_es.items():
        f = re.sub(r'\b' + mes + r'\b', num, f)
    f = re.sub(r'\bDE\b|\bDEL\b', '', f)
    f = re.sub(r'\s+', '/', f)  # cambiar a formato tipo dd/mm/yyyy
    return f
	

"""
	Transforma una columna de fecha en el DataFrame a un formato estandarizado.

	Recibe:
		df (pd.DataFrame): El DataFrame que contiene la columna de fecha.
		cols_date (list): Lista de nombres de columnas de fecha a transformar.

	Retorna:
		pd.DataFrame: El DataFrame con las columnas de fecha transformadas.
"""
def date_transformation(df, cols_date):
	for date_column in cols_date:
		if date_column not in df.columns:
			raise ValueError(f"Column '{date_column}' does not exist in the DataFrame.")
	for var in cols_date:
		df[var] = df[var].apply(limpiar_fecha)
	# Convierte la columna de fecha a tipo datetime, manejando errores
	df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

	return df

"""
	Función que trasnforma columnas numéricas a un formato estandarizado.
	Recibe:
		df (pd.DataFrame): El DataFrame que contiene las columnas numéricas.
		cols_numericas (list): Lista de nombres de columnas numéricas a transformar.
	Retorna:
		pd.DataFrame: El DataFrame con las columnas numéricas transformadas.
"""
def numeric_transformation(df, cols_numericas):
	if not isinstance(cols_numericas, list):
		raise ValueError("cols_numericas debe ser una lista de nombres de columnas.")

	for col in cols_numericas:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
		
		# Convierte la columna a tipo numérico, manejando errores
		df[col] = pd.to_numeric(df[col], errors='coerce')

	return df


"""	Función que transforma columnas categóricas a un formato estandarizado.
	Recibe:
		df (pd.DataFrame): El DataFrame que contiene las columnas categóricas.
		cols_categoricas (list): Lista de nombres de columnas categóricas a transformar.
	Retorna:
		pd.DataFrame: El DataFrame con las columnas categóricas transformadas.
"""
def categorical_transformation(df, cols_categoricas):
	if not isinstance(cols_categoricas, list):
		raise ValueError("cols_categoricas debe ser una lista de nombres de columnas.")

	for col in cols_categoricas:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
		
		# Convierte la columna a tipo categórico
		df[col] = df[col].astype('category')

	return df

"""Función que transforma columnas de tipo string a un formato estandarizado.
Recibe:			
	df (pd.DataFrame): El DataFrame que contiene las columnas de tipo string.
	cols_strings (list): Lista de nombres de columnas de tipo string a transformar.
Regresa:
	pd.DataFrame: El DataFrame con las columnas de tipo string transformadas.
"""
def string_transformation(df, cols_strings):
	if not isinstance(cols_strings, list):
		raise ValueError("cols_strings debe ser una lista de nombres de columnas.")

	for col in cols_strings:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
		
		# Convierte la columna a tipo string
		df[col] = df[col].astype('string')

	return df
"""Función que escala las columnas numéricas de un DataFrame.
Recibe:
	df (pd.DataFrame): El DataFrame que contiene las columnas numéricas.
	cols_numericas (list): Lista de nombres de columnas numéricas a escalar.
Regresa:
	pd.DataFrame: El DataFrame con las columnas numéricas escaladas.
"""
def scale_numeric_data(df, cols_numericas):
	if not isinstance(cols_numericas, list):
		raise ValueError("cols_numericas debe ser una lista de nombres de columnas.")
	scaler = StandardScaler()
	df[cols_numericas] = scaler.fit_transform(df[cols_numericas])
	return df

"""
	Guarda el DataFrame transformado en un archivo CSV.

	Recibe:
		df (pd.DataFrame): El DataFrame a guardar.
		output_path (str): Ruta del archivo CSV donde se guardará el DataFrame.

	Regresa:
		None
"""
def save_transformed_data(df, output_path):
	ruta = "E:/ProyectosPython/ProyectoFinalCD/output/" + output_path
	df.to_csv(ruta, index=False)


def one_hot_column(df, cols_categoricas):
	if not isinstance(cols_categoricas, list):
		raise ValueError("cols_categoricas debe ser una lista de nombres de columnas.")

	for col in cols_categoricas:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
		encoder = OneHotEncoder(sparse_output=False, drop='first')
		col_encoded = encoder.fit_transform(df[[col]])
		col_encoded_df = pd.DataFrame(col_encoded, columns=encoder.get_feature_names_out([col]))
		df = pd.concat([df, col_encoded_df], axis=1).drop(columns=[col])
	return df


def transform_data(df, cols_date, cols_numericas, cols_categoricas, cols_strings, one_hot_columns):
	df = date_transformation(df, cols_date)
	df = numeric_transformation(df, cols_numericas)
	df = categorical_transformation(df, cols_categoricas)
	df = scale_numeric_data(df, cols_numericas)
	df = string_transformation(df, cols_strings)
	df = one_hot_column(df, one_hot_columns)
	return df

