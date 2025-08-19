
from ingestion import *
import pandas as pd

from transformation import date_transformation, limpiar_fecha, numeric_transformation, string_transformation, transform_data
# PATH_LOAD = 'C:/PracticasEscom/Proyecto_CD/docs/'
# PATH_SAVE = 'C:/PracticasEscom/Proyecto_CD/output/'
PATH_LOAD = 'E:/ProyectosPython/ProyectoFinalCD/docs'
PATH_SAVE = 'E:/ProyectosPython/ProyectoFinalCD/output/'

def feature_generation(df: pd.DataFrame) -> pd.DataFrame:
	# Procesar medicamentos (ejemplo: contar numero de medicamentos)
	df[['presion_sistolica', 'presion_diastolica']] = df['presion_arterial'].str.extract(r'(\d+)/(\d+)').astype(float)
	df = drop_cols(df, ['presion_arterial'])
	df['num_medicamentos'] = df['medicamentos'].str.count(r'\|') + 1
	mediana_peso = df['peso'].median()
	# Reemplazar los valores nulos en 'altura' con la mediana
	mediana_altura = df['altura'].median()
	df['peso'] = df['peso'].fillna(mediana_peso)
	df['altura'] = df['altura'].fillna(mediana_altura)
	df['imc'] = df['peso'] / (df['altura'] ** 2)

	return df


def feature_selection(df: pd.DataFrame, vars: list) -> pd.DataFrame:
	# Seleccionar solo las columnas numéricas
	return df[vars]

def target_variable(df: pd.DataFrame) -> pd.Series:
	df['diagnosticos'] = df['diagnosticos'].fillna('')
	df['diabetes'] = df['diagnosticos'].str.contains('diabetes', case=False).astype(int)
	return df

def feature_engineering():
	df = ingest_file(PATH_LOAD + 'MuestraDM.csv')
	df = feature_generation(df)
	numeric_vars = ['glucosa', 'colesterol', 'urea', 'peso', 'altura', 'trigliceridos', 'hba1c',
					'plaquetas', 'creatinina', 'presion_sistolica', 'presion_diastolica', 'num_medicamentos']
	string_vars = ['newid','cx_curp', 'fuente', 'medicamentos', 'codigos_cie']
	date_variables = ['fechas_procesadas']
	categorical_vars = ['sexo', 'bandera_fechas_procesadas', 'hipertension']
	one_hot_columns = ['sexo']
	df = transform_data(df, date_variables, numeric_vars, categorical_vars, string_vars, one_hot_columns)
	df = target_variable(df)
	df = feature_selection(df, numeric_vars + date_variables  + ['sexo_M'] + ['imc'] + ['diabetes'])
	save_df(df, PATH_SAVE + 'DM_features.csv')

feature_engineering()