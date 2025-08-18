import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.utils import load_df
import seaborn as sns
from model_evaluation import *

#PATH = 'E:/ProyectosPython/ProyectoFinalCD/output/DM_features.csv'
PATH_LOAD = 'C:/PracticasEscom/Proyecto_CD/output/'
PATH_SAVE = 'C:/PracticasEscom/Proyecto_CD/models/'

algorithms_dict = {
		'tree': 'tree_grid_search',
		'random_forest': 'rf_grid_search',
		'logistic_regression': 'logistic_regression_grid_search',
		'naive_bayes': 'naive_bayes_grid_search',
		'k_nearest_neighbors': 'k_nearest_neighbors_grid_search',
	}
grid_search_dict = {
		'tree_grid_search': {'max_depth': [5,10,15,None], 'min_samples_leaf': [3,5,7]},
		'rf_grid_search': {'n_estimators': [100,300,500,800,1000], 'max_depth': [5,10,15,20,None], 'min_samples_leaf': [3,5,7,11]},
		'logistic_regression_grid_search': {'C': [0.1, 1, 10, 100]},
		'naive_bayes_grid_search': {'alpha': [0.1, 1, 10]},
		'k_nearest_neighbors_grid_search': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
	}
estimators_dict = {'tree': DecisionTreeClassifier(random_state=1111),
				  'random_forest': RandomForestClassifier(oob_score=True, random_state=2222),
				  'logistic_regression': LogisticRegression(random_state=3333),
				  'naive_bayes': GaussianNB(),
				  'k_nearest_neighbors': KNeighborsClassifier()}
	


def load_features(path= PATH_LOAD + 'DM_features.csv'):
	df = load_df(path)
	return df

def magic_loop(algorithms, features, labels, cv_strategy):
	best_estimators = []
	for algorithm in algorithms:
		estimator = estimators_dict[algorithm]
		grid_search_to_look = algorithms_dict[algorithm]
		grid_params = grid_search_dict[grid_search_to_look]
		
		gs = GridSearchCV(estimator, grid_params, scoring='precision', cv=cv_strategy, n_jobs=-1)
		
		#train
		gs.fit(features, labels)
		#best estimator
		best_estimators.append(gs)
		
		
	return best_estimators

def save_models(models, path=PATH_SAVE):
	for model in models:
		algorithm_name = model.estimator.__class__.__name__
		file_path = f"{path}{algorithm_name}.pkl"
		with open(file_path, 'wb') as file:
			pickle.dump(model, file)
		print(f"Model {algorithm_name} saved to {file_path}")


def modeling_pipeline(X_train, y_train, df):
	tscv = TimeSeriesSplit(n_splits=5)
	best_estimators = {}
	algorithms_dict = ['tree', 'random_forest', 'logistic_regression', 'k_nearest_neighbors']


	best_estimators = magic_loop(algorithms_dict, X_train, y_train, tscv)
	save_models(best_estimators)


def analisis_faltantes():
    # Porcentaje de valores faltantes
    df = load_features()
    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, palette="Reds_r")
    plt.title('Porcentaje de Valores Faltantes por Variable')
    plt.xlabel('Porcentaje faltante (%)')
    plt.ylabel('Variables')
    plt.show()
    
    return missing

# analisis_faltantes()
df = load_features()
df = df.sort_values(by="fechas_procesadas").reset_index(drop=True)
# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
test_size = int(len(df) * 0.2)
train, test = df.iloc[:-test_size], df.iloc[-test_size:]
X_train, y_train = train.drop(["diabetes", "fechas_procesadas"], axis=1), train["diabetes"]
X_test, y_test = test.drop(["diabetes", "fechas_procesadas"], axis=1), test["diabetes"]
modeling_pipeline(X_train, y_train, df)
print("Modeling pipeline completed successfully.")
models = load_model()
resultados = metric_evaluation(models, X_test, y_test)
save_evaluation_results(resultados)