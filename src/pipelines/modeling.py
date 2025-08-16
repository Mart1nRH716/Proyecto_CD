import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from pipelines.utils.utils import load_df


def load_features(path='E:/ProyectosPython/ProyectoFinalCD/output/DM_features.csv'):
	df = load_df(path)
	return df

def magic_loop(algorithms, features, labels):
	algorithms_dict = {
		'tree': 'tree_grid_search',
		'random_forest': 'rf_grid_search'
	}
	grid_search_dict = {
		'tree_grid_search': {'max_depth': [5,10,15,None], 'min_samples_leaf': [3,5,7]},
		'rf_grid_search': {'n_estimators': [100,300,500,800,1000], 'max_depth': [5,10,15,20,None], 'min_samples_leaf': [3,5,7,11]}
	}
	estimators_dict = {'tree': DecisionTreeClassifier(random_state=1111),
				  'random_forest': RandomForestClassifier(oob_score=True, random_state=2222)}
	
	best_estimators = []
	for algorithm in algorithms:
		estimator = estimators_dict[algorithm]
		grid_search_to_look = algorithms_dict[algorithm]
		grid_params = grid_search_dict[grid_search_to_look]
		
		gs = GridSearchCV(estimator, grid_params, scoring='precision', cv=5, n_jobs=-1)
		
		#train
		gs.fit(features, labels)
		#best estimator
		best_estimators.append(gs)
		
		
	return best_estimators
