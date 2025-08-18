import os
import pickle

def load_model(models_paths = 'C:/PracticasEscom/Proyecto_CD/models'):
    #Buscamos todos los archivos .pkl en la ruta especificada
    models = {}
    for filename in os.listdir(models_paths):
        if filename.endswith('.pkl'):
            model_name = filename[:-4]
            models[model_name] = load_single_model(os.path.join(models_paths, filename))
    return models

def load_single_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def metric_evaluation(models, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        score = model.score(X_test, y_test)
        results[model_name] = score
        print(f"Model: {model_name}, Score: {score}")
    return results

def save_evaluation_results(results, output_path='C:/PracticasEscom/Proyecto_CD/output/evaluation_results.txt'):
    with open(output_path, 'w') as file:
        for model_name, score in results.items():
            file.write(f"Model: {model_name}, Score: {score}\n")
    print(f"Evaluation results saved to {output_path}")
    

