import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from processing.load_data import load_california_housing_data
from training.model import initialize_model
import yaml

### agregar src al path



### 1. Cargar configuracion y el modelo
def load_model_and_data(experiment_config_path):
    """
    Cargar el modelo entrenado y los datos de un experimento especifico

    Args:
        experiment_config_path (str): Ruta al archivo de configuración del experimento (YAML)
    """

    if not Path(experiment_config_path).exists():
        raise FileNotFoundError(f"No se encuentra el archivo de configuración: {experiment_config_path}")

    with open(experiment_config_path, "r") as f:
        config_data = yaml.safe_load(f)  # Carga el archivo de configuración a un diccionario python


### Cargar datos
    _, _, X_test, _, _, y_test, _, feature_names = load_california_housing_data(
        test_size=config_data["training_params"]["test_size"],
        random_state=config_data["training_params"]["random_state"],
        feature_engineering=True,
        PolinomialFeatures=config_data.get("use_polynomial_features", False),
    )

    ### Inicializar el modelo
    model = initialize_model(config_data)

    ### cargar pesos del modelo entrenado
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / config_data["model_config"]["model_name"]
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, X_test, y_test, feature_names, config_data

### 2. funcion de analisis
def analyze_with_shap(model, X_test, feature_names, save_dir="reports/shap"):
    """
    Aplicar análisis SHAP al modelo usando PermutationExplainer (más robusto).

    Args:
        model: Modelo PyTorch entrenado
        X_test: Datos de prueba
        feature_names: Nombres de las características
        save_dir: Directorio para guardar los gráficos
    """
    # Crear directorio
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Crear función wrapper para el modelo
    def model_wrapper(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(next(model.parameters()).device)
            return model(x_tensor).cpu().numpy()

    # Usar muestra más pequeña para velocidad
    sample_size = min(100, len(X_test))
    X_sample = X_test[:sample_size]

    # Usar PermutationExplainer (más robusto para PyTorch)
    explainer = shap.Explainer(model_wrapper, X_sample)

    # Calcular valores SHAP
    shap_values = explainer(X_sample)
    shap_values.feature_names = feature_names

    # 1. Gráfico de barras (importancia global)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("Importancia de Características (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance_bar.png")
    plt.close()
    print(f"Guardado: {save_dir}/feature_importance_bar.png")

    # 2. Summary plot (importancia detallada)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance_summary.png")
    plt.close()
    print(f"Guardado: {save_dir}/feature_importance_summary.png")

    # 3. Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance_beeswarm.png")
    plt.close()
    print(f"Guardado: {save_dir}/feature_importance_beeswarm.png")

    # 4. Imprimir importancia en texto
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\nImportancia de características (SHAP):")
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.4f}")

    return shap_values

#### funcion principal

def main():
    ### 1. expermiten a analizar
    experiment_config = "config/training/experiments/05-price_house-mlp-housing-v100-training.yaml"

    ### 2. ruta absoluta
    script_dir = Path(__file__).parent.parent.parent ## <-- ruta del proyecto (price_house)
    config_path = script_dir / experiment_config

    print("Experiment config:", config_path)

    ### 3. cargar modelo y datos
    model, X_test, y_test, feature_names, config = load_model_and_data(config_path)

    print(f"Modelo cargado: {config['model_config']['model_name']}" )
    print(f"Caracteristicas: {len(feature_names)}")
    print(f"{feature_names}")


    ### 4. analisis aplicando shap

    shap_values = analyze_with_shap(model, X_test, feature_names)

    print("analisis SHAP terminado. Graficos guardados en:  reports/shap")


if __name__ == "__main__":
   main()
