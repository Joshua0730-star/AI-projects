"""Training script with MLflow tracking for house price prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch as mlflow_pytorch
from pathlib import Path
import yaml
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.load_data import load_california_housing_data, save_preprocessor
from training.model import initialize_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.

    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def save_histogram(x, name_features):
    project_root = Path(__file__).parent.parent.parent
    folder_path = project_root / "reports" / "histograms"
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    columns = 4
    rows = math.ceil(len(name_features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, feature_name) in enumerate(zip(axes, name_features)):
        sns.histplot(x[:, i], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribución de {feature_name}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frecuencia')

    for ax in axes[len(name_features):]:
        fig.delaxes(ax)

    ## guardamos el grafico sin mostrarlo
    plt.tight_layout()
    plt.savefig(folder_path / "histogramas.png")
    plt.close(fig)
    print(f"Grafico guardado en {folder_path}")


class LossConfig(BaseModel):
    name: str

TYPES_LOSS = {
    "MSELoss": nn.MSELoss()
    }

def build_loss(loss_config):
    name = loss_config.get("name")

    if name == "MSELoss":
        return nn.MSELoss()

    if name == "HuberLoss":
        return nn.HuberLoss(delta=loss_config.get("delta", 1.0))

    if name == "L1Loss":
        return nn.L1Loss()

    raise ValueError(f"Función de pérdida no soportada: {name}")


def build_optimizer(optimizer_config, model):
    name = optimizer_config.get("name")
    lr = optimizer_config.get("learning_rate", 0.0005)
    weight_decay = optimizer_config.get("weight_decay", 0.0001)

    if name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Optimizador no soportado: {name}")


def build_scheduler(scheduler_config, optimizer):
    name = scheduler_config.get("name")
    if name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_config.get('patience', 5),
            factor=scheduler_config.get('factor', 0.5),
            min_lr=scheduler_config.get("min_lr", 1e-6),
        )

    raise ValueError(f"Scheduler no soportado: {name}")
def train_model(config_path):
    """
    Train the model with MLflow tracking.

    Args:
        config_path (str): Path to configuration YAML file
    """
    print(f"Starting training with config: {config_path}", flush=True)

    # Load configuration
    print("Loading configuration...", flush=True)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully", flush=True)

    ### set seed
    seed = config["environment"]["seed"]
    set_seed(seed)

    # MLflow configuration
    mlops_config = config.get('mlops_config', {})
    mlflow.set_tracking_uri(mlops_config.get('tracking_uri', 'http://127.0.0.1:5000'))
    mlflow.set_experiment(mlops_config.get('experiment_name', 'price_house/training'))

    # Start MLflow run
    with mlflow.start_run(run_name=mlops_config.get('run_name', 'training_run')):
        # Log the effective configuration so each run can be understood in MLflow.
        training_config = config['training_params']
        optimizer_config = training_config['optimizer']
        loss_config = training_config['loss_function']
        scheduler_config = training_config.get('scheduler', {})
        early_stopping_config = training_config.get('early_stopping', {})
        architecture_config = config['model_config']['architecture']
        run_params = {
            'seed': seed,
            'random_state': training_config['random_state'],
            'test_size': training_config['test_size'],
            'validation_fraction_of_train': 0.2,
            'use_polynomial_features': str(config.get('use_polynomial_features', False)),
            'scaler': 'RobustScaler',
            'batch_size': training_config['batch_size'],
            'epochs_max': training_config['epochs'],
            'optimizer_name': optimizer_config['name'],
            'learning_rate_initial': optimizer_config['learning_rate'],
            'weight_decay': optimizer_config.get('weight_decay', 0.0),
            'loss_name': loss_config['name'],
            'scheduler_name': scheduler_config.get('name', 'None'),
            'early_stopping_patience': early_stopping_config.get('patience', 10),
            'early_stopping_delta': early_stopping_config.get('delta', 0.001),
            'hidden_layers': str(architecture_config['hidden_layers']),
            'activation_fn': architecture_config.get('activation_fn', 'ReLU'),
            'use_batch_norm': str(architecture_config.get('use_batch_norm', True)),
            'dropout_rate': architecture_config['dropout_rate'],
        }
        if 'delta' in loss_config:
            run_params['loss_delta'] = loss_config['delta']
        for key in ('patience', 'factor', 'min_lr'):
            if key in scheduler_config:
                run_params[f'scheduler_{key}'] = scheduler_config[key]
        mlflow.log_params(run_params)

        # Load data
        print("Loading data...")
        X_train,X_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_california_housing_data(
            test_size=config['training_params']['test_size'],
            random_state=config['training_params']['random_state'],
            feature_engineering=True,
            PolinomialFeatures=config.get("use_polynomial_features", False)
        )

        # Save preprocessor
        # Get project root directory (parent of src)


        project_root = Path(__file__).parent.parent.parent

        save_histogram(X_train, feature_names)
        artifacts_dir = project_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        preprocessor_path = artifacts_dir / config['data_source']['data_path']['preprocessor_filename']
        save_preprocessor(scaler, preprocessor_path)
        mlflow.log_artifact(str(preprocessor_path))


        # Transformación log al target para reducir skewness


        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = config['training_params']['batch_size']

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )
        print(f"DataLoader created with batch_size: {batch_size}")

        # Initialize model
        print("Initializing model...")
        model = initialize_model(config)

        # Set device (GPU if available, else CPU)
        # Force GPU if available regardless of config setting
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            print(f"Using device: CUDA (GPU)")
        else:
            device = torch.device('cpu')
            print(f"Using device: CPU (CUDA not available)")

        model = model.to(device)

        # Move tensors to device
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        X_val_tensor = X_val_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)

        print(f"Data moved to device: {device}")

        # Loss function and optimizer

        loss_config = config['training_params']['loss_function']
        criterion = build_loss(loss_config) ## <-- delta: controla el umbral entre MSE↔MAE
        optimizer = build_optimizer(config['training_params']['optimizer'], model)

        # Learning rate scheduler
        scheduler_config = config['training_params'].get('scheduler', {})
        scheduler = build_scheduler(scheduler_config, optimizer)

        # Training loop
        print("Starting training...")
        epochs = config['training_params']['epochs']
        best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = config['training_params'].get('early_stopping', {}).get('patience', 10)
        actual_epochs_run = 0

        for epoch in range(epochs):
            actual_epochs_run = epoch + 1
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                # The DataLoader stores CPU tensors; move each batch to the model's device.
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor).item()

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch)

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_loss - config['training_params'].get('early_stopping', {}).get('delta', 0.001):
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model_temp.pt')
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(torch.load('best_model_temp.pt'))
        os.remove('best_model_temp.pt')

        # Final evaluation (aqui se prueba el modelo con datos no conocidos para ver sus predicciones)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)

        # Convert predictions to CPU before calculating NumPy/sklearn metrics.
        test_metrics = calculate_metrics(y_test_tensor.detach().cpu().numpy(), test_predictions.detach().cpu().numpy())

        print("\nFinal Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
            mlflow.log_metric(f'test_{metric}', value)

        # Save model
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / config['model_config']['model_name']
        torch.save(model.state_dict(), model_path)

        # Log model to MLflow with input_example to avoid pt2 error
        input_example = X_test_tensor[:1].detach().cpu().numpy()
        mlflow_pytorch.log_model(
            model,
            "model",
            input_example=input_example,
            serialization_format="pickle"  # Use pickle instead of pt2
        )
        mlflow.log_artifact(str(model_path))

        print(f"\nModel saved to: {model_path}")
        print(f"Training completed! Total epochs run: {actual_epochs_run}/{epochs}")


if __name__ == "__main__":
    import os
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train house price prediction model')
    parser.add_argument('--config', type=str,
                        default='config/training/experiments/01-price_house-mlp-housing-v100-training.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Get absolute path to config file
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ### <-- price_house
    config_path = os.path.join(script_dir, args.config)

    print(f"Script directory: {script_dir}", flush=True)
    print(f"Config path: {config_path}", flush=True)
    print(f"Config exists: {os.path.exists(config_path)}", flush=True)

    train_model(config_path)
