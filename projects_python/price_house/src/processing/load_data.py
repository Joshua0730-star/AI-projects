"""Data loading module for California Housing dataset."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch
from typing import cast
import joblib
from pathlib import Path

### caracteristicas mas importantes en la prediccion final segun SHAP

# 1. Latitude: 0.3941 (6) idx
# 2. MedInc: 0.3138 (0) idx
# 3. Longitude: 0.3007 (7) idx
# 4. rooms_per_household: 0.1920 (9) idx
# 5. bedrooms_per_room: 0.1726

def load_california_housing_data(test_size=0.2, random_state=42, feature_engineering=True, PolinomialFeatures=False):
    """
    Load California Housing dataset from scikit-learn and split with custom parameters.

    Args:
        test_size (float): Proportion of dataset for test split
        random_state (int): Random seed for reproducibility
        feature_engineering (bool): Whether to apply feature engineering
        PolinomialFeatures (bool): Whether to apply PolynomialFeatures to top characteristics

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler, all_features)
    """
    # Load full dataset from scikit-learn
    # The current scikit-learn overload allows the return type to include a tuple
    # even with return_X_y=False, so narrow it to the Bunch form used here.
    housing = cast(Bunch, fetch_california_housing(return_X_y=False, as_frame=False))
    X_full: np.ndarray = np.asarray(housing.data)
    y_full: np.ndarray = np.asarray(housing.target)
    feature_names = list(housing.feature_names)

    if feature_engineering:
        # # Transformaciones logarítmicas para características asimétricas
        # # Índices: 0=MedInc, 2=AveRooms, 3=AveBedrms, 4=Population, 5=AveOccup
        # X_full[:, 0] = np.log1p(X_full[:, 0])  # MedInc
        # X_full[:, 2] = np.log1p(X_full[:, 2])  # AveRooms
        # X_full[:, 3] = np.log1p(X_full[:, 3])  # AveBedrms
        # X_full[:, 4] = np.log1p(X_full[:, 4])  # Population
        # X_full[:, 5] = np.log1p(X_full[:, 5])  # AveOccup

        # Feature engineering: características derivadas
        feature_names_derived = ['bedrooms_per_room', 'rooms_per_household',
                                'population_per_household', 'income_per_room']
        all_features = feature_names + feature_names_derived

        # Nueva característica de dormitorios por cuarto
        bedrooms_per_room = X_full[:, 3] / X_full[:, 2]
        # Habitaciones por persona
        rooms_per_household = X_full[:, 2] / X_full[:, 5]
        # Personas promedio por casa
        population_per_household = X_full[:, 4] / X_full[:, 5]
        # Ingreso de las personas por cuarto
        income_per_room = X_full[:, 0] / (X_full[:, 2] + 1)

        X_full = np.column_stack([X_full, bedrooms_per_room, rooms_per_household,
                                 population_per_household, income_per_room])

        ### aplicar caracteristicas polinomiales a top 4
        if PolinomialFeatures:
        # Índices: 0=MedInc, 6=Latitude, 7=Longitude, 9=rooms_per_household
            important_idx = [0, 6, 7, 9]
            important_names = [all_features[i] for i in important_idx]


            ### crear caracteristicas polinomiales de grado 2
            poly = PolynomialFeatures(degree=2, include_bias=False)
            x_poly = poly.fit_transform(X_full[:, important_idx])

            ### Generar nombres para las características polinomiales
            poly_feature_names = poly.get_feature_names_out(important_names)

            # Eliminar las características originales importantes
            X_full = np.delete(X_full, important_idx, axis=1)
            remaining_names = [name for i, name in enumerate(all_features) if i not in important_idx]


            ### Combinar características restantes con polinomios
            X_full = np.column_stack([X_full, x_poly])
            all_features = list(remaining_names) + list(poly_feature_names)

            print(f"Polynomial features aplicadas: {len(poly_feature_names)} nuevas características")
    else:
        all_features = feature_names




    # Custom train/test split using sklearn
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full,
        test_size=test_size,
        random_state=random_state
    )
    X_train_full, X_test = np.asarray(X_train_full), np.asarray(X_test)
    y_train_full, y_test = np.asarray(y_train_full), np.asarray(y_test)

    ##64% para entrenamiento, 16% para validación y 20% para test.
    X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=random_state,
)
    X_train, X_val = np.asarray(X_train), np.asarray(X_val)
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)

    # Scale features (fit on training data only, transform both)
    scaler = RobustScaler()

    scaler.fit(X_train)

    X_train = cast(np.ndarray, scaler.transform(X_train))
    X_val = cast(np.ndarray, scaler.transform(X_val))
    X_test = cast(np.ndarray, scaler.transform(X_test))

    print(f"Dataset loaded:")
    print(f"  Total samples: {X_full.shape[0]}")
    print(f"  Training samples: {X_train.shape[0]} ({len(X_train) / len(X_full) * 100:.1f}%)")
    print(f"  Validation samples: {X_val.shape[0]} ({len(X_val) / len(X_full) * 100:.1f}%)")
    print(f"  Test samples: {X_test.shape[0]} ({len(X_test) / len(X_full) * 100:.1f}%)")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, all_features


def save_preprocessor(scaler, save_path):
    """
    Save the preprocessor (scaler) to disk.

    Args:
        scaler: Fitted RobustScaler
        save_path (str): Path to save the preprocessor
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Preprocessor saved to: {save_path}")


def load_preprocessor(load_path):
    """
    Load the preprocessor from disk.

    Args:
        load_path (str): Path to load the preprocessor from

    Returns:
        scaler: Loaded RobustScaler
    """
    scaler = joblib.load(load_path)
    print(f"Preprocessor loaded from: {load_path}")
    return scaler


# if __name__ == "__main__":
#     # Test data loading
#     X_train, X_test, y_train, y_test, scaler = load_california_housing_data()

#     # Save preprocessor
#     save_path = "artifacts/california_housing_preprocessor.joblib"
#     save_preprocessor(scaler, save_path)
