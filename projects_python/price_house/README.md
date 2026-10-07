# Predicción de precios de viviendas con una red neuronal

Proyecto de aprendizaje para estimar el valor medio de viviendas en California con una red neuronal densa (MLP), PyTorch y el dataset California Housing de scikit-learn. El objetivo no es presentar un modelo listo para producción: es dejar un registro claro de los experimentos, lo que aprendí al construirlo y las limitaciones que fui encontrando.

## Qué hace el proyecto

1. Descarga y carga California Housing desde `sklearn.datasets`.
2. Crea variables derivadas y, en los experimentos 7 y 8, variables polinomiales.
3. Separa los datos en entrenamiento, validación y prueba.
4. Ajusta un `RobustScaler` solo con entrenamiento y lo aplica también a validación y prueba.
5. Construye una MLP configurable desde YAML, la entrena con PyTorch y registra parámetros y métricas en MLflow.
6. Guarda el preprocesador y el modelo localmente. Hay un script exploratorio para explicar predicciones con SHAP.

## Estructura

```text
price_house/
├── config/training/experiments/   # YAML de los experimentos 01 a 08
├── src/
│   ├── analysis/shap_analysis.py  # Explicación exploratoria con SHAP
│   ├── processing/load_data.py    # Dataset, features, divisiones y escalado
│   └── training/
│       ├── model.py               # Arquitectura MLP
│       └── train.py               # Entrenamiento, early stopping y MLflow
├── reports/                       # Salidas locales, no versionadas
├── artifacts/                     # Preprocesador local, no versionado
└── models/                         # Pesos locales, no versionados
```

## Preparar y ejecutar

Desde la carpeta `projects_python/price_house`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El dataset se obtiene con scikit-learn y puede requerir conexión a internet la primera vez.

Abre una terminal para MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

En otra terminal, desde la misma carpeta del proyecto, ejecuta por ejemplo el experimento 8:

```powershell
python src/training/train.py --config config/training/experiments/08-price_house-mlp-housing-v100-training.yaml
```

La configuración por defecto del script es el experimento 1; se recomienda pasar `--config` explícitamente para que quede claro qué YAML se está ejecutando. La URI de seguimiento configurada es `http://127.0.0.1:5000`.

## Autocompletado y tipos en VS Code

El entorno virtual del proyecto es `projects_python/price_house/.venv`. En VS Code, selecciona ese intérprete con `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path...** → `.venv/Scripts/python.exe`. Si abres todo el repositorio como workspace, asegúrate de elegir el intérprete que está dentro de `projects_python/price_house`; el `.venv` de la raíz es otro entorno y puede no tener PyTorch instalado. Después ejecuta **Developer: Reload Window** para que Pylance vuelva a indexar los paquetes.

Para instalar los stubs y el verificador de tipos dentro del entorno del proyecto:

```powershell
cd projects_python/price_house
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

PyTorch debe estar instalado en el intérprete seleccionado para que `import torch.nn as nn` pueda sugerir clases como `Linear`, `Sequential`, `MSELoss` y `HuberLoss`. `requirements-dev.txt` añade stubs para PyYAML y pandas, además de Pyright. La sección `[tool.pyright]` de `pyproject.toml` activa análisis gradual en modo básico y resuelve los imports desde `src/`. Puedes ejecutar el análisis desde esta carpeta con `pyright`.

Las anotaciones ayudan al editor a mostrar nombres, parámetros y tipos, pero no convierten Python en un lenguaje con tipos obligatorios ni garantizan que una operación sea válida en tiempo de ejecución. Algunas librerías solo tienen cobertura parcial; cuando no publican stubs compatibles, la calidad del autocompletado puede variar.

## Datos y evaluación

California Housing contiene 20,640 filas, ocho variables originales y el objetivo `MedHouseVal`. El objetivo está expresado en unidades de 100,000 dólares: por ejemplo, `0.5` representa $50,000. No se escala el objetivo en este pipeline.

La separación actual es 64% entrenamiento, 16% validación y 20% prueba. Se hace en dos pasos: primero se reserva el test (20%); luego se aparta el 20% del 80% restante para validación. Con 20,640 filas, eso produce aproximadamente 13,209 de entrenamiento, 3,303 de validación y 4,128 de prueba.

- **Entrenamiento:** los lotes que actualizan los pesos de la red.
- **Validación:** calcula `val_loss` al final de cada época; esa señal elige el mejor checkpoint, alimenta `ReduceLROnPlateau` y controla early stopping.
- **Prueba:** se evalúa una vez al final, después de recuperar el checkpoint con mejor validación. No debe decidir épocas ni hiperparámetros.

El escalador `RobustScaler` se ajusta con `X_train` y transforma también `X_val` y `X_test`. De ese modo, estadísticas del test no participan en el preprocesamiento del entrenamiento.

El MLP recibe 12 variables con ingeniería básica (8 originales y 4 derivadas). Si `use_polynomial_features: true`, cuatro variables seleccionadas se reemplazan por 14 términos polinomiales de grado 2, para un total de 22 entradas. La arquitectura del experimento 8 es `[256, 128, 64]`, con Batch Normalization, ReLU y dropout de 0.15.

El entrenamiento lee del YAML la función de pérdida (`MSELoss`, `HuberLoss` o `L1Loss`), el optimizador (Adam o AdamW), sus parámetros, el scheduler `ReduceLROnPlateau` y los parámetros de early stopping. Los nombres no soportados producen un error explícito. El scheduler y early stopping usan validación; las métricas `test_*` se registran al final.

## Recorrido de experimentación

Los YAML conservan la intención de cada prueba. El recorrido fue exploratorio, no una búsqueda sistemática con validación cruzada:

| Experimento | Qué exploré |
|---|---|
| 01 | Primera MLP `[128, 64]`, ReLU, dropout 0.2 y learning rate 0.001. |
| 02 | Más capacidad `[256, 128, 64]`, LeakyReLU y learning rate 0.01. |
| 03 | Bajé el learning rate a 0.0005 y subí dropout a 0.3. |
| 04 | El YAML conserva esencialmente la misma configuración que el 03; no es una comparación aislada que permita atribuir diferencias a un cambio concreto. |
| 05 | Probé una capa adicional: `[256, 128, 64, 32]`. |
| 06 | Probé una red más pequeña, LeakyReLU, learning rate 0.0001 y hasta 100 épocas. |
| 07 | Añadí variables polinomiales, usé `[256, 128, 64]`, batch size 64 y dropout 0.15. Fue el mejor resultado preliminar. |
| 08 | Mantuve la base del 07 y configuré Huber (`delta: 1.0`) para reducir la influencia de errores grandes. |

El `RobustScaler` ya estaba en el cargador de datos; por eso no fue un cambio exclusivo del 08. Asimismo, los primeros runs son antecedentes exploratorios: en el código inicial la pérdida estaba escrita directamente y el YAML no determinaba qué pérdida se construía. También se consultaba el test durante la selección de épocas. Por estas razones, sus métricas no son comparables de forma limpia con la evaluación de tres conjuntos actual.

### Referencia actual del experimento 08

La última ejecución, ya con train/validation/test separados, detuvo el entrenamiento en la época 15. En la salida se observó `val_loss = 0.4094` en la época 10 y las siguientes métricas finales de test:

| Métrica de test | Resultado |
|---|---:|
| MSE | 0.3710 |
| RMSE | 0.6091 (aprox. $60,910) |
| MAE | 0.4024 (aprox. $40,240) |
| R² | 0.7169 |
| MAPE | 22.4104% |

Esta es la referencia más honesta hasta ahora porque el test quedó apartado de las decisiones de entrenamiento. El RMSE preliminar del experimento 07 (aprox. 0.523) salió de una configuración de evaluación anterior que usaba el test durante el entrenamiento; **no debe interpretarse como una victoria comparable del 07 frente al 08 actual**. Una métrica peor después de corregir el protocolo no implica por sí sola que el modelo haya empeorado: ahora estamos midiendo con datos que no guiaron su selección.

En la época 10, la pérdida de entrenamiento (0.1679) era menor que la pérdida de validación (0.4094). Esa diferencia es una señal de que el modelo aprende mejor los ejemplos vistos que los no vistos y puede estar sobreajustándose. Early stopping detuvo la corrida en la época 15; conviene revisar las curvas de todas las épocas antes de decidir si hace falta más regularización o una arquitectura distinta.

El objetivo está expresado en cientos de miles de dólares, por eso los errores se convierten multiplicando por 100,000. Huber con `delta: 1.0` considera cuadráticos los residuos menores a 1 unidad y lineales los mayores; aquí 1 unidad equivale a $100,000. El MAPE se conserva como referencia, pero no lo uso como único criterio para escoger el modelo.

## Reproducibilidad y registro

La semilla del YAML (`environment.seed`) se aplica a Python, NumPy, PyTorch y al generador del `DataLoader`. `training_params.random_state` controla las divisiones de scikit-learn. Esto hace las corridas más repetibles, aunque algunas operaciones en GPU pueden seguir teniendo diferencias entre equipos.

MLflow conserva las corridas, parámetros, curvas de pérdida, métricas finales y artefactos del entrenamiento. Los archivos locales de MLflow, los pesos, el preprocesador y las gráficas generadas se excluyen de Git para no subir bases de datos ni resultados binarios grandes. Cada persona puede regenerarlos al ejecutar el proyecto.

## Lo que aprendí

- Una configuración YAML solo tiene efecto si el código la lee y construye el objeto correspondiente. Ahora `build_loss`, `build_optimizer` y `build_scheduler` hacen explícita esa conexión.
- Train, validation y test cumplen trabajos distintos. Usar test para early stopping contamina la medición final, aunque el nombre de la variable diga `test`.
- El escalador se ajusta únicamente con entrenamiento para evitar filtrar información de validación o prueba.
- La función de pérdida y las métricas finales responden preguntas distintas. Huber reduce el peso de residuos grandes durante el entrenamiento, pero el RMSE sigue penalizando esos errores con fuerza.
- Una red más grande, más épocas o un learning rate menor no garantizan mejores predicciones. Hay que comparar bajo la misma división y cambiar una decisión a la vez.
- Las semillas ayudan a distinguir cambios reales de variaciones aleatorias; MLflow permite conservar qué parámetros produjeron cada resultado.
- El mejor resultado que vi en un test usado repetidamente para tomar decisiones no es una evaluación independiente. Mantener el test intacto importa más que perseguir una cifra menor.

## Próximo paso

Usar esta ejecución del 08 como referencia y probar OneCycleLR, manteniendo fijos datos, semilla, arquitectura, pérdida y demás parámetros. OneCycleLR ajusta la tasa en cada lote, por lo que requiere una implementación diferente a `ReduceLROnPlateau`, que se actualiza por época con `val_loss`. Comparar varias semillas antes de concluir que una opción es mejor.

## Limitaciones

- Es un proyecto de aprendizaje y no un estimador de vivienda para uso comercial.
- Los experimentos históricos no siempre variaron un único parámetro y algunos YAML repetían configuraciones.
- El test actual es una única partición aleatoria; una evaluación más sólida futura podría usar validación cruzada para seleccionar y comparar configuraciones antes de una evaluación final.
- El entrenamiento guarda archivos locales y depende de la estructura de carpetas descrita; todavía no es un paquete de inferencia ni una API.
