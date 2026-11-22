# Photovoltaic XAI

Este trabajo compara tres redes de clasificación multietiqueta para seis tipos de defecto en imágenes de electroluminiscencia. También estudia qué aporta una rama regional a ConvNeXt-Tiny en las predicciones y en los mapas que las explican.

## Datos

Descarga PVEL-AD desde el [repositorio de sus autores](https://github.com/binyisu/PVEL-AD), incluidas las anotaciones de prueba. Coloca los archivos en la siguiente estructura.

```text
data/raw/pvel_ad/EL2021/
├── trainval/
│   ├── JPEGImages/
│   └── Annotations/
├── test/
│   ├── JPEGImages/
│   └── Annotations/
└── othertypes/
    └── good/
```

Si los XML de `test` vienen en un archivo separado, extráelos en `test/Annotations/`. Cada imagen de `trainval` y `test` debe tener un XML con el mismo nombre base. El cuaderno 01 lee ambas carpetas y las imágenes sanas de `good`. Los nombres de las carpetas originales indican su procedencia. El cuaderno crea sus propias particiones de entrenamiento, validación y prueba por grupos.

Con los datos utilizados, el cuaderno 01 selecciona 33 659 imágenes y genera `data/prepared/pvel6-eda-v1.json`.

## Ejecución

Para reproducir el experimento completo, usa Linux x86-64, Python 3.12 y una GPU NVIDIA compatible con CUDA 12.8. Elige una de las dos opciones desde la raíz del proyecto.

### Con `uv`

El entorno se instala con las versiones fijadas en `uv.lock`.

```bash
uv sync --locked --extra cuda
uv run --locked --extra cuda jupyter lab
```

### Con `pip`

`requirements.txt` contiene las mismas versiones de las dependencias para CUDA. También instala el paquete del proyecto para que los notebooks puedan importarlo. Este fichero se autogenera a partir del entorno de `uv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

### Notebooks

Abre los notebooks de la carpeta `notebooks` en orden y ejecuta todas las celdas con un kernel nuevo del entorno `.venv` para cada notebook.

| Notebook | Qué calcula |
| --- | --- |
| [01 · Datos](notebooks/01_data_preparation.ipynb) | Revisa las imágenes y anotaciones. Agrupa duplicados y calcula las particiones, el balance y la normalización. |
| [02 · Referencias](notebooks/02_reference_models.ipynb) | Entrena y compara EfficientNet-B0, ConvNeXt-Tiny y Swin-Tiny en validación. |
| [03 · Errores y explicaciones](notebooks/03_reference_errors_xai.ipynb) | Examina los fallos de las referencias y sus mapas Grad-CAM, Layer-CAM y RISE. |
| [04 · Variante](notebooks/04_global_regional_variant.ipynb) | Entrena la variante global–regional con el ConvNeXt de referencia congelado. Primero ajusta el extractor regional y después compara seis entrenamientos de la fusión. |
| [05 · Prueba y coste](notebooks/05_test_and_cost.ipynb) | Evalúa los cuatro modelos, estima la incertidumbre con bootstrap por grupos y mide su coste con CUDA. |
| [06 · Comparación XAI](notebooks/06_variant_xai_comparison.ipynb) | Compara los mapas de la variante y las referencias sobre los mismos casos de validación. |

Los notebooks entrenan o calculan lo que falta y reutilizan los archivos compatibles de `outputs/`. La primera ejecución puede descargar los pesos de ImageNet y tardar varias horas. Los checkpoints, las características y los archivos de XAI sirven para continuar el cálculo o alimentar el siguiente notebook.

## Usar los modelos exportados

La última celda del 06 exporta los modelos a `outputs/models/`. El directorio contiene los pesos y un `manifest.json` con el orden de las clases, la normalización, los umbrales y la geometría. La variante comparte los pesos de ConvNeXt-Tiny con la referencia.

Para clasificar una imagen con la variante, ejecuta lo siguiente desde la raíz del proyecto.

```python
from pathlib import Path

import torch
from PIL import Image

from photovoltaic_xai.variant import load_variant, load_variant_manifest, variant_views

path = Path("outputs/models/manifest.json")
settings = load_variant_manifest(path)
model = load_variant(path, manifest=settings)
with Image.open("imagen.jpg") as image:
    views = variant_views(image, settings["normalization"])

with torch.inference_mode():
    logits = model(*(views[key][None] for key in ("image", "tiles", "tile_valid")))["logits"][0]
probabilities = logits.sigmoid()
defects = [
    name for name, score, threshold in zip(settings["class_names"], probabilities, settings["thresholds"])
    if score >= threshold
]
```

Las referencias se crean con `create_model(nombre)` y cargan el archivo `settings["references"][nombre]["weights"]` del directorio exportado. Para ellas se usa `PairedELTransform` con el tamaño, la media y la desviación del manifiesto. Cada referencia usa su propia lista `settings["references"][nombre]["thresholds"]`.

## Comprobaciones

Con cualquiera de las dos instalaciones, activa el entorno y ejecuta las comprobaciones desde la raíz del proyecto.

```bash
source .venv/bin/activate
python -m ruff check src tests scripts notebooks/0[1-6]_*.ipynb
python scripts/check.py
python -m pytest
```
