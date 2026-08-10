# DNAPAN.md

Archivo de documentación técnica para `DNAPAN.py` y `DNAPAN/model.py`.

## ¿Qué hace el archivo?

`DNAPAN.py` implementa el modelo **DNAPAN**, un modelo de clasificación de secuencias que determina el nivel de agrado de una respuesta del LLM, devolviendo `positive` o `negative`.

El flujo general es:

1. Lee las conversaciones en `json/conversaciones` (formato OpenAI `messages`, ver `formato_openai.md`).
2. Para cada par user->assistant (de `pares_entrenamiento`) combina `input` (pregunta) y `output` (respuesta) en un texto: `Pregunta: ... \nRespuesta: ...`.
3. Infiere la calificación usando el modelo DNAPAN (clase 0 = `negative`, clase 1 = `positive`).
4. Guarda los resultados en `json/entrenamiento/dataset_etiquetado.json`.
5. Devuelve una lista de diccionarios con `{id, qualification, archivo_origen}`.

`DNAPAN/model.py` es el **script de entrenamiento** del clasificador DNAPAN. Recolecta ejemplos etiquetados desde el dataset y las conversaciones, entrena un `AutoModelForSequenceClassification` desde el modelo base y guarda el resultado en `models/DNAPAN/model`.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `models/DNAPAN/model` | Directorio | Peso y tokenizador del modelo DNAPAN. |
| `json/conversaciones` | Directorio | Conversaciones en JSON (entrada). |
| `json/entrenamiento/dataset_etiquetado.json` | Archivo (salida) | Resultados de la inferencia. |
| `json/entrenamiento/dataset_actualizado.json` | Archivo (salida) | Dataset actualizado con calificaciones. |
| `models/LLM-base` | Directorio | Modelo base desde el que se entrena el clasificador. |
| `json/entrenamiento/dataset.json` | Archivo (entrada) | Dataset con `input`/`output`/`qualification` usado por el entrenamiento. |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`.
- **datasets** (Hugging Face): `Dataset`.
- **PyTorch**: `torch`, inferencia con `torch.no_grad()` y `softmax`; semilla fija para reproducibilidad.

## Funciones principales

| Función | Rol |
|---------|-----|
| `DNAPAN_json()` | Procesa todas las conversaciones, infiere y guarda los resultados. |
| `DNAPAN_inferir_texto(texto)` | Infiere la calificación de un solo texto. |
| `DNAPAN_actualizar_dataset()` | Actualiza el dataset existente añadiendo `qualification` por ID. |
| `entrenar_dnapan()` | Entrena el clasificador DNAPAN y guarda el modelo en `models/DNAPAN/model`. |
| `_cargar_ejemplos()` | Recolecta ejemplos `{texto, label}` desde dataset y conversaciones. |
| `_calcular_epocas(total, batch)` | Aplica la fórmula de iteraciones por época (ver más abajo). |

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Las tres funciones de `DNAPAN.py` están decoradas: `DNAPAN_json` y `DNAPAN_actualizar_dataset` con `@manejar_errores(default=[])`, y `DNAPAN_inferir_texto` con `@manejar_errores(default=None)`. Así, ante un error devuelven lista vacía o `None` según corresponda. El bloque `if __name__ == "__main__":` se envuelve en `try/except` que imprime `[ERROR FATAL]`.

En `DNAPAN/model.py`, `entrenar_dnapan` y `_cargar_ejemplos` también usan `@manejar_errores` (con `default=None` y `default=[]` respectivamente).

## Mapeo de predicción

- Clase predicha `1` → `"positive"`.
- Clase predicha `0` → `"negative"`.
- Error durante inferencia → `"error"`.

## Parámetros de tokenización

- `truncation=True`, `padding=True`, `max_length=512`.

## Entrenamiento de `DNAPAN/model.py`

### Origen de los datos

`_cargar_ejemplos()` recolecta ejemplos desde dos fuentes:

1. `json/entrenamiento/dataset.json` (formato `input`/`output`/`qualification`).
2. `json/conversaciones` (conversaciones en formato OpenAI `messages`, descompuestas con `pares_entrenamiento()`).

Solo se usan ejemplos con `qualification` `positive` (label `1`) o `negative` (label `0`); los `neutra`/`error`/sin campo se descartan. El texto se combina como `Pregunta: {input}\nRespuesta: {output}`.

### Fórmula de épocas

Las épocas no son un número fijo: se calculan con la fórmula

```
Iteraciones por época = Tamaño total de datos / Tamaño del lote (Batch Size)
```

implementada en `_calcular_epocas(total_ejemplos, batch_size)`, que redondea hacia arriba (`ceil`) y garantiza un mínimo de 1 época. Con el batch de 8 por defecto y `N` ejemplos de entrenamiento, las épocas resultan `ceil(N / 8)`.

### Parámetros mínimos del modelo

El modelo base (`models/LLM-base`, Qwen2 0.5B ≈ 494M de parámetros) debe cumplir el mínimo exigido de **5 millones de parámetros** (`PARAMETROS_MINIMOS = 5_000_000`). Si el modelo cargado tiene menos, se imprime el aviso y se omite el entrenamiento.

### Reproducibilidad

Se fija `SEMILLA = 42` (`torch.manual_seed` y `seed` en `TrainingArguments`) y el split de train/test usa `seed=SEMILLA`, de modo que entrenar dos veces produce el mismo resultado.

## Notas

- Se aplica `model.eval()` para desactivar dropout/gradientes en inferencia.
- El texto procesado combina pregunta y respuesta: `Pregunta: {input}\nRespuesta: {output}` (tomados de los pares de `formato_openai.pares_entrenamiento`).
- Las estadísticas finales (positivos/negativos/errores) se imprimen en consola.