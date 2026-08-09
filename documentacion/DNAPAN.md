# DNAPAN.md

Archivo de documentación técnica para `DNAPAN.py`.

## ¿Qué hace el archivo?

`DNAPAN.py` implementa el modelo **DNAPAN**, un modelo de clasificación de secuencias que determina el nivel de agrado de una respuesta del LLM, devolviendo `positive` o `negative`.

El flujo general es:

1. Lee las conversaciones en `json/conversaciones` (formato OpenAI `messages`, ver `formato_openai.md`).
2. Para cada par user->assistant (de `pares_entrenamiento`) combina `input` (pregunta) y `output` (respuesta) en un texto: `Pregunta: ... \nRespuesta: ...`.
3. Infiere la calificación usando el modelo DNAPAN (clase 0 = `negative`, clase 1 = `positive`).
4. Guarda los resultados en `json/entrenamiento/dataset_etiquetado.json`.
5. Devuelve una lista de diccionarios con `{id, qualification, archivo_origen}`.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `models/DNAPAN/model` | Directorio | Peso y tokenizador del modelo DNAPAN. |
| `json/conversaciones` | Directorio | Conversaciones en JSON (entrada). |
| `json/entrenamiento/dataset_etiquetado.json` | Archivo (salida) | Resultados de la inferencia. |
| `json/entrenamiento/dataset_actualizado.json` | Archivo (salida) | Dataset actualizado con calificaciones. |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer`, `AutoModelForSequenceClassification`.
- **PyTorch**: `torch`, inferencia con `torch.no_grad()` y `softmax`.

## Funciones principales

| Función | Rol |
|---------|-----|
| `DNAPAN_json()` | Procesa todas las conversaciones, infiere y guarda los resultados. |
| `DNAPAN_inferir_texto(texto)` | Infiere la calificación de un solo texto. |
| `DNAPAN_actualizar_dataset()` | Actualiza el dataset existente añadiendo `qualification` por ID. |

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Las tres funciones están decoradas: `DNAPAN_json` y `DNAPAN_actualizar_dataset` con `@manejar_errores(default=[])`, y `DNAPAN_inferir_texto` con `@manejar_errores(default=None)`. Así, ante un error devuelven lista vacía o `None` según corresponda. El bloque `if __name__ == "__main__":` se envuelve en `try/except` que imprime `[ERROR FATAL]`.

## Mapeo de predicción

- Clase predicha `1` → `"positive"`.
- Clase predicha `0` → `"negative"`.
- Error durante inferencia → `"error"`.

## Parámetros de tokenización

- `truncation=True`, `padding=True`, `max_length=512`.

## Notas

- Se aplica `model.eval()` para desactivar dropout/gradientes en inferencia.
- El texto procesado combina pregunta y respuesta: `Pregunta: {input}\nRespuesta: {output}` (tomados de los pares de `formato_openai.pares_entrenamiento`).
- Las estadísticas finales (positivos/negativos/errores) se imprimen en consola.