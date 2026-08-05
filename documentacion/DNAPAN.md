# DNAPAN.md

Archivo de documentación técnica para `DNAPAN.py`.

## ¿Qué hace el archivo?

`DNAPAN.py` implementa el modelo **DNAPAN**, un modelo de clasificación de secuencias que determina el nivel de agrado de una respuesta del LLM, devolviendo `positive` o `negative`.

El flujo general es:

1. Lee las conversaciones en `json/conversaciones`.
2. Para cada mensaje combina `input` (pregunta) y `output` (respuesta) en un texto: `Pregunta: ... \nRespuesta: ...`.
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

## Mapeo de predicción

- Clase predicha `1` → `"positive"`.
- Clase predicha `0` → `"negative"`.
- Error durante inferencia → `"error"`.

## Parámetros de tokenización

- `truncation=True`, `padding=True`, `max_length=512`.

## Notas

- Se aplica `model.eval()` para desactivar dropout/gradientes en inferencia.
- El texto procesado combina pregunta y respuesta: `Pregunta: {input}\nRespuesta: {output}`.
- Las estadísticas finales (positivos/negativos/errores) se imprimen en consola.