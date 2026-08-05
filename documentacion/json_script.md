# json_script.md

Archivo de documentación técnica para `json_script.py`.

## ¿Qué hace el archivo?

`json_script.py` es el **módulo de utilidades JSON**. Se encarga de:

1. `listar_chats()`: Leer las conversaciones de `json/conversaciones`, extraer los mensajes que tengan `qualification` y `text`, y generar un dataset de entrenamiento `{text, label}`.
2. `id_json()`: Extraer los IDs de los mensajes clasificados como `positive` o `negative`, junto con el archivo de origen, para su posterior procesamiento con DNAPAN.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json/conversaciones` | Directorio | Conversaciones en JSON (entrada). |
| `json/entrenamiento/dataset_filtrado.json` | Archivo (salida) | Dataset filtrado con `{text, label}`. |

## Tecnologías

- **Python**.
- **json** (estándar) para lectura/escritura.
- **os** (estándar) para manejo de rutas.

## Funciones principales

### `listar_chats()`

- Lista los archivos `.json` de `json/conversaciones`.
- Solo guarda mensajes que tengan `qualification` y `text` (los demás se descartan).
- Genera y guarda el dataset en `json/entrenamiento/dataset_filtrado.json`.
- Retorna un diccionario con: `archivos`, `chats_procesados`, `total_mensajes`, `dataset`, `ruta_dataset`.

### `id_json()`

- Recorre todas las conversaciones.
- Filtra mensajes con `qualification` en `["positive", "negative"]` que tengan `id`.
- Retorna una lista de `{id, archivo, qualification}`.

## Estructura de datos de entrada

Cada mensaje en `json/conversaciones/*.json` tiene al menos:

```json
{
    "id": 0,
    "date": "2026-08-04T12:22:45.766086",
    "input": "pregunta del usuario",
    "output": "respuesta del modelo",
    "qualification": "positive|negative|neutra"
}
```

## Notas

- Ambos scripts son dependencia de `clasificador.py`, que los importa como `id_json` y `listar_chats`.
- Las conversaciones sin mensajes clasificados generan un dataset vacío y no se guarda archivo.