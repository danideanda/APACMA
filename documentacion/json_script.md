# json_script.md

Archivo de documentación técnica para `json_script.py`.

## ¿Qué hace el archivo?

`json_script.py` es el **módulo de utilidades JSON**. Se encarga de:

- `id_json()`: Leer las conversaciones de `json/conversaciones`, extraer los IDs de los mensajes `assistant` clasificados como `positive` o `negative`, junto con el archivo de origen, para su posterior procesamiento con DNAPAN.

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

### `id_json()`

- Recorre todas las conversaciones.
- Filtra mensajes `assistant` con `qualification` en `["positive", "negative"]` que tengan `id`.
- Retorna una lista de `{id, archivo, qualification}`.

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

`id_json` está decorado con `@manejar_errores(default=[])`. Ante un error devuelve el valor por defecto en lugar de propagar la excepción.

## Estructura de datos de entrada

Las conversaciones se leen con `formato_openai.leer_conversacion()`, que normaliza al **formato estándar OpenAI `messages`** (ver `formato_openai.md`):

```json
{
    "messages": [
        {"role": "user", "content": "pregunta del usuario"},
        {"role": "assistant", "content": "respuesta del modelo", "qualification": "positive"}
    ]
}
```

`id_json()` recorre los mensajes con `obtener_mensajes()` y extrae los de `role: assistant` con `qualification` en `["positive", "negative"]`.

## Notas

- `id_json` es dependencia de `clasificador.py`, que lo importa para construir el dataset.
- Las conversaciones sin mensajes clasificados generan una lista vacía.