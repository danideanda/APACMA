# clasificador.md

Archivo de documentación técnica para `clasificador.py`.

## ¿Qué hace el archivo?

`clasificador.py` es el **orquestador de la clasificación**. Combina los IDs de `json_script.py` con la inferencia del modelo DNAPAN para construir datasets de entrenamiento completos.

Flujo general:

1. Obtiene los IDs de mensajes clasificados como `positive`/`negative` vía `id_json()`.
2. Agrupa los IDs por archivo y extrae los mensajes completos (`date`, `input`, `output`).
3. Los procesa con `DNAPAN_inferir_texto()` para re-asignar su calificación.
4. Guarda el dataset resultante en `json/entrenamiento/dataset.json`.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json/conversaciones` | Directorio | Conversaciones en JSON (entrada). |
| `json/entrenamiento/dataset.json` | Archivo (salida) | Dataset filtrado con calificaciones de DNAPAN. |
| `json/entrenamiento/dataset_dnapan_filtrado.json` | Archivo (salida) | Dataset generado por `juntar_con_dnapan_completo()`. |

## Tecnologías

- **Python**.
- **os / json** (estándar).
- **Transformers / PyTorch** (a través de los módulos DNAPAN importados).

## Módulos que importa

| Módulo | Uso |
|--------|-----|
| `json_script.id_json` | IDs de mensajes `positive`/`negative`. |
| `json_script.listar_chats` | Procesamiento de chats (importado aunque no se usa directamente en este módulo). |
| `DNAPAN.DNAPAN_inferir_texto` | Inferencia de calificación para un texto. |
| `DNAPAN.DNAPAN_json` | Inferencia masiva sobre todas las conversaciones. |

## Funciones principales

| Función | Rol |
|---------|-----|
| `juntar()` | Extrae mensajes por ID, los procesa con DNAPAN y guarda `dataset.json`. |
| `juntar_con_dnapan_completo()` | Usa `DNAPAN_json()` para clasificar todos y luego filtra por IDs interesantes. |
| `extraer_por_ids(lista_ids)` | Extrae mensajes completos a partir de una lista de IDs provista por el usuario. |

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Las tres funciones (`juntar`, `juntar_con_dnapan_completo`, `extraer_por_ids`) están decoradas con `@manejar_errores(default=[])`, por lo que ante un error devuelven lista vacía en lugar de propagar la excepción. El bloque `if __name__ == "__main__":` se envuelve en `try/except` que imprime `[ERROR FATAL]`.

## Campos del mensaje de salida

```json
{
    "id": 0,
    "date": "2026-08-04T12:22:45.766086",
    "input": "pregunta",
    "output": "respuesta",
    "qualification": "positive|negative|neutra|error",
    "archivo_origen": "conversacion-1.json"
}
```

> Las conversaciones de **entrada** se leen con `formato_openai.leer_conversacion()` (formato OpenAI `messages`, ver `formato_openai.md`) y se convierten a pares con `pares_entrenamiento()`. El dataset de **salida** (`dataset.json`) sí usa `input`/`output` porque es el formato que consume `fine.py`.

## Comportamiento ante fallo de DNAPAN

Si `DNAPAN_inferir_texto()` retorna `None`, se conserva la `qualification` original del mensaje (por defecto `"neutra"`).

## Notas

- Los mensajes se ordenan por `id` en `juntar()`.
- `extraer_por_ids()` usa un `set` para búsqueda eficiente de IDs.
- Las estadísticas (positivos/negativos/neutros/errores) se imprimen al finalizar.