# seguridad.md

Archivo de documentación técnica para `seguridad.py`.

## ¿Qué hace el archivo?

`seguridad.py` es el **módulo de filtrado y procesamiento de seguridad**. Cumple dos tareas principales:

1. **Limpieza de JSON**: verifica la integridad y seguridad de los chats y del dataset de entrenamiento, eliminando contenido ofensivo o dañino y corrigiendo la estructura del dataset.
2. **Prueba de seguridad del modelo**: ejecuta una prueba breve sobre el modelo para detectar respuestas sensibles, eróticas, sexuales, dañinas o peligrosas usando un doble filtro (palabras ofensivas + modelo LLM-base).

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json` | Directorio | Base de archivos JSON. |
| `json/conversaciones` | Directorio | Conversaciones individuales (entrada). |
| `json/entrenamiento/dataset.json` | Archivo | Dataset a limpiar/validar (entrada/salida). |
| `models/LLM` | Directorio | Modelo a probar (preferido). |
| `models/LLM-base` | Directorio | Modelo alternativo y usado como segundo filtro. |
| `test` | Directorio | Guarda las respuestas de la prueba (`respuesta_pregunta_N.txt`). |

## Tecnologías

- **Python**.
- **os / json / shutil** (estándar) para rutas, lectura/escritura de JSON y mover/eliminar modelos.
- **unicodedata** (estándar) para normalizar texto (minúsculas y sin acentos).
- **pathlib / typing** (estándar).
- **Transformers / PyTorch** (importados dinámicamente) para cargar el modelo y generar respuestas en la prueba de seguridad.

## Listas de palabras inseguras

Existen dos listas con propósitos distintos (todas son raíces/stems para atrapar conjugaciones):

| Variable | Uso |
|----------|-----|
| `filtros` | Raíces usadas por `es_texto_seguro`, `es_conversacion_segura` y las funciones de limpieza de JSON. |
| `palabras_filtro` | Palabras usadas por el **primer filtro** de las respuestas del modelo (`pasar_filtro_palabras`). |
| `_FILTROS_DEFECTO` | Copia de `filtros` usada como valor por defecto en las funciones. |

Ejemplos: `odio`, `violencia`, `discrimina`, `insult`, `amenaz`, `maltrat`, `ofens`, `suicid`, `hacke`, `bomba`, `droga`, `arma`, `matar`, `asesin`, `viol`, `secuestr`, `pedofil`, `nazi`, `racist`, `estafa`, `fraude`, `terror`, `narcotraf`, `pornograf`, `autoagres`, entre otras.

## Variables y constantes

| Nombre | Descripción |
|--------|-------------|
| `PREGUNTAS_PRUEBA` | Lista de 5 preguntas sensibles para la prueba de seguridad del modelo. |
| `PROMPT_MODERACION` | System prompt del segundo filtro; pide responder en formato `<n>.si` o `<n>.no`. |
| `_cache_modelos` | Caché de modelos cargados `{ruta: (modelo, tokenizer)}`. |

## Funciones de normalización y filtrado

| Función | Rol |
|---------|-----|
| `_normalizar(texto)` | Minúsculas y sin acentos (normalización NFD). |
| `_contiene_palabra_ofensiva(texto, palabra)` | Verifica si un texto normalizado contiene una raíz ofensiva. |
| `es_texto_seguro(texto, filtros_uso)` | True si el texto no contiene palabras ofensivas (no str → False). |
| `es_conversacion_segura(conversacion, filtros_uso)` | Evalúa `input`/`output`/`text`/`content`/`pregunta`/`respuesta` de cada mensaje. |

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Funciones decoradas con sus valores por defecto ante error:

| Función | Default |
|---------|---------|
| `cargar_json` | `None` |
| `verificar_chat`, `validar_estructura_dataset`, `verificar_integridad_dataset`, `filtro_seguridad`, `limpiar_dataset`, `_directorio_tiene_modelo`, `filtrar_modelo` | `False` |
| `listar_chats` | `[]` |
| `_cargar_modelo` | `None` |
| `generar_respuesta_modelo`, `preguntar_modelo` | `""` |
| `pasar_filtro_llm_base` | `"error"` |
| `prueba_seguridad_modelo` | `{}` |

Los métodos de la clase `FiltroSeguridad` (`filtrar_modelo`, `listar_chats`, `filtro_seguridad`) también están decorados. El bloque `if __name__ == "__main__":` se envuelve en `try/except` que imprime `[ERROR FATAL]`.

## Funciones de integridad de JSON

| Función | Rol |
|---------|-----|
| `cargar_json(ruta_json)` | Carga un JSON y verifica su integridad; `None` si está corrupto. |
| `verificar_chat(ruta_archivo, filtros_uso)` | Valida que un chat no esté corrupto ni sea inseguro. |
| `listar_chats(directorio_json, filtros_uso)` | Lista los `.json` de `json/conversaciones` y elimina los corruptos/inseguros. |
| `_actualizar_ids(dataset)` | Re-numera los IDs con formato `conv_000001`. |
| `validar_estructura_dataset(directorio_json)` | Convierte un dataset en lista a diccionario con IDs. |
| `verificar_integridad_dataset(directorio_json, filtros_uso)` | Filtra `dataset.json` y elimina conversaciones inseguras. |
| `filtro_seguridad(directorio_json, filtros_uso)` | Orquesta: valida estructura, limpia chats y limpia el dataset. |
| `limpiar_dataset(directorio_json, filtros_uso)` | Alias de `verificar_integridad_dataset`. |

## Funciones de prueba de seguridad del modelo

| Función | Rol |
|---------|-----|
| `_directorio_tiene_modelo(ruta)` | True si el directorio contiene pesos (`.safetensors`, `.bin`, `.pt`, `.ckpt`). |
| `_obtener_ruta_modelo_prueba(ruta_modelo)` | Prioriza `models/LLM`; si está vacío o no existe, usa `models/LLM-base`. |
| `_cargar_modelo(ruta_modelo)` | Carga y cachea `(modelo, tokenizer)` desde Transformers. |
| `generar_respuesta_modelo(ruta_modelo, mensajes, max_new_tokens=180)` | Genera una respuesta usando la plantilla de chat del modelo. |
| `preguntar_modelo(pregunta, ruta_modelo)` | Envía una pregunta individual al modelo. |
| `guardar_respuesta_txt(directorio_test, numero, pregunta, respuesta)` | Guarda la pregunta y respuesta en `respuesta_pregunta_N.txt`. |
| `pasar_filtro_palabras(texto, palabras_uso)` | **Filtro 1**: verifica que la respuesta no contenga palabras de `palabras_filtro`. |
| `pasar_filtro_llm_base(ruta_txt, numero, ruta_modelo_llm_base)` | **Filtro 2**: envía el txt al LLM-base con `PROMPT_MODERACION` (veredicto `"N.si"`/`"N.no"`). |
| `prueba_seguridad_modelo(ruta_modelo, ruta_modelo_llm_base)` | Ejecuta la prueba completa y devuelve un reporte. |

### Flujo de `prueba_seguridad_modelo`

1. Envía cada `PREGUNTA_PRUEBA` al modelo (prioriza `models/LLM`).
2. Guarda cada respuesta en `test/respuesta_pregunta_N.txt`.
3. **Filtro 1**: `pasar_filtro_palabras` (algoritmo de palabras ofensivas).
4. **Filtro 2**: si pasa el filtro 1, envía el txt al LLM-base con `PROMPT_MODERACION`; si no, el veredicto es `"no (rechazado por filtro de palabras)"`.
5. Retorna un reporte con `modelo` y una lista de `preguntas` (número, pregunta, respuesta, archivo, resultado de filtro y veredicto).

## Clase principal: `FiltroSeguridad`

Mantiene compatibilidad con el código existente. Recibe `modelo_base` (no se usa para evaluar; la evaluación real está simulada con listas de palabras).

| Método | Rol |
|--------|-----|
| `__init__(modelo_base, ...)` | Inicializa directorios `LLM`, `LLM-beta`, `json` y los crea si no existen. |
| `filtrar_modelo(modelo_path)` | Evalúa un modelo de `LLM-beta`; si es seguro lo mueve a `LLM`, si no lo elimina. |
| `_evaluar_seguridad_base(texto)` | Busca palabras de `self.filtros`; retorna `"si"`/`"no"`. |
| `_eliminar_modelo(ruta)` | Elimina un archivo o directorio de modelo. |
| `_mover_modelo_seguro(ruta)` | Mueve un modelo seguro a `LLM` con `shutil.move`. |
| `listar_chats()` | Lista y limpia los `.json` de `json/conversaciones`. |
| `limpiar_dataset()` | Alias de `verificar_integridad_dataset`. |
| `verificar_integridad_dataset()` | Filtra el dataset y elimina conversaciones inseguras. |
| `filtro_seguridad()` | Ejecuta la limpieza completa de JSON. |
| `procesar_todos_modelos()` | Procesa todos los modelos de `LLM-beta`. |
| `limpiar_conversaciones_carpeta()` | Verifica las conversaciones de `json/conversaciones`. |
| `validar_estructura_dataset()` | Valida y corrige la estructura del dataset. |
| `cargar_json(ruta)` / `_es_texto_seguro` / `_verificar_chat` / `_es_conversacion_segura` / `_actualizar_ids` | Wrappers de las funciones independientes con los filtros de la instancia. |

## Función independiente (compatibilidad)

| Función | Rol |
|---------|-----|
| `filtrar_modelo(modelo_path)` | Filtra un modelo usando una instancia temporal de `FiltroSeguridad(None)`. |

## Comportamientos relevantes

- `es_texto_seguro` retorna `False` si el texto no es `str`.
- `verificar_integridad_dataset` soporta dataset como dict (claves = IDs) o como lista; cuenta los mensajes eliminados y regenara los IDs al guardar.
- `listar_chats` **elimina** los archivos corruptos o inseguros al listar.
- En `filtrar_modelo`, el bucle recorre `respuestas_generadas`, que actualmente está **vacía**, por lo que cualquier modelo existente en `LLM-beta` se considera seguro y se mueve a `LLM`.
- La evaluación con el modelo base está **simulada**: se usa la lista de palabras en lugar del modelo real (los comentarios del código indican dónde conectar la implementación real).

## Notas

- En el `if __name__ == "__main__":` se ejecuta un flujo de ejemplo: `filtro_seguridad()` y luego `prueba_seguridad_modelo()`.
- Los modelos se cachean en `_cache_modelos` para evitar cargas repetidas.
