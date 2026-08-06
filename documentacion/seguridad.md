# seguridad.md

Archivo de documentación técnica para `seguridad.py`.

## ¿Qué hace el archivo?

`seguridad.py` es el **módulo de filtrado y procesamiento de seguridad**. Garantiza que las respuestas del modelo de IA sean seguras y limpia los datasets de conversaciones eliminando contenido inapropiado, ofensivo o dañino.

Flujo general:

1. Filtra los modelos entrenados en `LLM-beta` evaluando su seguridad; los aprobados se mueven a `LLM`.
2. Limpia el dataset (`dataset.json`) eliminando conversaciones inseguras.
3. Procesa las conversaciones individuales de `json/conversaciones`.
4. Valida y corrige la estructura del dataset.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `LLM` | Directorio | Modelos aprobados como seguros (se crea automáticamente). |
| `LLM-beta` | Directorio | Modelos en prueba pendientes de filtrar (se crea automáticamente). |
| `json` | Directorio | Base de archivos JSON (se crea automáticamente). |
| `json/conversaciones` | Directorio | Conversaciones individuales (entrada). |
| `json/entrenamiento/dataset.json` | Archivo | Dataset a limpiar/validar (entrada/salida). |

## Tecnologías

- **Python**.
- **os / json** (estándar) para rutas y lectura/escritura de JSON.
- **shutil** (estándar) para mover y eliminar modelos.
- **pathlib / typing** (estándar).

## Clase principal: `FiltroSeguridad`

| Método | Rol |
|--------|-----|
| `__init__(modelo_base, ...)` | Inicializa directorios y crea las carpetas si no existen. |
| `filtrar_modelo(modelo_path)` | Evalúa un modelo de `LLM-beta`; si es seguro lo mueve a `LLM`, si no lo elimina. |
| `_evaluar_seguridad_base(texto)` | Verifica si un texto contiene palabras inseguras; retorna `"si"`/`"no"`. |
| `_eliminar_modelo(ruta)` | Elimina un archivo o directorio de modelo. |
| `_mover_modelo_seguro(ruta)` | Mueve un modelo seguro a `LLM`. |
| `listar_chats()` | Lista los `.json` de `json/conversaciones`. |
| `limpiar_dataset()` | Filtra `dataset.json` y elimina conversaciones inseguras. |
| `_es_conversacion_segura(conversacion)` | Evalúa cada mensaje de una conversación. |
| `_actualizar_ids(dataset)` | Re-numera los IDs con formato `conv_000001`. |
| `procesar_todos_modelos()` | Filtra todos los modelos de `LLM-beta`. |
| `limpiar_conversaciones_carpeta()` | Elimina los chats inseguros de `json/conversaciones`. |
| `validar_estructura_dataset()` | Convierte un dataset en lista a diccionario con IDs. |

## Funciones independientes (compatibilidad)

Envuelven a la clase `FiltroSeguridad` para mantener compatibilidad con el código existente (todas crean una instancia temporal con `modelo_base=None`):

| Función | Rol |
|---------|-----|
| `filtrar_modelo(modelo_path)` | Filtra un modelo (usa una instancia temporal de la clase). |
| `limpiar_dataset()` | Limpia el dataset completo. |
| `listar_chats()` | Lista los archivos de chat. |
| `validar_estructura_dataset()` | Valida la estructura del dataset. |

## Lista de palabras inseguras

`_evaluar_seguridad_base()` detecta estas palabras (comparando en minúsculas):

`odio`, `violencia`, `discriminacion`, `insulto`, `amenaza`, `maltrato`, `ofensa`

## Comportamiento de `limpiar_dataset`

- Soporta `dataset.json` como diccionario (claves = IDs) o como lista.
- Las conversaciones sin `mensajes` se consideran inseguras.
- Si una conversación falla la evaluación, se descarta y se cuentan sus mensajes como eliminados.
- Después de filtrar, re-genera los IDs con `_actualizar_ids` y guarda con `ensure_ascii=False, indent=2`.
- Si la estructura del dataset no es ni dict ni list, retorna `False`.

## Comportamiento de `filtrar_modelo`

- El modelo debe existir en `LLM-beta`; si no, retorna `False`.
- Define `respuestas_prueba`, pero recorre `respuestas_generadas`, que actualmente está **vacía**. Por eso el bucle de verificación nunca se ejecuta.
- En la práctica, cualquier modelo existente en `LLM-beta` se considera seguro y se mueve a `LLM`.
- En la implementación real se conectará `modelo_generado.generate(...)` para llenar `respuestas_generadas` y `modelo_base.evaluar_seguridad(...)` para la evaluación.

## Comportamiento de `validar_estructura_dataset`

- Si `dataset.json` es una lista, la convierte a diccionario con `_actualizar_ids` y guarda con `ensure_ascii=False, indent=2`.
- Si es un diccionario, la considera correcta.
- Cualquier otra estructura retorna `False`.

## Notas

- La evaluación con el modelo base está **simulada** (`modelo_base` se pasa pero se usa la lista de palabras).
- Los métodos privados (`_...`) contienen comentarios que indican dónde conectar la implementación real con el framework.
- `_mover_modelo_seguro` usa `shutil.move` tanto para archivos como para directorios.
- `limpiar_conversaciones_carpeta` elimina chats que no sean un dict, que no pasen la evaluación o que tengan JSON inválido.
- En el `if __name__ == "__main__":` se ejecuta un flujo de ejemplo: validar estructura, listar chats, limpiar dataset, limpiar conversaciones y procesar modelos.
