# formato_openai.md

Archivo de documentación técnica para `formato_openai.py`.

## ¿Qué hace el archivo?

`formato_openai.py` es el **módulo central de normalización de conversaciones** del proyecto. Garantiza que **todos los chats se lean y escriban en el formato estándar OpenAI `messages`**, el mismo esquema que usan OpenAI y la industria para guardar conversaciones y datasets de fine-tuning.

El módulo actúa como única puerta de entrada/salida para los archivos de `json/conversaciones`, de modo que el resto de módulos (`chat/llm.py`, `json_script.py`, `clasificador.py`, `DNAPAN.py`, `fine.py`, `seguridad.py`) no tienen que conocer los detalles del formato: solo llaman a estos helpers.

## Formato estándar (OpenAI `messages`)

Cada archivo de conversación es un objeto JSON con una clave `messages`, una lista de mensajes:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "hola"
    },
    {
      "role": "assistant",
      "content": "¿Cómo puedo ayudarte?",
      "qualification": "neutra"
    }
  ]
}
```

| Campo | Descripción |
|-------|-------------|
| `role` | `system`, `user` o `assistant`. |
| `content` | Texto del mensaje. |
| `qualification` | Opcional, **solo en mensajes `assistant`**: `positive`, `negative` o `neutra`. |
| `id` | Opcional: identificador del par user->assistant (ambos mensajes del par lo comparten). |
| `date` | Opcional: fecha en formato ISO 8601. |

> La `qualification` se conserva en el mensaje `assistant` porque es la **respuesta** la que se califica.

## Formato antiguo (pares input/output)

Antes de este módulo, las conversaciones eran una **lista de pares**:

```json
[
  {
    "id": 1,
    "date": "2026-08-08T18:27:32.795863",
    "input": "hola",
    "output": "¿Cómo puedo ayudarte?",
    "qualification": "neutra"
  }
]
```

`formato_openai.py` lee automáticamente ambos formatos y siempre devuelve/escribe en formato `messages`, por lo que no hace falta migrar archivos manualmente.

## Funciones principales

### Detección de formato

| Función | Descripción |
|---------|-------------|
| `es_formato_mensajes(conversacion)` | `True` si la conversación ya usa formato OpenAI (`messages`). |
| `es_formato_antiguo(conversacion)` | `True` si la conversación usa pares `input`/`output`. |

### Normalización

| Función | Descripción |
|---------|-------------|
| `normalizar_a_mensajes(conversacion)` | Convierte cualquier formato (messages, lista de mensajes `{role, content}`, o pares `input/output`) al estándar `{"messages": [...]}`. |
| `leer_conversacion(ruta)` | Lee un archivo JSON y lo devuelve en formato `messages`. |
| `guardar_conversacion(ruta, conversacion)` | Guarda una conversación en formato `messages`. |

### Acceso

| Función | Descripción |
|---------|-------------|
| `obtener_mensajes(conversacion)` | Devuelve la lista de mensajes `{role, content[, qualification]}`. |
| `mensajes_para_llm(conversacion)` | Devuelve solo `{role, content}` (sin metadatos) para enviar al LLM. |
| `pares_entrenamiento(conversacion)` | Devuelve los pares `user -> assistant` como `{id, date, input, output, qualification}` (la qualification se toma del assistant). |

## Cómo se integra con el resto del proyecto

| Módulo | Uso de `formato_openai` |
|--------|--------------------------|
| `chat/llm.py` | Lee/escribe chats con `leer_conversacion`/`guardar_conversacion`; construye el payload del modelo con `mensajes_para_llm`; califica buscando en `obtener_mensajes`. |
| `json_script.py` | Recorre los mensajes con `obtener_mensajes` para extraer los IDs calificados. |
| `clasificador.py` | Obtiene los pares con `pares_entrenamiento` para generar el dataset y aplicar DNAPAN. |
| `DNAPAN.py` | Procesa los pares user->assistant con `pares_entrenamiento`. |
| `fine.py` | Acepta conversaciones en formato `messages` además del formato clásico de dataset. |
| `seguridad.py` | `_obtener_mensajes` soporta la clave `messages` además de `mensajes`. |

## Notas

- La conversión antigua -> nueva asigna el mismo `id` y `date` a ambos mensajes del par y coloca la `qualification` en el `assistant`.
- `_limpiar_mensaje()` elimina prefijos residuales `user\n`/`assistant\n` que pudieran quedar en el `content` al migrar.
- Este módulo no maneja errores propios con `manejar_errores` (a diferencia del resto); los errores de lectura de archivo se propagan para que el llamador los gestione.
