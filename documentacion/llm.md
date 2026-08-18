# llm.md

Archivo de documentación técnica para `chat/llm.py`.

## ¿Qué hace el archivo?

`chat/llm.py` es el **motor principal del chat LLM**. Levanta un servidor web local (Flask) que expone una interfaz de chat para interactuar con el modelo de lenguaje, y gestiona la persistencia de las conversaciones en JSON.

Flujo general:

1. Carga el modelo causal local (`models/LLM-base`).
2. Expone endpoints REST para listar chats, crear, cargar, enviar mensajes y calificar respuestas.
3. Almacena cada conversación como un archivo JSON en `json/conversaciones`.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `chat/index.html` | Archivo | HTML servido en la ruta raíz `GET /`. |
| `models/LLM-base` | Directorio | Modelo causal local (cargado con `AutoModelForCausalLM`). |
| `json/conversaciones` | Directorio | Conversaciones en JSON (creado automáticamente). |
| `json/contexto_global.json` | Archivo | Contexto global entre chats en formato OpenAI `messages` (creado automáticamente). |

## Tecnologías

- **Python**.
- **Flask**: servidor web (`Flask`, `Response`, `jsonify`, `request`, `stream_with_context`).
- **Transformers** (Hugging Face): `AutoModelForCausalLM`, `AutoTokenizer`, `GenerationConfig`, `TextIteratorStreamer`.
- **os, json, re, datetime, io, threading, queue**.

## Endpoints de la API

| Método | Ruta | Función | Acción |
|--------|------|---------|--------|
| GET | `/` | `index()` | Sirve `index.html`. |
| GET | `/chats` | `chats()` | Lista chats y el chat actual. |
| POST | `/chat` | `chat()` | Carga una conversación por nombre. |
| POST | `/nuevo` | `nuevo()` | Crea un nuevo chat (`conversacion-N`). |
| POST | `/enviar` | `enviar()` | Envía un mensaje al modelo (streaming). |
| POST | `/calificar` | `calificar()` | Actualiza la `qualification` de un mensaje. |

## Funciones principales

| Función | Rol |
|---------|-----|
| `nombre_archivo(nombre)` | Sanitiza el nombre y retorna la ruta JSON. |
| `listar_chats()` | Lista los archivos `.json` de conversaciones. |
| `cargar_chat(nombre)` | Carga una conversación desde JSON. |
| `guardar_chat(nombre, conversacion)` | Guarda una conversación en JSON. |
| `generar_respuesta_stream(mensajes)` | Genera respuesta en streaming con el modelo local (o mensaje de contexto global si no hay modelo). |
| `cargar_contexto_global()` | Carga (o crea) el contexto global entre chats en formato OpenAI. |
| `guardar_contexto_global(contexto)` | Guarda el contexto global entre chats. |
| `agregar_a_contexto_global(usuario, respuesta)` | Agrega un par user->assistant al contexto global y lo recorta a `MAX_CONTEXTO_GLOBAL`. |
| `mensajes_con_contexto(mensajes)` | Prefija el contexto global a los mensajes actuales del chat. |
| `run_server()` | Inicia el servidor en `127.0.0.1:8000`. |

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Funciones decoradas:

- `nombre_archivo`, `guardar_chat`, `run_server` → `@manejar_errores` (relanzan como `ErrorAPACMA`).
- `listar_chats`, `cargar_chat` → `@manejar_errores(default=[])` (devuelven lista vacía ante error).

El bloque `if __name__ == "__main__":` tiene `try/except` propio para la carga del modelo: si falla, imprime el error y continúa sirviendo con el contexto global (ya no termina con `SystemExit`).

## Estructura de una conversación guardada

Las conversaciones se guardan en **formato estándar OpenAI `messages`** (ver `formato_openai.md`):

```json
{
    "messages": [
        {
            "role": "user",
            "content": "pregunta del usuario"
        },
        {
            "role": "assistant",
            "content": "respuesta del modelo",
            "qualification": "neutra"
        }
    ]
}
```

- `id` y `date` (opcionales) se asignan al par user->assistant y se comparten entre ambos.
- La `qualification` vive solo en el mensaje `assistant` y la asigna la ruta `/calificar`.

## Comportamiento del servidor

- Puerto: `8000`, host: `127.0.0.1`, `debug=False`.
- `/enviar` usa streaming (`stream_with_context`), desactivando buffering con `X-Accel-Buffering: no`.
- Si no hay respuesta del modelo, devuelve el texto `"no hay modelo disponible en local"`.
- Preferencia de generación: `max_new_tokens=512`, `do_sample=True`, `temperature=0.7`.

## Contexto global entre chats

Cuando el modelo local **no** está disponible en `models/LLM-base`, el servidor no se detiene: usa un **contexto global entre chats** guardado en `json/contexto_global.json` en el mismo formato OpenAI `messages` que usan las empresas (un mensaje `system` inicial más memoria de pares `user`/`assistant` recientes). Esto permite mantener continuidad de información entre conversaciones sin depender del modelo local.

- `cargar_contexto_global()` crea el archivo con un `system` inicial si no existe.
- `agregar_a_contexto_global(usuario, respuesta)` guarda cada intercambio y recorta a los últimos `MAX_CONTEXTO_GLOBAL = 20` mensajes.
- `mensajes_con_contexto(mensajes)` antepone ese contexto a los mensajes del chat antes de llamar al modelo (`/enviar`).
- Sin modelo local, `/enviar` responde `"no hay modelo disponible en local, se mantiene el contexto global"`, pero el contexto global se sigue actualizando.

## Notas

- No se usa ninguna API externa (ni OpenAI ni similares); todo es local.
- Al iniciar como ejecutable, intenta cargar el modelo local; si falla, continúa sirviendo con el contexto global.
- Los directorios `json/conversaciones` y `models/LLM-base` se crean automáticamente con `os.makedirs`.