# llm.md

Archivo de documentación técnica para `chat/llm.py`.

## ¿Qué hace el archivo?

`chat/llm.py` es el **motor principal del chat LLM**. Levanta un servidor web local (Flask) que expone una interfaz de chat para interactuar con el modelo de lenguaje, y gestiona la persistencia de las conversaciones en JSON.

Flujo general:

1. Carga el modelo causal local (`models/LLM-base`) o, si existe `OPENAI_API_KEY`, llama a una API compatible con OpenAI.
2. Expone endpoints REST para listar chats, crear, cargar, enviar mensajes y calificar respuestas.
3. Almacena cada conversación como un archivo JSON en `json/conversaciones`.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `chat/index.html` | Archivo | HTML servido en la ruta raíz `GET /`. |
| `models/LLM-base` | Directorio | Modelo causal local (cargado con `AutoModelForCausalLM`). |
| `json/conversaciones` | Directorio | Conversaciones en JSON (creado automáticamente). |

## Tecnologías

- **Python**.
- **Flask**: servidor web (`Flask`, `Response`, `jsonify`, `request`, `stream_with_context`).
- **Transformers** (Hugging Face): `AutoModelForCausalLM`, `AutoTokenizer`, `GenerationConfig`, `TextIteratorStreamer`.
- **requests**: cliente HTTP para la API OpenAI.
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
| `generar_respuesta_stream(mensajes)` | Genera respuesta en streaming con el modelo local. |
| `completar_openai_stream(mensajes)` | Streams desde OpenAI si hay API key, si no usa el modelo local. |
| `run_server()` | Inicia el servidor en `127.0.0.1:8000`. |

## Estructura de un mensaje guardado

```json
{
    "id": 1,
    "date": "2026-08-04T12:22:45.766086",
    "input": "pregunta del usuario",
    "output": "respuesta del modelo",
    "qualification": "neutra"
}
```

## Comportamiento del servidor

- Puerto: `8000`, host: `127.0.0.1`, `debug=False`.
- `/enviar` usa streaming (`stream_with_context`), desactivando buffering con `X-Accel-Buffering: no`.
- Si no hay respuesta del modelo, devuelve el texto `"no hay modelo disponible en local"`.
- Preferencia de generación: `max_new_tokens=512`, `do_sample=True`, `temperature=0.7`.

## Notas

- `DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"`.
- `API_BASE_URL = "http://localhost:11434/v1"` (API local compatible con OpenAI).
- Al iniciar como ejecutable, intenta cargar el modelo local; si falla, termina con `SystemExit(1)`.
- Los directorios `json/conversaciones` y `models/LLM-base` se crean automáticamente con `os.makedirs`.