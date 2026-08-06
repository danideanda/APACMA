# limites.md

Archivo de documentación técnica para `limites.py`.

## ¿Qué hace el archivo?

`limites.py` es el **control de límite de tokens del dataset de entrenamiento**. Cuenta los tokens del dataset y, si supera el límite configurado, elimina los mensajes más largos (más de 500 tokens) hasta volver a estar dentro del límite.

Flujo general:

1. Cuenta el total de tokens de `json/entrenamiento/dataset.json`.
2. Compara el total con el límite (`3_000_000`).
3. Si se excede, elimina mensajes de más de 500 tokens hasta quedar dentro del límite.
4. Guarda el dataset actualizado.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `models/LLM-base` | Directorio | Modelo base del que se carga el tokenizador. |
| `json/entrenamiento/dataset.json` | Archivo | Dataset de entrenamiento (entrada/salida). |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer` con `use_fast=True`.
- **json** (estándar) para lectura/escritura del dataset.

## Variables globales

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `tokenizer` | `AutoTokenizer` | Tokenizador del modelo base cargado de `models/LLM-base`. |
| `limite_tokens` | `3_000_000` | Límite máximo de tokens permitido en el dataset. |
| `tokens_actuales` | `0` | Contador global de tokens del dataset. |
| `dataset` | `list` | Dataset cargado de `json/entrenamiento/dataset.json`. |

## Funciones principales

| Función | Rol |
|---------|-----|
| `contar_tokens()` | Reinicia `tokens_actuales` y suma los tokens de cada mensaje usando el tokenizador; imprime el total. |
| `eliminar_mensajes()` | Recorre una copia del dataset y elimina los mensajes con más de 500 tokens, recalculando el contador tras cada borrado, hasta que `tokens_actuales <= limite_tokens`. Al final guarda el dataset con `indent=4`. |
| `limites_tokens()` | Verifica si `tokens_actuales` supera `limite_tokens`; si es así llama a `eliminar_mensajes()`, si no imprime que el dataset está dentro del límite. |

## Detalle de `eliminar_mensajes`

- Itera sobre `dataset[:]` (copia) para poder eliminar elementos sin problemas de iteración.
- Cada mensaje se tokeniza con `tokenizer(mensaje["text"], return_tensors="pt")` y el conteo se obtiene con `input_ids.size(1)`.
- Tras cada eliminación se recalcula el total con `contar_tokens()`; si el total ya está dentro del límite, se detiene con `break`.
- Finalmente guarda el dataset con `json.dump(..., ensure_ascii=False, indent=4)`.

## Flujo de ejecución

```
if __name__ == "__main__":
     │
     ▼
contar_tokens()  →  calcula tokens_actuales
     │
     ▼
limites_tokens()  →  si tokens_actuales > limite_tokens → eliminar_mensajes()
                                    │
                                    └─ si no → "El dataset está dentro del límite."
```

## Notas

- El script asume que el dataset es una **lista** de mensajes, cada uno con un campo `"text"`.
- `tokens_actuales` solo tiene valor correcto después de ejecutar `contar_tokens()`; `limites_tokens()` depende de ese cálculo previo.
- Si el tokenizador o el archivo de dataset no existen, el script fallará en la importación o lectura inicial.
