# limites.md

Archivo de documentación técnica para `limites.py`.

## ¿Qué hace el archivo?

`limites.py` es el **control de límite de tokens del dataset de entrenamiento**. Cuenta los tokens del dataset y, si supera el límite configurado, elimina mensajes siguiendo una **jerarquía de prioridades** hasta volver a estar dentro del límite.

Flujo general:

1. Cuenta el total de tokens de `json/entrenamiento/dataset.json`.
2. Compara el total con el límite (`3_000_000`).
3. Si se excede, aplica la jerarquía de eliminación:
   - **Prioridad 0.9:** Elimina mensajes positivos de más de 500 tokens.
   - **Prioridad 0.6:** Elimina mensajes positivos repetidos menos de 3 veces.
   - **Prioridad 0.3:** Elimina mensajes negativos repetidos menos de 3 veces.
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
| `eliminar_mensajes()` | Aplica la jerarquía de eliminación en tres fases, con límite de 1,000 eliminaciones por fase, hasta que el dataset esté dentro del límite de tokens. |
| `limites_tokens()` | Verifica si `tokens_actuales` supera `limite_tokens`; si es así llama a `eliminar_mensajes()`, si no imprime que el dataset está dentro del límite. |

## Jerarquía de eliminación

La función `eliminar_mensajes()` aplica las siguientes reglas en orden de prioridad:

### Fase 1: Prioridad 0.9 (Más prioritaria)
- **Condición:** `cantidad_tokens > 500` AND `qualification == "positive"`
- **Acción:** Eliminar el mensaje.
- **Razón:** Los mensajes positivos largos consumen muchos tokens y su valor no justifica el espacio.

### Fase 2: Prioridad 0.6 (Prioridad media)
- **Condición:** `dataset.count(mensaje) < 3` AND `qualification == "positive"`
- **Acción:** Eliminar el mensaje.
- **Razón:** Los mensajes positivos que aparecen menos de 3 veces son menos representativos y tienen menor peso en el aprendizaje.

### Fase 3: Prioridad 0.3 (Menos prioritaria)
- **Condición:** `dataset.count(mensaje) < 3` AND `qualification == "negative"`
- **Acción:** Eliminar el mensaje.
- **Razón:** Los mensajes negativos repetidos menos de 3 veces son ruido que no aporta valor al entrenamiento.

## Seguridad y límites

| Mecanismo | Valor | Propósito |
|-----------|-------|-----------|
| **Límite de eliminaciones por fase** | `1_000` | Evita que el script se ejecute indefinidamente si el dataset es muy grande. |
| **Break al alcanzar límite** | `if tokens_actuales <= limite_tokens: break` | Detiene la ejecución en cuanto el dataset está dentro del límite, sin eliminar más de lo necesario. |
| **Break al alcanzar tope de eliminaciones** | `elif mensajes_eliminados >= 1_000: break` | Limita el número de eliminaciones por ejecución para controlar el tiempo de procesamiento. |

## Detalle de `eliminar_mensajes`

### Fase 1: Eliminar mensajes positivos largos

- Itera sobre `dataset[:]` (copia) para poder eliminar elementos sin problemas de iteración.
- Cada mensaje se tokeniza con `tokenizer(mensaje["text"], return_tensors="pt")`.
- El conteo se obtiene con `input_ids.size(1)`.
- Si `cantidad_tokens > 500` y `qualification == "positive"`, se elimina el mensaje.
- Tras cada eliminación:
  - Se incrementa `mensajes_eliminados`.
  - Se recalcula el total con `contar_tokens()`.
  - Si `tokens_actuales <= limite_tokens` o `mensajes_eliminados >= 1_000`, se detiene el bucle.

### Fase 2: Eliminar mensajes positivos repetidos

- Itera sobre `dataset[:]` (copia) para poder eliminar elementos sin problemas de iteración.
- Si `dataset.count(mensaje) < 3` y `qualification == "positive"`, se elimina el mensaje.
- Tras cada eliminación:
  - Se incrementa `mensaje_contador`.
  - Se recalcula el total con `contar_tokens()`.
  - Si `tokens_actuales <= limite_tokens` o `mensaje_contador >= 1_000`, se detiene el bucle.

### Fase 3: Eliminar mensajes negativos repetidos

- Itera sobre `dataset[:]` (copia) para poder eliminar elementos sin problemas de iteración.
- Si `dataset.count(mensaje) < 3` y `qualification == "negative"`, se elimina el mensaje.
- Tras cada eliminación:
  - Se incrementa `mensaje_contador`.
  - Se recalcula el total con `contar_tokens()`.
  - Si `tokens_actuales <= limite_tokens` o `mensaje_contador >= 1_000`, se detiene el bucle.

### Guardado final

- Tras completar las tres fases, se guarda el dataset con `json.dump(dataset, f, ensure_ascii=False, indent=4)`.

## Flujo de ejecución

```
if __name__ == "__main__":
     │
     ▼
contar_tokens()  →  calcula tokens_actuales
     │
     ▼
limites_tokens()  →  if tokens_actuales > limite_tokens:
                         │
                         ▼
                    eliminar_mensajes()
                         │
                         ├─ Fase 1: Eliminar positivos largos (>500 tokens)
                         │    └─ Hasta 1,000 eliminaciones o bajo el límite
                         │
                         ├─ Fase 2: Eliminar positivos repetidos (<3 veces)
                         │    └─ Hasta 1,000 eliminaciones o bajo el límite
                         │
                         ├─ Fase 3: Eliminar negativos repetidos (<3 veces)
                         │    └─ Hasta 1,000 eliminaciones o bajo el límite
                         │
                         └─ Guardar dataset actualizado
                    │
                    └─ si no → "El dataset está dentro del límite."
```

## Notas

- El script asume que el dataset es una **lista** de mensajes, cada uno con un campo `"text"` y `"qualification"`.
- `tokens_actuales` solo tiene valor correcto después de ejecutar `contar_tokens()`; `limites_tokens()` depende de ese cálculo previo.
- Si el tokenizador o el archivo de dataset no existen, el script fallará en la importación o lectura inicial.
- Los mensajes con `qualification` desconocida no se eliminan en ninguna fase.
- Los mensajes negativos largos (>500 tokens) NO se eliminan por tamaño, solo por repetición.
- Los mensajes positivos repetidos (<3 veces) se eliminan ANTES que los negativos repetidos, según la jerarquía.

## Limitaciones conocidas

| Limitación | Impacto | Posible mejora |
|------------|---------|----------------|
| `dataset.count(mensaje)` cuenta objetos completos, no texto similar | No detecta mensajes con mismo contenido pero diferentes metadatos | Normalizar texto o usar embeddings para similitud |
| Recalcular tokens tras cada eliminación | Ineficiente para datasets grandes | Recalcular al final de cada fase o en lotes |
| Límite de 1,000 eliminaciones por fase | Puede requerir múltiples ejecuciones | Hacerlo configurable o ejecutar en bucle hasta cumplir |
| Sin logs persistentes | No se puede auditar qué se eliminó | Guardar archivo de log con fecha y mensajes eliminados |
| Sin respaldo del dataset | Si el script falla, se pierden datos | Guardar respaldo antes de modificar |

## Mejoras propuestas (futuras)

1. **Campo de relevancia:** Agregar `relevance_score` (0-1) para priorizar eliminación de mensajes con baja relevancia.
2. **Normalización de texto:** Para detectar repeticiones reales independientemente de mayúsculas o espacios.
3. **Logs persistentes:** Guardar historial de eliminaciones en `logs/pruning_log.txt`.
4. **Respaldo automático:** Crear `dataset_backup.json` antes de cualquier modificación.
5. **Configuración externa:** Mover `limite_tokens`, `max_eliminaciones` y jerarquía a un archivo `config.json`.
6. **Progreso visual:** Agregar barra de progreso para datasets grandes.
