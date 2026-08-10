# fine.md

Archivo de documentación técnica para `fine.py`.

## ¿Qué hace el archivo?

`fine.py` implementa el **fine-tuning del modelo LLM** usando la técnica **LoRA** (Low-Rank Adaptation). Toma un modelo base causal, un dataset en formato JSON y entrena el modelo ajustado (fine-tuned), guardándolo en un directorio de salida. Además genera un registro (`model.json`) con la versión y fecha del último modelo entrenado.

La novedad principal es que ahora **toma en cuenta el campo `qualification`** de cada mensaje del dataset:

- `qualification == "positive"` → **SFT estándar**: el modelo aprende a producir esa respuesta (refuerza el buen comportamiento).
- `qualification == "negative"` → **Unlikelihood loss**: el modelo **reduce la probabilidad** de producir esa respuesta dañina (ejemplo de "cómo NO responder").
- `qualification == "neutra"`, `"error"` o ausente → **excluidos** del entrenamiento.

### Técnica: Unlikelihood loss (opción ligera)

Se eligió unlikelihood en lugar de DPO porque es la opción **más liviana en RAM/CPU/GPU**:

- DPO requiere una **copia congelada del modelo de referencia** (≈2× memoria), instalar `trl` y pares (chosen/rejected) que hoy no existen en el dataset.
- Unlikelihood solo necesita **un forward pass** del modelo LoRA, sin dependencias nuevas ni memoria extra.
- En cada paso, los ejemplos negativos contribuyen con `-log(1 - p)` sobre los tokens de la respuesta, escalado por `LAMBDA_NEGATIVO`. Así el modelo "desaprende" a emitir el contenido dañino.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json/entrenamiento/dataset.json` | Archivo (entrada) | Dataset con `input`/`output`/`qualification`. |
| `models/LLM` | Directorio | Modelo base (preferido) y directorio de salida del modelo fine-tuneado. |
| `models/LLM-base` | Directorio | Modelo base alternativo. |
| `models/LLM-base/model_name.txt` | Archivo (entrada) | Nombre del modelo base, usado por `guardar_registro_modelo`. |
| `models/LLM/model.json` | Archivo (salida) | Registro del último modelo entrenado (versión, fecha y modelo base). |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer`, `AutoModelForCausalLM`, `TrainingArguments`, `Trainer`, `DataCollatorForSeq2Seq`.
- **PEFT**: `LoraConfig`, `get_peft_model`.
- **datasets** (Hugging Face): `Dataset`, `map`, `filter`, `train_test_split`.
- **PyTorch**: tensores, `torch.bfloat16` (GPU) / `torch.float32` (CPU), `torch.logsumexp`, `torch.log1p`.
- **datetime**: fecha del registro del modelo.

## Variables globales

| Variable | Valor inicial | Descripción |
|----------|---------------|-------------|
| `modelo_path` | `""` | Ruta al modelo base; se carga el tokenizador y modelo desde aquí. |
| `dataset_path` | `"json/entrenamiento/dataset.json"` | Ruta del archivo JSON con los datos de entrenamiento. |
| `output_dir` | `"models/LLM/"` | Directorio donde se guarda el modelo fine-tuneado y el registro. |
| `LAMBDA_NEGATIVO` | `0.5` | Peso del término de unlikelihood sobre los ejemplos negativos. |
| `MAX_LENGTH` | `512` | Longitud máxima de tokenización. |
| `PARAMETROS_MINIMOS` | `5_000_000` | Mínimo de parámetros exigido al modelo (5 millones). |

## Compatibilidad con el entorno (venv)

Proyecto probado en `.venv` con `transformers 5.14.1`, `peft 0.20.0`, `torch 2.13.0+cpu` (sin CUDA local) y sin `trl`/`bitsandbytes`. Se corrigieron incompatibilidades con transformers 5.x:

| Antes | Después | Motivo |
|-------|---------|--------|
| `evaluation_strategy="steps"` | `eval_strategy="steps"` | En 5.x el argumento antiguo lanza `TypeError`. |
| `fp16=True` fijo | `fp16=usa_cuda` | En CPU `fp16` falla; en CPU se usa `fp32`. |
| (sin `use_cpu`) | `use_cpu=not usa_cuda` | Sin esto, entrenar en CPU con `bf16`/AMP falla. |
| `report_to=None` | `report_to="none"` | En 5.x `None` lanza `ValueError`. |
| `torch_dtype=` | `dtype=` (y `bf16` en GPU) | `torch_dtype` quedó deprecado. |
| `Trainer(tokenizer=...)` | collator con tokenizer | En 5.x `Trainer` no acepta el kwarg `tokenizer`. |
| `device_map="auto"` | GPU: `device_map="auto"`; CPU: sin device_map | Evita conflictos con `use_cpu`. |
| `prepare_model_for_kbit_training` | eliminado | No hay bitsandbytes ni modelo cuantizado; era un no-op. |
| `os.path.exists("models/LLM")` | `_extraer_items` robusto | Se corrigió el aplanado de dicts de mensajes. |

## Funciones principales

| Función | Rol |
|---------|-----|
| `entrenar_fine()` | Realiza el fine-tuning con LoRA (SFT + unlikelihood) y guarda el modelo y el registro. |
| `guardar_registro_modelo(output_dir, dataset_path, modelo_path)` | Crea o actualiza `model.json` con el registro del último modelo entrenado. |
| `_extraer_items(data)` | Normaliza el dataset a lista de mensajes (soporta lista, dict de dicts y dict de listas). |
| `_calcular_epocas(total, batch)` | Aplica la fórmula de iteraciones por época (ver más abajo). |
| `CollatorConNegativos(DataCollatorForSeq2Seq)` | Añade el tensor binario `is_negative` a cada batch. |
| `TrainerUnlikelihood(Trainer)` | Trainer con loss dual: CE para positivos + unlikelihood para negativos. |

> Nota: la lógica de resolución de ruta (`models/LLM` → `models/LLM-base`) está embebida dentro de `entrenar_fine()`.

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Se aplica a `entrenar_fine` y `guardar_registro_modelo`. Si el modelo o dataset no existen, el decorador imprime el error; `entrenar_fine` sigue lanzando `FileNotFoundError` cuando `dataset_path` no existe.

## Parámetros de `entrenar_fine`

`entrenar_fine()` **no recibe parámetros** (compatible con `main.py` y `main_logs.py`). Usa las variables globales del módulo:

| Variable usada | Descripción |
|----------------|-------------|
| `modelo_path` | Ruta al modelo base. |
| `dataset_path` | Ruta al archivo JSON con los datos. |
| `output_dir` | Directorio donde guardar el modelo. |
| `LAMBDA_NEGATIVO` | Peso del unlikelihood. |
| `MAX_LENGTH` | Longitud máxima de tokenización. |

## Preparación de datos y clasificación por `qualification`

1. Resuelve la ruta del modelo (`models/LLM` → `models/LLM-base`).
2. Verifica que exista `dataset_path`; si no, lanza `FileNotFoundError`.
3. Carga el JSON y lo normaliza con `_extraer_items` (soporta lista, dict de dicts y dict de listas).
4. Para cada mensaje con `input`/`output` de texto válido, clasifica según `qualification`:
   - `positive` → SFT (`is_negative=0`).
   - `negative` → unlikelihood (`is_negative=1`).
   - `neutra` / `error` / sin campo → excluidos.
5. Si un item es una conversación en **formato OpenAI `messages`** (ver `formato_openai.md`), se descompone en pares con `pares_entrenamiento()` antes de clasificar.
6. Si no hay ejemplos SFT ni unlikelihood, imprime el aviso y omite el fine-tuning (`return None`).

## Tokenización

- Texto por ejemplo: `Pregunta: {input}\nRespuesta: {output}`.
- `truncation=True`, `max_length=MAX_LENGTH`, `return_offsets_mapping=True` (transformers 5.x: clave `offset_mapping`).
- Se enmascaran los tokens del prompt (`-100`) y se dejan los de la respuesta como objetivos.
- Se añade la columna `is_negative` (0/1) por ejemplo.
- `Dataset.map(..., batched=True)` + `filter` descarta ejemplos sin respuesta válida.

## Loss dual (SFT + Unlikelihood)

`TrainerUnlikelihood.compute_loss`:

1. Pop de `is_negative` del batch antes del forward (para no pasarlo al modelo).
2. Forward del modelo → `logits`.
3. Shift de logits/labels y máscara de tokens de respuesta (`!= -100`).
4. `token_logp` se calcula con `gather - logsumexp` (evita materializar el softmax completo → ahorra memoria).
5. **Positivos**: loss CE = `-log p` sobre tokens de respuesta.
6. **Negativos**: unlikelihood = `-log(1 - p)` sobre tokens de respuesta (con clamp para estabilidad numérica).
7. Loss total = `loss_pos + LAMBDA_NEGATIVO * loss_neg`.

El collator `CollatorConNegativos` garantiza que `is_negative` llegue al loss aunque `DataCollatorForSeq2Seq` no lo procese.

## Configuración de LoRA

| Parámetro | Valor |
|-----------|-------|
| `r` | 8 |
| `lora_alpha` | 32 |
| `target_modules` | `["q_proj", "v_proj"]` |
| `lora_dropout` | 0.1 |
| `bias` | `"none"` |
| `task_type` | `"CAUSAL_LM"` |

## Configuración de entrenamiento (`TrainingArguments`)

- **Fórmula de épocas**: las épocas se calculan con la fórmula `Iteraciones por época = Tamaño total de datos / Tamaño del lote (Batch Size)`, implementada en `_calcular_epocas(total_ejemplos, batch_efectivo)` (con `ceil` y mínimo 1). El batch efectivo es `per_device_train_batch_size (2) × gradient_accumulation_steps (4) = 8`, así que las épocas resultan `ceil(ejemplos / 8)`.
- Batch por dispositivo de 2, gradiente acumulado de 4.
- Evaluación por steps (cada 50) **solo si hay ≥ 10 ejemplos** (con menos, `eval_strategy="no"`).
- `save_steps=100`, `save_total_limit=2`, `load_best_model_at_end` solo con split de eval.
- `metric_for_best_model="eval_loss"`, `greater_is_better=False`.
- Precisión: `fp16=usa_cuda` (GPU), `bf16=False`; en CPU `use_cpu=True` y modelo en `torch.float32`.
- `learning_rate=2e-4`, `weight_decay=0.01`, `warmup_steps=100`.
- `report_to="none"` (desactiva wandb/tensorboard).
- `remove_unused_columns=False` (imprescindible para que `is_negative` no se elimine del batch).

## Parámetros mínimos del modelo

Tras aplicar LoRA se verifica que el modelo cumpla el mínimo de **5 millones de parámetros** (`PARAMETROS_MINIMOS = 5_000_000`). Si tiene menos, se imprime el aviso y se omite el entrenamiento (`return None`). El modelo base usado (Qwen2 0.5B ≈ 494M de parámetros) lo cumple holgadamente.

## Registro del modelo (`model.json`)

`guardar_registro_modelo` crea `model.json` dentro de `output_dir` (por defecto `models/LLM/model.json`) al terminar el entrenamiento:

- Si el archivo ya existe y su `version` es numérica, la incrementa (`+1`); si no existe o no es válida, empieza en `1`.
- Lee el nombre del modelo base de `models/LLM-base/model_name.txt`; si no existe o está vacío, usa `os.path.basename(modelo_path)`.
- Solo conserva el último registro (sobrescribe, no guarda historial).

### Estructura del registro

```json
{
  "version": 1,
  "fecha": "2026-08-07T00:00:00.000000",
  "modelo_base": "nombre-del-modelo",
  "output_dir": "models/LLM/",
  "dataset_path": "json/entrenamiento/dataset.json"
}
```

## Flujo interno de `entrenar_fine()`

1. Resuelve la ruta del modelo (`models/LLM` → `models/LLM-base`).
2. Verifica `dataset_path`; carga y normaliza el JSON.
3. Clasifica por `qualification` (positive→SFT, negative→unlikelihood, resto excluido) e imprime estadísticas.
4. Carga tokenizador (`local_files_only=True`) y modelo (GPU `bf16` + `device_map="auto"`; CPU `fp32`).
5. Aplica LoRA e imprime parámetros entrenables; verifica el mínimo de 5M de parámetros.
6. Construye `Dataset`, tokeniza con enmascarado del prompt y filtro.
7. Divide 90/10 solo si hay ≥ 10 ejemplos.
8. Calcula las épocas con `_calcular_epocas(len(train_dataset), 8)`.
9. Crea collator y Trainer con loss dual; entrena.
10. Guarda modelo y tokenizador con `save_pretrained(output_dir)`.
11. Llama a `guardar_registro_modelo` y devuelve `output_dir`.

## Notas

- Lanza `FileNotFoundError` si `dataset_path` no existe.
- Si no hay ejemplos con `qualification` `positive`/`negative`, omite el entrenamiento (devuelve `None`).
- Los modelos y tokenizadores se cargan con `local_files_only=True` (el proyecto usa modelos locales; evita accesos al hub).
- Compatible con `main.py` y `main_logs.py`: ambos llaman `entrenar_fine()` sin argumentos y sin cambios.
- `LAMBDA_NEGATIVO=0.0` desactiva el castigo de negativos (equivale al comportamiento anterior).
