# fine.md

Archivo de documentación técnica para `fine.py`.

## ¿Qué hace el archivo?

`fine.py` implementa el **fine-tuning del modelo LLM** usando la técnica **LoRA** (Low-Rank Adaptation). Toma un modelo base causal, un dataset en formato JSON y entrena el modelo ajustado (fine-tuned), guardándolo en un directorio de salida. Además genera un registro (`model.json`) con la versión y fecha del último modelo entrenado.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json/entrenamiento/dataset.json` | Archivo (entrada) | Dataset con `input`/`output`. |
| `models/LLM` | Directorio | Modelo base (preferido) y directorio de salida del modelo fine-tuneado. |
| `models/LLM-base` | Directorio | Modelo base alternativo. |
| `models/LLM-base/model_name.txt` | Archivo (entrada) | Nombre del modelo base, usado por `guardar_registro_modelo`. |
| `models/LLM/model.json` | Archivo (salida) | Registro del último modelo entrenado (versión, fecha y modelo base). |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer`, `AutoModelForCausalLM`, `TrainingArguments`, `Trainer`, `DataCollatorForSeq2Seq`.
- **PEFT**: `LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training`.
- **datasets** (Hugging Face): `Dataset` y `train_test_split`.
- **PyTorch**: tensores y `torch.bfloat16`.
- **huggingface_hub**: `login`.
- **datetime**: fecha del registro del modelo.

## Variables globales

| Variable | Valor inicial | Descripción |
|----------|---------------|-------------|
| `modelo_path` | `""` | Ruta al modelo base; se carga el tokenizador y modelo desde aquí. |
| `dataset_path` | `"json/entrenamiento/dataset.json"` | Ruta del archivo JSON con los datos de entrenamiento. |
| `output_dir` | `"models/LLM/"` | Directorio donde se guarda el modelo fine-tuneado y el registro. |

## Funciones principales

| Función | Rol |
|---------|-----|
| `entrenar_fine()` | Realiza el fine-tuning con LoRA y guarda el modelo y el registro. |
| `guardar_registro_modelo(output_dir, dataset_path, modelo_path)` | Crea o actualiza `model.json` con el registro del último modelo entrenado. |

> Nota: ya no existe la función `verificar_ruta_modelo()`. La lógica de resolución de ruta (`models/LLM` → `models/LLM-base`) está embebida dentro de `entrenar_fine()`.

## Parámetros de `entrenar_fine`

`entrenar_fine()` **no recibe parámetros**. Usa las variables globales del módulo:

| Variable usada | Descripción |
|----------------|-------------|
| `modelo_path` | Ruta al modelo base. |
| `dataset_path` | Ruta al archivo JSON con los datos. |
| `output_dir` | Directorio donde guardar el modelo. |

## Flujo interno de `entrenar_fine()`

1. Resuelve la ruta del modelo (lógica local con `global model_path`): si existe `models/LLM` usa esa ruta; si no y existe `models/LLM-base`, usa esa; si ninguno, asigna `"error fatal: no se encontró la ruta del modelo"`.
2. Verifica que exista `dataset_path`; si no, lanza `FileNotFoundError`.
3. Carga el JSON y, si es un diccionario, lo envuelve en una lista.
4. Construye el texto `Pregunta: {input}\nRespuesta: {output}` para cada item.
5. Crea el dataset de HuggingFace y lo divide en train/eval (90/10, `seed=42`).
6. Carga tokenizador y modelo desde `modelo_path` (`torch.bfloat16`, `device_map="auto"`, `trust_remote_code=True`).
7. Si el tokenizador no tiene `pad_token`, usa `eos_token`.
8. Prepara el modelo para k-bit y aplica la configuración LoRA.
9. Tokeniza los datasets y crea el `DataCollatorForSeq2Seq`.
10. Ejecuta el entrenamiento con `Trainer`.
11. Guarda modelo y tokenizador en `output_dir` con `save_pretrained`.
12. Llama a `guardar_registro_modelo(output_dir, dataset_path, modelo_path)` y devuelve `output_dir`.

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

- 3 épocas, batch de 2, gradiente acumulado de 4.
- Evaluación por steps (cada 50), guardado cada 100 steps, `save_total_limit=2`, `load_best_model_at_end=True`.
- `metric_for_best_model="eval_loss"`, `greater_is_better=False`.
- `fp16=True`, `learning_rate=2e-4`, `weight_decay=0.01`, `warmup_steps=100`.
- `report_to=None` (desactiva wandb/tensorboard).

## Preparación de datos

- Lee el JSON, si es un diccionario lo envuelve en una lista.
- Convierte cada item en el texto: `Pregunta: {input}\nRespuesta: {output}`.
- Divide en entrenamiento/validación (90/10) con `seed=42`.

## Tokenización

- `truncation=True`, `padding=True`, `max_length=512`.
- Si el tokenizador no tiene `pad_token`, se usa `eos_token`.

## Notas

- Lanza `FileNotFoundError` si `dataset_path` no existe.
- Después del entrenamiento guarda el modelo y tokenizador con `save_pretrained(output_dir)`.
- Al final del entrenamiento llama a `guardar_registro_modelo` para generar/actualizar el registro.

## Registro del modelo (`model.json`)

`guardar_registro_modelo` crea `model.json` dentro de `output_dir` (por defecto `models/LLM/model.json`) al terminar el entrenamiento:

- Si el archivo ya existe y su `version` es numérica, la incrementa (`+1`); si no existe o no es válida, empieza en `1`.
- Lee el nombre del modelo base de `models/LLM-base/model_name.txt` (ruta resuelta desde `fine.py`); si el archivo no existe o está vacío, usa como respaldo `os.path.basename(modelo_path)`.
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

### Comportamiento

- Primera ejecución → crea `model.json` con `version: 1`.
- Ejecuciones siguientes → incrementa la `version` y actualiza `fecha` y `modelo_base`.
- La fecha se guarda en formato ISO (`datetime.now().isoformat()`).
- Escribe el JSON con `ensure_ascii=False, indent=4` e imprime una confirmación con la ruta y la versión.
