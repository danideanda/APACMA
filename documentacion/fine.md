# fine.md

Archivo de documentación técnica para `fine.py`.

## ¿Qué hace el archivo?

`fine.py` implementa el **fine-tuning del modelo LLM** usando la técnica **LoRA** (Low-Rank Adaptation). Toma un modelo base causal, un dataset en formato JSON y entrena el modelo ajustado (fine-tuned), guardándolo en un directorio de salida.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `models/LLM` | Directorio | Modelo base (configuración de carga usada por `verificar_ruta_modelo`). |
| `models/LLM-base` | Directorio | Modelo base alternativo. |
| `models/LLM-base/model_name.txt` | Archivo (entrada) | Nombre del modelo base, usado por `guardar_registro_modelo`. |
| `./dataset.json` (por defecto) | Archivo (entrada) | Dataset con `input`/`output`. |
| `./fine_tuned_model` (por defecto) | Directorio (salida) | Modelo y tokenizador fine-tuneados. |
| `./fine_tuned_model/model.json` | Archivo (salida) | Registro del último modelo entrenado (versión, fecha y modelo base). |

## Tecnologías

- **Python**.
- **Transformers** (Hugging Face): `AutoTokenizer`, `AutoModelForCausalLM`, `TrainingArguments`, `Trainer`, `DataCollatorForSeq2Seq`, `GenerationConfig`.
- **PEFT**: `LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training`.
- **datasets** (Hugging Face): `Dataset` y `train_test_split`.
- **PyTorch**: tensores y `torch.bfloat16`.
- **huggingface_hub**: `login`.
- **datetime**: fecha del registro del modelo.

## Funciones principales

| Función | Rol |
|---------|-----|
| `verificar_ruta_modelo()` | Detecta la ruta del modelo base (`models/LLM` o `models/LLM-base`). |
| `entrenar_fine(modelo_path, dataset_path, output_dir)` | Realiza el fine-tuning con LoRA y guarda el modelo. |
| `guardar_registro_modelo(output_dir, dataset_path, modelo_path)` | Crea o actualiza `model.json` con el registro del último modelo entrenado. |

## Parámetros de `entrenar_fine`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `modelo_path` | — | Ruta al modelo base o nombre en HuggingFace. |
| `dataset_path` | `"./dataset.json"` | Ruta al archivo JSON con los datos. |
| `output_dir` | `"./fine_tuned_model"` | Directorio donde guardar el modelo. |

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

`guardar_registro_modelo` crea `model.json` dentro de `output_dir` al terminar el entrenamiento:

- Si el archivo ya existe y su `version` es numérica, la incrementa (`+1`); si no existe o no es válida, empieza en `1`.
- Lee el nombre del modelo base de `models/LLM-base/model_name.txt` (ruta resuelta desde `fine.py`); si el archivo no existe o está vacío, usa como respaldo `os.path.basename(modelo_path)`.
- Solo conserva el último registro (sobrescribe, no guarda historial).

### Estructura del registro

```json
{
  "version": 1,
  "fecha": "2026-08-07T00:00:00.000000",
  "modelo_base": "nombre-del-modelo",
  "output_dir": "./fine_tuned_model",
  "dataset_path": "./dataset.json"
}
```

### Comportamiento

- Primera ejecución → crea `model.json` con `version: 1`.
- Ejecuciones siguientes → incrementa la `version` y actualiza `fecha` y `modelo_base`.
- La fecha se guarda en formato ISO (`datetime.now().isoformat()`).
- Escribe el JSON con `ensure_ascii=False, indent=4` e imprime una confirmación con la ruta y la versión.