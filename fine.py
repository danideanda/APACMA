import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from datetime import datetime
from functools import wraps
import math

modelo_path = ""
dataset_path = "json/entrenamiento/dataset.json"
output_dir = "models/LLM/"

# Peso del termino de unlikelihood para los negativos.
# 0.0 desactiva el castigo (equivale a no usar negativos).
LAMBDA_NEGATIVO = 0.5

# Longitud maxima de tokenizacion
MAX_LENGTH = 512

# Parametros minimos exigidos al modelo base (5 millones)
PARAMETROS_MINIMOS = 5_000_000

# ========== manejo de errores ==========

class ErrorAPACMA(Exception):
    """Excepción base del proyecto APACMA."""


_SIN_DEFAULT = object()


def manejar_errores(func=None, default=_SIN_DEFAULT):
    """Captura excepciones, imprime mensaje claro y falla con gracia.

    Uso:
        @manejar_errores                      # relanza como ErrorAPACMA
        @manejar_errores(default=[])          # devuelve default en caso de error
    """
    def decorador(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ErrorAPACMA:
                raise
            except Exception as e:
                print(f"[ERROR] {fn.__name__}: {e}")
                if default is not _SIN_DEFAULT:
                    return default
                raise ErrorAPACMA(f"Error en {fn.__name__}: {e}") from e
        return wrapper
    if func is None:
        return decorador
    return decorador(func)


def _directorio_tiene_modelo(ruta):
    """Verifica si un directorio contiene archivos de pesos de un modelo."""
    if not os.path.exists(ruta):
        return False
    for archivo in os.listdir(ruta):
        if archivo.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
            return True
    return False


def _calcular_epocas(total_ejemplos, batch_size):
    """
    Calcula las epocas de entrenamiento con la formula:

        Iteraciones por epoca = Tamaño total de datos / Tamaño del lote

    Se redondea hacia arriba (ceil) para que nunca haya una epoca incompleta
    y se garantiza un minimo de 1 epoca.
    """
    if batch_size <= 0 or total_ejemplos <= 0:
        return 1
    return max(1, math.ceil(total_ejemplos / batch_size))


def _extraer_items(data):
    """
    Normaliza las estructuras posibles del dataset en una lista de mensajes.

    Soporta:
        - Lista de mensajes (dicts).
        - Dict de conversaciones {id: [mensajes]}.
        - Dict de mensajes {id: mensaje}.
    """
    items = []
    if isinstance(data, list):
        items.extend(data)
    elif isinstance(data, dict):
        for valor in data.values():
            if isinstance(valor, list):
                items.extend(valor)
            elif isinstance(valor, dict):
                items.append(valor)
    return items


class CollatorConNegativos(DataCollatorForSeq2Seq):
    """Data collator que anade el tensor binario `is_negative` al batch."""

    def __call__(self, features, return_tensors=None):
        negativos = torch.tensor(
            [int(f["is_negative"]) for f in features], dtype=torch.long
        )
        features = [{k: v for k, v in f.items() if k != "is_negative"} for f in features]
        batch = super().__call__(features, return_tensors)
        batch["is_negative"] = negativos
        return batch


class TrainerUnlikelihood(Trainer):
    """
    Trainer con loss dual:
        - Positivos (SFT): cross-entropy estandar sobre tokens de respuesta.
        - Negativos (unlikelihood): se baja la probabilidad de los tokens de la
          respuesta danina con `-log(1 - p)`, escalado por LAMBDA_NEGATIVO.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_negative = inputs.pop("is_negative")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mascara_respuesta = shift_labels != -100
        objetivo = shift_labels.clamp(min=0)

        # log-probabilidad del token objetivo sin materializar softmax completo
        logsumexp = torch.logsumexp(shift_logits, dim=-1)
        token_logp = (
            shift_logits.gather(-1, objetivo.unsqueeze(-1)).squeeze(-1) - logsumexp
        )

        # loss SFT para positivos: -log p
        ce = -token_logp * mascara_respuesta

        # unlikelihood para negativos: -log(1 - p)
        p = torch.exp(token_logp).clamp(max=1 - 1e-7)
        unlikelihood = -torch.log1p(-p) * mascara_respuesta

        pos_mask = (1 - is_negative).unsqueeze(-1) * mascara_respuesta
        neg_mask = is_negative.unsqueeze(-1) * mascara_respuesta

        loss_pos = (ce * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        loss_neg = (unlikelihood * neg_mask).sum() / neg_mask.sum().clamp(min=1)
        loss = loss_pos + LAMBDA_NEGATIVO * loss_neg

        return (loss, outputs) if return_outputs else loss


@manejar_errores
def entrenar_fine():
    """
    Realiza fine-tuning con LoRA del modelo base usando el campo `qualification`.

    - qualification == "positive": SFT estandar (el modelo aprende la respuesta).
    - qualification == "negative": loss de unlikelihood (el modelo baja la
      probabilidad de la respuesta danina).
    - neutra / error / sin campo: se excluyen del entrenamiento.

    Returns:
        str: Ruta del modelo guardado (o None si no hay datos de entrenamiento).
    """
    global modelo_path
    # verificar ruta del modelo (solo si contiene pesos del modelo)
    if not modelo_path:
        if _directorio_tiene_modelo("models/LLM"):
            modelo_path = "models/LLM"
        elif _directorio_tiene_modelo("models/LLM-base"):
            modelo_path = "models/LLM-base"
        else:
            raise ErrorAPACMA("no se encontró la ruta del modelo")

    # Verificar que existe el archivo de datos
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"El archivo {dataset_path} no existe")

    # Cargar el dataset
    print(f"Cargando datos desde {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalizar estructura y clasificar por qualification
    registros = []

    def _registro_desde_input_output(entrada, salida, qualification):
        """Crea un registro de entrenamiento si input/output son validos y la
        qualification es positive/negative. Devuelve None en otro caso."""
        if not isinstance(entrada, str) or not isinstance(salida, str):
            return None
        if not entrada.strip() or not salida.strip():
            return None
        qual = str(qualification or "").strip().lower()
        if qual == "positive":
            tipo = "sft"
            is_negative = 0
        elif qual == "negative":
            tipo = "unlikelihood"
            is_negative = 1
        else:
            return None  # neutra / error / sin campo: excluidos
        return {
            "input": entrada,
            "output": salida,
            "qualification": qual,
            "tipo": tipo,
            "is_negative": is_negative,
        }

    for item in _extraer_items(data):
        if not isinstance(item, dict):
            continue

        # Conversacion en formato OpenAI messages -> pares user/assistant
        if "messages" in item:
            from formato_openai import pares_entrenamiento
            for par in pares_entrenamiento(item):
                registro = _registro_desde_input_output(
                    par.get("input", ""), par.get("output", ""), par.get("qualification")
                )
                if registro:
                    registros.append(registro)
            continue

        registro = _registro_desde_input_output(
            item.get("input"), item.get("output"), item.get("qualification")
        )
        if registro:
            registros.append(registro)

    if not registros:
        print("No hay datos de entrenamiento con qualification 'positive' o 'negative' en el dataset. Se omite el fine-tuning.")
        return None

    positivos = sum(1 for r in registros if r["tipo"] == "sft")
    negativos = sum(1 for r in registros if r["tipo"] == "unlikelihood")
    print(f"Ejemplos SFT (positivos): {positivos}")
    print(f"Ejemplos unlikelihood (negativos): {negativos}")

    # Cargar tokenizador y modelo
    print(f"Cargando modelo desde {modelo_path}...")
    tokenizer = AutoTokenizer.from_pretrained(modelo_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    usa_cuda = torch.cuda.is_available()
    if usa_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            modelo_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            modelo_path,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )

    # Configuración de LoRA
    lora_config = LoraConfig(
        r=8,  # Dimensión de la matriz de adaptación
        lora_alpha=32,  # Factor de escala
        target_modules=["q_proj", "v_proj"],  # Módulos a adaptar
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Aplicar LoRA al modelo
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Verificar que el modelo cumple el minimo de parametros exigido
    num_parametros = sum(p.numel() for p in model.parameters())
    print(f"Parametros del modelo: {num_parametros:,}")
    if num_parametros < PARAMETROS_MINIMOS:
        print(f"El modelo tiene menos de {PARAMETROS_MINIMOS} parametros. Se omite el entrenamiento.")
        return None

    # Construir dataset de HuggingFace
    dataset = Dataset.from_list(registros)

    # Función de tokenización con enmascarado del prompt
    def tokenize_function(ejemplos):
        inputs_ids = []
        atenciones = []
        labels_ids = []
        negativos_flag = []
        for entrada, salida, is_neg in zip(
            ejemplos["input"], ejemplos["output"], ejemplos["is_negative"]
        ):
            prompt = f"Pregunta: {entrada}\nRespuesta: "
            texto = prompt + salida
            enc = tokenizer(
                texto,
                truncation=True,
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"]
            # primer token que pertenece a la respuesta
            resp_start = len(input_ids)
            for idx, (inicio, _fin) in enumerate(enc["offset_mapping"]):
                if inicio >= len(prompt):
                    resp_start = idx
                    break
            if resp_start >= len(input_ids):
                continue
            labels = [-100] * resp_start + input_ids[resp_start:]
            inputs_ids.append(input_ids)
            atenciones.append(enc["attention_mask"])
            labels_ids.append(labels)
            negativos_flag.append(is_neg)
        return {
            "input_ids": inputs_ids,
            "attention_mask": atenciones,
            "labels": labels_ids,
            "is_negative": negativos_flag,
        }

    dataset = dataset.map(
        lambda ejemplos: tokenize_function(ejemplos),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(lambda ej: len(ej["input_ids"]) > 0)

    if len(dataset) == 0:
        print("Tras tokenizar no quedan ejemplos validos. Se omite el fine-tuning.")
        return None

    # Dividir en train y eval (90% train, 10% eval) si hay datos suficientes
    train_dataset = dataset
    eval_dataset = None
    if len(dataset) >= 10:
        dividido = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dividido["train"]
        eval_dataset = dividido["test"]
        print(
            f"Dataset cargado: {len(train_dataset)} ejemplos de entrenamiento, "
            f"{len(eval_dataset)} de evaluación"
        )
    else:
        print(f"Dataset cargado: {len(train_dataset)} ejemplos de entrenamiento (sin split de eval)")

    # Data collator
    data_collator = CollatorConNegativos(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Calcular epocas con la formula: iteraciones por epoca = total / batch size
    # El batch efectivo es batch por dispositivo x acumulacion de gradientes
    batch_efectivo = 2 * 4
    epocas = _calcular_epocas(len(train_dataset), batch_efectivo)
    print(f"Epocas calculadas (total {len(train_dataset)} / batch {batch_efectivo}): {epocas}")

    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epocas,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=usa_cuda,
        bf16=False,
        use_cpu=not usa_cuda,
        learning_rate=2e-4,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
    )

    # Crear trainer
    trainer = TrainerUnlikelihood(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Entrenar
    print("Iniciando entrenamiento...")
    trainer.train()

    # Guardar el modelo fine-tuneado
    print(f"Guardando modelo en {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    guardar_registro_modelo(output_dir, dataset_path, modelo_path)

    print("¡Fine-tuning completado!")
    print("eliminando checkpoint...")
    os.system("rm -r models/LLM/chekpoint")
    print("eliminado")
    return output_dir


@manejar_errores
def guardar_registro_modelo(output_dir, dataset_path, modelo_path):
    """
    Guarda o actualiza el archivo model.json con el registro del último
    modelo entrenado: versión (incremental), fecha y nombre del modelo base.

    Args:
        output_dir (str): Directorio del modelo fine-tuneado.
        dataset_path (str): Ruta del dataset usado en el entrenamiento.
        modelo_path (str): Ruta al modelo base.
    """
    modelo_json_path = os.path.join(output_dir, "model.json")

    version = 1
    if os.path.exists(modelo_json_path):
        try:
            with open(modelo_json_path, "r", encoding="utf-8") as f:
                registro_previo = json.load(f)
            if isinstance(registro_previo.get("version"), int):
                version = registro_previo["version"] + 1
        except (json.JSONDecodeError, OSError):
            version = 1

    nombre_modelo_base = ""
    ruta_nombre = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "LLM-base", "model_name.txt"
    )
    if os.path.exists(ruta_nombre):
        with open(ruta_nombre, "r", encoding="utf-8") as f:
            nombre_modelo_base = f.read().strip()
    if not nombre_modelo_base:
        nombre_modelo_base = os.path.basename(modelo_path)

    registro = {
        "version": version,
        "fecha": datetime.now().isoformat(),
        "modelo_base": nombre_modelo_base,
        "output_dir": output_dir,
        "dataset_path": dataset_path
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(modelo_json_path, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=4)

    print(f"Registro guardado en {modelo_json_path} (versión {version})")


@manejar_errores(default=False)
def ya_entrenado_hoy(modelo_json_path=None):
    """Comprueba si el modelo ya fue entrenado hoy.

    Lee el model.json generado por guardar_registro_modelo tras el entrenamiento.
    Si la fecha registrada corresponde al dia actual devuelve True; en cualquier
    otro caso (archivo inexistente, sin fecha o fecha de otro dia) devuelve False.

    Args:
        modelo_json_path (str, opcional): Ruta al model.json. Si es None se usa
            la ruta por defecto del modelo entrenado (models/LLM/model.json).

    Returns:
        bool: True si el modelo ya fue entrenado hoy, False en caso contrario.
    """
    if modelo_json_path is None:
        modelo_json_path = os.path.join(output_dir, "model.json")
    if not os.path.exists(modelo_json_path):
        return False

    with open(modelo_json_path, "r", encoding="utf-8") as f:
        registro = json.load(f)

    fecha_entrenamiento = registro.get("fecha")
    if not fecha_entrenamiento:
        return False

    fecha_registro = datetime.fromisoformat(fecha_entrenamiento)
    return fecha_registro.date() == datetime.now().date()
