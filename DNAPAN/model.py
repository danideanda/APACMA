import json
import math
import os
import time
from functools import wraps

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from formato_openai import leer_conversacion, pares_entrenamiento

# ========== variables globales ==========
RUTA_MODELO_BASE = "models/LLM-base"
RUTA_SALIDA = "models/DNAPAN/model"
RUTA_DATASET = "json/entrenamiento/dataset.json"
RUTA_CONVERSACIONES = "json/conversaciones"

MAX_LENGTH = 512
NUM_CLASES = 2
BATCH_SIZE = 8
PARAMETROS_MINIMOS = 5_000_000
SEMILLA = 42

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


# ========== utilidades ==========

def _directorio_tiene_modelo(ruta):
    """Verifica si un directorio contiene archivos de pesos de un modelo."""
    if not os.path.exists(ruta):
        return False
    for archivo in os.listdir(ruta):
        if archivo.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
            return True
    return False


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


def _texto_desde_item(item):
    """Combina input y output de un item en el texto que analiza DNAPAN."""
    entrada = item.get("input", "")
    salida = item.get("output", "")
    if not entrada or not salida:
        return None
    return f"Pregunta: {entrada}\nRespuesta: {salida}"


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


# ========== carga de datos ==========

@manejar_errores(default=[])
def _cargar_ejemplos():
    """
    Recolecta ejemplos {texto, label} desde el dataset y las conversaciones.

    label: 0 = negative, 1 = positive. Los ejemplos 'neutra' se descartan.
    """
    ejemplos = []

    # 1. Desde el dataset de entrenamiento (input/output/qualification)
    if os.path.exists(RUTA_DATASET):
        with open(RUTA_DATASET, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in _extraer_items(data):
            qual = str(item.get("qualification", "")).strip().lower()
            texto = _texto_desde_item(item)
            if not texto or qual not in ("positive", "negative"):
                continue
            label = 1 if qual == "positive" else 0
            ejemplos.append({"texto": texto, "label": label, "qualification": qual})

    # 2. Desde las conversaciones en formato OpenAI messages
    if os.path.exists(RUTA_CONVERSACIONES):
        for archivo in sorted(os.listdir(RUTA_CONVERSACIONES)):
            if not archivo.endswith(".json"):
                continue
            ruta = os.path.join(RUTA_CONVERSACIONES, archivo)
            conversacion = leer_conversacion(ruta)
            for par in pares_entrenamiento(conversacion):
                qual = str(par.get("qualification") or "").strip().lower()
                if qual not in ("positive", "negative"):
                    continue
                entrada = par.get("input", "")
                salida = par.get("output", "")
                if not entrada or not salida:
                    continue
                texto = f"Pregunta: {entrada}\nRespuesta: {salida}"
                label = 1 if qual == "positive" else 0
                ejemplos.append({"texto": texto, "label": label, "qualification": qual})

    return ejemplos


# ========== entrenamiento ==========

@manejar_errores(default=None)
def entrenar_dnapan():
    """
    Entrena el clasificador DNAPAN (positive/negative) sobre los ejemplos
    recolectados y guarda el modelo en models/DNAPAN/model.

    Returns:
        str: Ruta del modelo guardado (o None si no hay ejemplos suficientes).
    """
    ejemplos = _cargar_ejemplos()
    positivos = sum(1 for e in ejemplos if e["label"] == 1)
    negativos = sum(1 for e in ejemplos if e["label"] == 0)
    print(f"Ejemplos recolectados: {len(ejemplos)} (positive={positivos}, negative={negativos})")

    if len(ejemplos) < 2 or positivos == 0 or negativos == 0:
        print("No hay suficientes ejemplos (se necesitan al menos un positive y un negative). Se omite el entrenamiento.")
        return None

    # Resolver el modelo base (igual criterio que el resto del proyecto)
    if not _directorio_tiene_modelo(RUTA_MODELO_BASE):
        print(f"No se encontró el modelo base en {RUTA_MODELO_BASE}")
        return None

    # Fijar semilla para reproducibilidad (mismo resultado siempre)
    torch.manual_seed(SEMILLA)

    print(f"Cargando tokenizador y modelo base desde {RUTA_MODELO_BASE}...")
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO_BASE, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modelo = AutoModelForSequenceClassification.from_pretrained(
        RUTA_MODELO_BASE,
        num_labels=NUM_CLASES,
        local_files_only=True,
    )

    # Verificar que el modelo cumple el minimo de parametros exigido
    num_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Parametros del modelo: {num_parametros:,}")
    if num_parametros < PARAMETROS_MINIMOS:
        print(f"El modelo tiene menos de {PARAMETROS_MINIMOS} parametros. Se omite el entrenamiento.")
        return None

    # Construir dataset de HuggingFace y tokenizar
    dataset = Dataset.from_list([{"texto": e["texto"], "label": e["label"]} for e in ejemplos])

    def tokenize_function(ejemplos):
        enc = tokenizer(
            ejemplos["texto"],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": ejemplos["label"],
        }

    dataset = dataset.map(
        lambda ejemplos: tokenize_function(ejemplos),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=0.2, seed=SEMILLA)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Entrenamiento: {len(train_dataset)} ejemplos | Validación: {len(eval_dataset)} ejemplos")

    # Calcular epocas con la formula: iteraciones por epoca = total / batch size
    epocas = _calcular_epocas(len(train_dataset), BATCH_SIZE)
    print(f"Epocas calculadas (total {len(train_dataset)} / batch {BATCH_SIZE}): {epocas}")

    # Argumentos de entrenamiento
    args = TrainingArguments(
        output_dir="test/logs_dnapan",
        num_train_epochs=epocas,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="test/logs_dnapan",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=False,
        seed=SEMILLA,
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    inicio = time.time()
    trainer.train()
    print(f"Entrenamiento finalizado en {time.time() - inicio:.1f}s")

    # Guardar el modelo final en la ruta que usa DNAPAN.py
    os.makedirs(RUTA_SALIDA, exist_ok=True)
    trainer.save_model(RUTA_SALIDA)
    tokenizer.save_pretrained(RUTA_SALIDA)
    print(f"Modelo guardado en {RUTA_SALIDA}")

    return RUTA_SALIDA


# ========== ejecucion ==========

if __name__ == "__main__":
    try:
        ruta = entrenar_dnapan()
        print("Modelo DNAPAN entrenado y guardado en:", ruta)
    except ErrorAPACMA as e:
        print(f"[ERROR FATAL] {e}")
    except Exception as e:
        print(f"[ERROR FATAL] {e}")
