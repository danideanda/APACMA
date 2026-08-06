from transformers import AutoTokenizer
import json

# ========== Variables ==========

# Cargar el tokenizador del modelo base
tokenizer = AutoTokenizer.from_pretrained("models/LLM-base", use_fast=True)

# Límite máximo de tokens permitido
limite_tokens = 3_000_000

# Contador global de tokens
tokens_actuales = 0

# Cargar el dataset
with open("json/entrenamiento/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


def contar_tokens():
    """
    Cuenta la cantidad total de tokens del dataset.
    """
    global tokens_actuales

    # Reiniciar el contador
    tokens_actuales = 0

    for mensaje in dataset:
        tokens = tokenizer(mensaje["text"], return_tensors="pt")
        tokens_actuales += tokens["input_ids"].size(1)

    print(f"Total de tokens: {tokens_actuales}")


def eliminar_mensajes():
    """
    Elimina mensajes con más de 500 tokens hasta que el dataset
    quede por debajo del límite establecido.
    """
    global dataset

    # Recorrer una copia para poder eliminar elementos sin problemas
    for mensaje in dataset[:]:

        tokens = tokenizer(mensaje["text"], return_tensors="pt")
        cantidad_tokens = tokens["input_ids"].size(1)

        if cantidad_tokens > 500:

            print(f"Eliminando mensaje ({cantidad_tokens} tokens)")
            print(mensaje["text"])
            print("-" * 60)

            dataset.remove(mensaje)

            # Recalcular el total
            contar_tokens()

            if tokens_actuales <= limite_tokens:
                break

    # Guardar el dataset actualizado
    with open("json/entrenamiento/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print("Dataset actualizado correctamente.")


def limites_tokens():
    """
    Verifica si el dataset supera el límite de tokens.
    """
    if tokens_actuales > limite_tokens:
        print("Se excedió el límite de tokens.")
        eliminar_mensajes()
    else:
        print("El dataset está dentro del límite.")


if __name__ == "__main__":

    # Contar tokens
    contar_tokens()

    # Verificar el límite
    limites_tokens()