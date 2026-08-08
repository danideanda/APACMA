from transformers import AutoTokenizer
import json
from functools import wraps

# ========== variables ==========

# Cargar el tokenizador del modelo base
try:
    tokenizer = AutoTokenizer.from_pretrained("models/LLM-base", use_fast=True)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el tokenizador: {e}")
    tokenizer = None

# Límite máximo de tokens permitido
limite_tokens = 3_000_000

# Contador global de tokens
tokens_actuales = 0

# Cargar el dataset
try:
    with open("json/entrenamiento/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el dataset: {e}")
    dataset = []

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


# ========== funciones ==========

@manejar_errores
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


@manejar_errores
def eliminar_mensajes():
    """
    Elimina mensajes con más segun la sigientes jerarquias:
    0.9 se dejaran solo los mensajes que se repiten mas de 3 veces
    0.6 se dejan los que son menores a 500 tokens
    0.3 se dejan los que son negativos

    esta jerarqia ayuda a optimizar los mensajes de manera mas sensilla ademas
    que al usar un algoritmo de optimizacion en vez de un modelo de ai se
    optimiza el proceso.
    """
    global dataset

    # Recorrer una copia para poder eliminar elementos mayores a 500 tokens sin afectar la iteración
    for mensaje in dataset[:]:

        tokens = tokenizer(mensaje["text"], return_tensors="pt")
        cantidad_tokens = tokens["input_ids"].size(1)
        mensajes_eliminados = 0

        if cantidad_tokens > 500 and mensaje["qualification"] == "positive": # hace que si en el valor qualification es positive y es superior a 500 tokens se elimine el mensaje pero si es negative no se elimine

            print(f"Eliminando mensaje ({cantidad_tokens} tokens)")
            print(mensaje["text"])
            print("-" * 60)

            dataset.remove(mensaje)
            mensajes_eliminados += 1
            # Recalcular el total
            contar_tokens()

            if tokens_actuales <= limite_tokens:
                break
            elif mensajes_eliminados >= 1_000:  # Limitar a 10 eliminaciones por ejecución
                break
    
    # buscar mensajes que se repiten menos de 3 veses y eliminarlos
    mensaje_contador = 0
    for mensaje in dataset:
        if dataset.count(mensaje) < 3 and mensaje["qualification"] == "positive": # hace que si en el valor qualification es positive y se repite menos de 3 veces se elimine el mensaje pero si es negative no se elimine
            mensaje_contador += 1
            print(f"Eliminando mensaje que se repite menos de 3 veces ({mensaje_contador})")
            print(mensaje["text"])
            print("-" * 60)
            dataset.remove(mensaje)

            # Recalcular el total
            contar_tokens()

            if tokens_actuales <= limite_tokens:
                break
            elif mensaje_contador >= 1_000:  # Limitar a 10 eliminaciones por ejecución
                break



    # Guardar el dataset actualizado
    with open("json/entrenamiento/dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)



    

@manejar_errores
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
    try:
        # Contar tokens
        contar_tokens()

        # Verificar el límite
        limites_tokens()
    except ErrorAPACMA as e:
        print(f"[ERROR FATAL] {e}")
    except Exception as e:
        print(f"[ERROR FATAL] {e}")
