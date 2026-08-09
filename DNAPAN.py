import json
import os
from functools import wraps
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from formato_openai import leer_conversacion, obtener_mensajes, pares_entrenamiento

# ========== variables ==========
ruta_modelo = "models/DNAPAN/model"
ruta_conversaciones = "json/conversaciones"
ruta_resultados = "json/entrenamiento/dataset_etiquetado.json"

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

@manejar_errores(default=[])
def DNAPAN_json():
    """
    Funcion principal que procesa los archivos JSON de conversaciones,
    infiere la calificacion (positive/negative) usando el modelo DNAPAN,
    y guarda los resultados.
    
    Retorna:
        list: Lista de diccionarios con los resultados de la inferencia
    """
    # Verificar que el modelo existe
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encuentra el modelo en {ruta_modelo}")
        return []
    
    # Cargar modelo y tokenizador
    print(f"Cargando modelo DNAPAN desde {ruta_modelo}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
        model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
        model.eval()  # Modo evaluacion
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return []
    
    # Obtener todos los archivos JSON
    archivos = [
        f for f in sorted(os.listdir(ruta_conversaciones))
        if f.endswith(".json")
    ]
    
    if not archivos:
        print("No se encontraron archivos JSON en la carpeta de conversaciones")
        return []
    
    print(f"Procesando {len(archivos)} archivos...")
    
    # Lista para almacenar todos los resultados
    todos_resultados = []
    
    # Procesar cada archivo
    for archivo in archivos:
        ruta = os.path.join(ruta_conversaciones, archivo)
        
        conversacion = leer_conversacion(ruta)
        
        print(f"  Procesando {archivo}...")

        # Procesar cada par user->assistant de la conversacion
        for par in pares_entrenamiento(conversacion):
            id_mensaje = par.get("id")
            texto_input = par.get("input")
            texto_output = par.get("output")

            # Combinar input y output para el analisis
            texto_completo = f"Pregunta: {texto_input}\nRespuesta: {texto_output}"

            # Inferir con el modelo
            try:
                # Tokenizar el texto
                inputs = tokenizer(
                    texto_completo,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Realizar la inferencia
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediccion = torch.softmax(logits, dim=1)
                    clase_predicha = torch.argmax(prediccion, dim=1).item()

                # Mapear la prediccion a positive/negative
                # Asumiendo que 0 = negative, 1 = positive
                if clase_predicha == 1:
                    qualification = "positive"
                else:
                    qualification = "negative"

                # Guardar resultado
                resultado = {
                    "id": id_mensaje,
                    "qualification": qualification,
                    "archivo_origen": archivo
                }
                todos_resultados.append(resultado)

                print(f"    ID {id_mensaje}: {qualification}")

            except Exception as e:
                print(f"    Error al procesar ID {id_mensaje}: {e}")
                # Guardar como neutro en caso de error
                todos_resultados.append({
                    "id": id_mensaje,
                    "qualification": "error",
                    "archivo_origen": archivo
                })
    
    # Guardar resultados en un archivo JSON
    if todos_resultados:
        os.makedirs(os.path.dirname(ruta_resultados), exist_ok=True)
        
        with open(ruta_resultados, "w", encoding="utf-8") as fh:
            json.dump(todos_resultados, fh, ensure_ascii=False, indent=4)
        
        # Mostrar estadisticas
        positivos = sum(1 for r in todos_resultados if r["qualification"] == "positive")
        negativos = sum(1 for r in todos_resultados if r["qualification"] == "negative")
        errores = sum(1 for r in todos_resultados if r["qualification"] == "error")
        
        print(f"\n=== RESUMEN ===")
        print(f"Total procesados: {len(todos_resultados)}")
        print(f"  - Positivos: {positivos}")
        print(f"  - Negativos: {negativos}")
        print(f"  - Errores: {errores}")
        print(f"Resultados guardados en: {ruta_resultados}")
    
    return todos_resultados


@manejar_errores(default=None)
def DNAPAN_inferir_texto(texto):
    """
    Funcion auxiliar para inferir un solo texto con el modelo DNAPAN.
    
    Args:
        texto (str): Texto a analizar
    
    Returns:
        str: "positive" o "negative"
    """
    # Verificar que el modelo existe
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encuentra el modelo en {ruta_modelo}")
        return None
    
    # Cargar modelo y tokenizador
    try:
        tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
        model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
        model.eval()
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None
    
    # Tokenizar y predecir
    try:
        inputs = tokenizer(
            texto,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediccion = torch.softmax(logits, dim=1)
            clase_predicha = torch.argmax(prediccion, dim=1).item()
        
        if clase_predicha == 1:
            return "positive"
        else:
            return "negative"
            
    except Exception as e:
        print(f"Error en la inferencia: {e}")
        return None


@manejar_errores(default=[])
def DNAPAN_actualizar_dataset():
    """
    Funcion que actualiza el dataset existente con las calificaciones
    del modelo DNAPAN.
    
    Retorna:
        list: Lista de mensajes actualizados con qualification
    """
    # Primero obtener las calificaciones
    calificaciones = DNAPAN_json()
    
    if not calificaciones:
        print("No se obtuvieron calificaciones")
        return []
    
    # Crear un diccionario para acceso rapido por ID
    dict_calificaciones = {
        item["id"]: item["qualification"] 
        for item in calificaciones
    }
    
    # Procesar cada archivo y actualizar
    archivos = [
        f for f in sorted(os.listdir(ruta_conversaciones))
        if f.endswith(".json")
    ]
    
    mensajes_actualizados = []
    
    for archivo in archivos:
        ruta = os.path.join(ruta_conversaciones, archivo)
        
        conversacion = leer_conversacion(ruta)
        
        for par in pares_entrenamiento(conversacion):
            id_mensaje = par.get("id")
            
            if id_mensaje in dict_calificaciones:
                mensaje_actualizado = dict(par)
                mensaje_actualizado["qualification"] = dict_calificaciones[id_mensaje]
                mensajes_actualizados.append(mensaje_actualizado)
    
    # Guardar dataset actualizado
    if mensajes_actualizados:
        ruta_actualizado = "json/entrenamiento/dataset_actualizado.json"
        os.makedirs(os.path.dirname(ruta_actualizado), exist_ok=True)
        
        with open(ruta_actualizado, "w", encoding="utf-8") as fh:
            json.dump(mensajes_actualizados, fh, ensure_ascii=False, indent=4)
        
        print(f"Dataset actualizado guardado en: {ruta_actualizado}")
    
    return mensajes_actualizados


if __name__ == "__main__":
    # Ejecutar la funcion principal
    print("=== INFERENCIA DNAPAN ===")
    try:
        resultados = DNAPAN_json()

        # Mostrar los primeros resultados como ejemplo
        if resultados:
            print("\n=== PRIMEROS RESULTADOS ===")
            for resultado in resultados[:5]:
                print(json.dumps(resultado, ensure_ascii=False))
    except ErrorAPACMA as e:
        print(f"[ERROR FATAL] {e}")
    except Exception as e:
        print(f"[ERROR FATAL] {e}")