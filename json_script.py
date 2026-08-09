import json
import os
from functools import wraps
from formato_openai import leer_conversacion, obtener_mensajes

# ========== variables ==========
ruta_conversaciones = "json/conversaciones"

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
def id_json():
    """
    Extrae los IDs de mensajes con qualification 'positive' o 'negative'
    de todos los archivos JSON en la carpeta de conversaciones.
    Retorna una lista de diccionarios con el ID y el nombre del archivo.
    """
    # Obtener todos los archivos JSON
    archivos = [
        f for f in sorted(os.listdir(ruta_conversaciones))
        if f.endswith(".json")
    ]
    
    if not archivos:
        print("No se encontraron archivos JSON en la carpeta de conversaciones")
        return []
    
    print(f"Buscando mensajes con qualification 'positive' o 'negative' en {len(archivos)} archivos...")
    
    # Lista para almacenar los IDs y archivos encontrados
    ids_encontrados = []
    
    # Procesar cada archivo
    for archivo in archivos:
        ruta = os.path.join(ruta_conversaciones, archivo)

        conversacion = leer_conversacion(ruta)

        # Buscar mensajes assistant con qualification positive o negative
        for mensaje in obtener_mensajes(conversacion):
            qualification = mensaje.get("qualification", "").lower()
            id_mensaje = mensaje.get("id")

            # Solo extraer si es positive o negative y tiene ID
            if qualification in ["positive", "negative"] and id_mensaje is not None:
                ids_encontrados.append({
                    "id": id_mensaje,
                    "archivo": archivo,
                    "qualification": qualification
                })
    
    # Mostrar resumen
    print(f"Se encontraron {len(ids_encontrados)} mensajes con qualification 'positive' o 'negative'")
    
    # Contar cuántos son positivos y negativos
    positivos = sum(1 for item in ids_encontrados if item["qualification"] == "positive")
    negativos = sum(1 for item in ids_encontrados if item["qualification"] == "negative")
    print(f"  - Positivos: {positivos}")
    print(f"  - Negativos: {negativos}")
    
    # Retornar lista con ID y archivo
    return [
        {
            "id": item["id"],
            "archivo": item["archivo"],
            "qualification": item["qualification"]
        }
        for item in ids_encontrados
    ]