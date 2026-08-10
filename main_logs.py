from seguridad import filtro_seguridad, prueba_seguridad_modelo
from fine import entrenar_fine
from clasificador import juntar, juntar_con_dnapan_completo
from chat.llm import *
import datetime
import logging
import os
import time
from functools import wraps

# ========== variables ==========
# rutas
model_path = ""
dataset_path = "json/entrenamiento/dataset.json"
fecha = datetime.datetime.now()

# ========== configuración de logs ==========

DIRECTORIO_LOGS = "test/logs"
os.makedirs(DIRECTORIO_LOGS, exist_ok=True)
RUTA_LOG = os.path.join(DIRECTORIO_LOGS, f"main_{fecha.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(RUTA_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("APACMA")

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
                log.error(f"[ERROR] {fn.__name__}: {e}")
                if default is not _SIN_DEFAULT:
                    return default
                raise ErrorAPACMA(f"Error en {fn.__name__}: {e}") from e
        return wrapper
    if func is None:
        return decorador
    return decorador(func)


# ========== funciones ==========
def _directorio_tiene_modelo(ruta):
    """Verifica si un directorio contiene archivos de pesos de un modelo."""
    if not os.path.exists(ruta):
        return False
    for archivo in os.listdir(ruta):
        if archivo.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
            return True
    return False


@manejar_errores
def verificar_ruta_modelo():
    global model_path
    # verificar ruta del modelo (solo si contiene pesos del modelo)
    if _directorio_tiene_modelo("models/LLM"):
        model_path = "models/LLM"
        log.info(f"Ruta del modelo encontrada: {model_path}")
    elif _directorio_tiene_modelo("models/LLM-base"):
        model_path = "models/LLM-base"
        log.info(f"Ruta del modelo encontrada: {model_path}")
    else:
        model_path = "error fatal: no se encontró la ruta del modelo"
        log.error(model_path)

@manejar_errores
def main():
    verificar_ruta_modelo()
    while True:
        fecha_actual = datetime.datetime.now()
        if fecha_actual.hour == 11: # <--- comŕueba que es otro mes (debe estar en 1 pero estamos de pruebas asi que estara en en hour y con otro numero)
            #1. generar dataset
            log.info("=== PASO 1: GENERAR DATASET ===")
            juntar()
            juntar_con_dnapan_completo()
            # 2. Limpiar los JSON mediante el filtro de seguridad
            log.info("=== PASO 2: FILTRO DE SEGURIDAD DE JSON ===")
            filtro_seguridad()
            # 3. entrena el modelo
            log.info("=== PASO 3: ENTRENAMIENTO (FINE-TUNING) ===")
            entrenar_fine()
            # 4. Prueba breve de seguridad del modelo entrenado
            log.info("=== PASO 4: PRUEBA BREVE DE SEGURIDAD DEL MODELO ===")
            prueba_seguridad_modelo()
            log.info("Proceso completado")
            break
        else:
            log.info("Hoy no corresponde ejecutar el pipeline (condición de día no cumplida).")
            time.sleep(60)
    # en pruebas


fecha_inicio = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Inicio del programa: {fecha_inicio}")
log.info(f"Logs guardados en: {RUTA_LOG}")
if __name__ == "__main__":
    try:
        main()
        log.info(f"Ruta del modelo verificada: {model_path}")
    except ErrorAPACMA as e:
        log.error(f"[ERROR FATAL] {e}")
    except Exception as e:
        log.error(f"[ERROR FATAL] {e}")
