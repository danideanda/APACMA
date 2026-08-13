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
model_path = "models/LLM-base"
model_fine_path = "models/LLM-base"
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
        modelo_json_path = os.path.join(model_fine_path, "model.json")
    if not os.path.exists(modelo_json_path):
        return False

    with open(modelo_json_path, "r", encoding="utf-8") as f:
        registro = json.load(f)

    fecha_entrenamiento = registro.get("fecha")
    if not fecha_entrenamiento:
        return False

    fecha_registro = datetime.fromisoformat(fecha_entrenamiento)
    return fecha_registro.date() == datetime.now().date()

@manejar_errores
def main():
    while True:
        fecha_actual = datetime.datetime.now()
        if fecha_actual.hour == 15: # <--- comŕueba que es otro mes (debe estar en 1 pero estamos de pruebas asi que estara en en hour y con otro numero)
            if ya_entrenado_hoy():
                log.info("El modelo ya fue entrenado hoy. No se dispara el proceso programado.")
            else:
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
