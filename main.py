from seguridad import filtro_seguridad, prueba_seguridad_modelo
from fine import entrenar_fine, ya_entrenado_hoy
from clasificador import juntar, juntar_con_dnapan_completo
from chat.llm import *
import datetime
from functools import wraps

# ========== variables ==========
# rutas
model_path = "models/LLM-base"
dataset_path = "json/entrenamiento/dataset.json"
fecha = datetime.datetime.now()

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
def _directorio_tiene_modelo(ruta):
    """Verifica si un directorio contiene archivos de pesos de un modelo."""
    if not os.path.exists(ruta):
        return False
    for archivo in os.listdir(ruta):
        if archivo.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
            return True
    return False

@manejar_errores
def main():
    while True:
        fecha_actual = datetime.datetime.now()
        if fecha_actual.day == 1: # <--- comŕueba que es otro mes (debe estar en 1)
            if ya_entrenado_hoy():
                print("El modelo ya fue entrenado hoy. No se dispara el proceso programado.")
                continue
            #1. generar dataset
            juntar()
            juntar_con_dnapan_completo()
            # 2. Limpiar los JSON mediante el filtro de seguridad
            print("\n=== FILTRO DE SEGURIDAD DE JSON ===")
            filtro_seguridad()
            # 3. entrena el modelo
            entrenar_fine()
            # 4. Prueba breve de seguridad del modelo entrenado
            print("\n=== PRUEBA BREVE DE SEGURIDAD DEL MODELO ===")
            prueba_seguridad_modelo()
            print("\nProceso completado")


fecha_inicio = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Inicio del programa: {fecha_inicio}")
if __name__ == "__main__":
    try:
        main()
        print("Ruta del modelo verificada:", model_path)
    except ErrorAPACMA as e:
        print(f"[ERROR FATAL] {e}")
    except Exception as e:
        print(f"[ERROR FATAL] {e}")
