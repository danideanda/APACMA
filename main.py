from fine import entrenar_fine
from DNAPAN import *
from seguridad import *
from clasificador import *
from chat.llm import *

# ========== variables ==========
# ===== paths =====
model_path = ""
dataset_path = "json/entrenamiento/dataset.json"

def verificar_ruta_modelo():
    global model_path
    # ========== verificar ruta del modelo ==========
    if os.path.exists("models/LLM"):
        model_path = "models/LLM"
    elif os.path.exists("models/LLM-base"):
        model_path = "models/LLM-base"
    else:
        model_path = "error fatal: no se encontró la ruta del modelo"

def main():
    verificar_ruta_modelo()
    return
    # no hay nada jeje
    # en desarrollo


if __name__ == "__main__":
    main()
    print("Ruta del modelo verificada:", model_path)
    entrenar_fine(modelo_path=model_path)