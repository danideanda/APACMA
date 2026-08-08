from seguridad import filtro_seguridad, prueba_seguridad_modelo
from fine import entrenar_fine
from clasificador import juntar, juntar_con_dnapan_completo
from chat.llm import *
import datetime

# ========== variables ==========
# rutas
model_path = ""
dataset_path = "json/entrenamiento/dataset.json"
fecha = datetime.now()

# ========== funciones ==========
def verificar_ruta_modelo():
    global model_path
    # verificar ruta del modelo
    if os.path.exists("models/LLM"):
        model_path = "models/LLM"
    elif os.path.exists("models/LLM-base"):
        model_path = "models/LLM-base"
    else:
        model_path = "error fatal: no se encontró la ruta del modelo"

def main():
    verificar_ruta_modelo()
    if fecha.day == 1: # <--- comŕueba que es otro mes
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
    # en desarrollo


fecha_inicio = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Inicio del programa: {fecha_inicio}")
if __name__ == "__main__":
    main()
    print("Ruta del modelo verificada:", model_path)
    entrenar_fine()