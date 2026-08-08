import os
import json
from functools import wraps
from json_script import id_json
from DNAPAN import DNAPAN_inferir_texto, DNAPAN_json

# ========== variables ==========
ruta_conversaciones = "json/conversaciones"
ruta_dataset_filtrado = "json/entrenamiento/dataset.json"

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
def juntar():
    """
    Toma los IDs y archivos obtenidos por id_json(), extrae los datos completos
    de esos mensajes, los procesa con DNAPAN y guarda el resultado.
    """
    # Obtener los IDs con sus archivos correspondientes
    ids_info = id_json()
    
    if not ids_info:
        print("No hay IDs para procesar")
        return []
    
    print(f"Procesando {len(ids_info)} IDs con DNAPAN...")
    
    # Lista para almacenar los mensajes completos
    mensajes_completos = []
    
    # Agrupar IDs por archivo para optimizar la lectura
    archivos_ids = {}
    for item in ids_info:
        archivo = item["archivo"]
        id_mensaje = item["id"]
        
        if archivo not in archivos_ids:
            archivos_ids[archivo] = []
        archivos_ids[archivo].append(id_mensaje)
    
    # Procesar cada archivo
    for archivo, ids_buscar in archivos_ids.items():
        ruta = os.path.join(ruta_conversaciones, archivo)
        
        if not os.path.exists(ruta):
            print(f"Advertencia: No se encuentra el archivo {archivo}")
            continue
        
        with open(ruta, "r", encoding="utf-8") as fh:
            conversacion = json.load(fh)
        
        # Buscar mensajes con los IDs especificados
        for mensaje in conversacion:
            id_mensaje = mensaje.get("id")
            
            if id_mensaje in ids_buscar:
                # Preparar el texto para DNAPAN (combinar input y output)
                texto_input = mensaje.get("input", "")
                texto_output = mensaje.get("output", "")
                texto_completo = f"Pregunta: {texto_input}\nRespuesta: {texto_output}"
                
                # Inferir con DNAPAN
                print(f"  - Analizando ID {id_mensaje} con DNAPAN...")
                qualification = DNAPAN_inferir_texto(texto_completo)
                
                # Si DNAPAN falla, usar la qualification existente
                if qualification is None:
                    qualification = mensaje.get("qualification", "neutra")
                    print(f"    DNAPAN fallo, usando qualification original: {qualification}")
                else:
                    print(f"    DNAPAN resultado: {qualification}")
                
                # Extraer los campos necesarios
                mensaje_completo = {
                    "id": mensaje.get("id"),
                    "date": mensaje.get("date"),
                    "input": mensaje.get("input"),
                    "output": mensaje.get("output"),
                    "qualification": qualification,  # Usar la de DNAPAN
                    "archivo_origen": archivo
                }
                mensajes_completos.append(mensaje_completo)
                print(f"  - ID {id_mensaje} extraido de {archivo} con qualification: {qualification}")
    
    # Ordenar por ID
    mensajes_completos.sort(key=lambda x: x["id"])
    
    # Guardar en el archivo dataset_filtrado.json
    if mensajes_completos:
        os.makedirs(os.path.dirname(ruta_dataset_filtrado), exist_ok=True)
        
        with open(ruta_dataset_filtrado, "w", encoding="utf-8") as f:
            json.dump(mensajes_completos, f, ensure_ascii=False, indent=4)
        
        print(f"\nSe guardaron {len(mensajes_completos)} mensajes en {ruta_dataset_filtrado}")
        
        # Mostrar estadisticas
        positivos = sum(1 for m in mensajes_completos if m["qualification"] == "positive")
        negativos = sum(1 for m in mensajes_completos if m["qualification"] == "negative")
        neutros = sum(1 for m in mensajes_completos if m["qualification"] == "neutra")
        errores = sum(1 for m in mensajes_completos if m["qualification"] == "error")
        
        print(f"  - Positivos: {positivos}")
        print(f"  - Negativos: {negativos}")
        print(f"  - Neutros: {neutros}")
        print(f"  - Errores: {errores}")
    else:
        print("No se encontraron mensajes para guardar")
    
    return mensajes_completos


@manejar_errores(default=[])
def juntar_con_dnapan_completo():
    """
    Version que usa DNAPAN_json() para procesar todos los mensajes
    y luego los filtra con id_json().
    """
    print("=== PROCESANDO CON DNAPAN COMPLETO ===")
    
    # Primero obtener todas las calificaciones con DNAPAN
    resultados_dnapan = DNAPAN_json()
    
    if not resultados_dnapan:
        print("No se obtuvieron resultados de DNAPAN")
        return []
    
    print(f"DNAPAN proceso {len(resultados_dnapan)} mensajes")
    
    # Obtener los IDs que nos interesan (positive y negative)
    ids_info = id_json()
    ids_interesantes = {item["id"] for item in ids_info}
    
    print(f"IDs interesantes (positive/negative): {len(ids_interesantes)}")
    
    # Filtrar los resultados de DNAPAN
    mensajes_filtrados = []
    
    for resultado in resultados_dnapan:
        id_mensaje = resultado.get("id")
        if id_mensaje in ids_interesantes:
            # Buscar el mensaje completo en los archivos
            archivo = resultado.get("archivo_origen")
            if archivo:
                ruta = os.path.join(ruta_conversaciones, archivo)
                if os.path.exists(ruta):
                    with open(ruta, "r", encoding="utf-8") as fh:
                        conversacion = json.load(fh)
                    
                    for mensaje in conversacion:
                        if mensaje.get("id") == id_mensaje:
                            mensaje_completo = {
                                "id": mensaje.get("id"),
                                "date": mensaje.get("date"),
                                "input": mensaje.get("input"),
                                "output": mensaje.get("output"),
                                "qualification": resultado.get("qualification"),
                                "archivo_origen": archivo
                            }
                            mensajes_filtrados.append(mensaje_completo)
                            break
    
    # Guardar resultados
    if mensajes_filtrados:
        ruta_dnapan_filtrado = "json/entrenamiento/dataset_dnapan_filtrado.json"
        os.makedirs(os.path.dirname(ruta_dnapan_filtrado), exist_ok=True)
        
        with open(ruta_dnapan_filtrado, "w", encoding="utf-8") as f:
            json.dump(mensajes_filtrados, f, ensure_ascii=False, indent=4)
        
        print(f"Se guardaron {len(mensajes_filtrados)} mensajes en {ruta_dnapan_filtrado}")
    
    return mensajes_filtrados


@manejar_errores(default=[])
def extraer_por_ids(lista_ids):
    """
    Funcion alternativa que extrae mensajes basados en una lista de IDs proporcionada.
    
    Args:
        lista_ids (list): Lista de IDs a extraer
    
    Returns:
        list: Lista de mensajes completos
    """
    if not lista_ids:
        print("No se proporcionaron IDs")
        return []
    
    # Obtener todos los archivos JSON
    archivos = [
        f for f in sorted(os.listdir(ruta_conversaciones))
        if f.endswith(".json")
    ]
    
    mensajes_encontrados = []
    ids_buscar = set(lista_ids)  # Convertir a set para busqueda eficiente
    
    for archivo in archivos:
        ruta = os.path.join(ruta_conversaciones, archivo)
        
        with open(ruta, "r", encoding="utf-8") as fh:
            conversacion = json.load(fh)
        
        for mensaje in conversacion:
            id_mensaje = mensaje.get("id")
            
            if id_mensaje in ids_buscar:
                # Preparar texto para DNAPAN
                texto_input = mensaje.get("input", "")
                texto_output = mensaje.get("output", "")
                texto_completo = f"Pregunta: {texto_input}\nRespuesta: {texto_output}"
                
                # Inferir con DNAPAN
                qualification = DNAPAN_inferir_texto(texto_completo)
                
                if qualification is None:
                    qualification = mensaje.get("qualification", "neutra")
                
                mensaje_completo = {
                    "id": mensaje.get("id"),
                    "date": mensaje.get("date"),
                    "input": mensaje.get("input"),
                    "output": mensaje.get("output"),
                    "qualification": qualification,
                    "archivo_origen": archivo
                }
                mensajes_encontrados.append(mensaje_completo)
    
    return mensajes_encontrados


if __name__ == "__main__":
    # Ejecutar la funcion principal
    print("=== EJECUTANDO juntar() ===")
    try:
        resultado = juntar()

        # Mostrar ejemplo del primer mensaje
        if resultado:
            print(f"\n=== PRIMER MENSAJE EJEMPLO ===")
            print(json.dumps(resultado[0], ensure_ascii=False, indent=2))

        # Opcional: ejecutar la version con DNAPAN completo
        print("\n=== EJECUTANDO juntar_con_dnapan_completo() ===")
        resultado_dnapan = juntar_con_dnapan_completo()
    except ErrorAPACMA as e:
        print(f"[ERROR FATAL] {e}")
    except Exception as e:
        print(f"[ERROR FATAL] {e}")