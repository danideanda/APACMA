import json
import os

# ===== variables =====
ruta_conversaciones = "json/conversaciones"
ruta_dataset = "json/entrenamiento/dataset_filtrado.json"


def listar_chats():
    """
    Lista los chats, procesa los mensajes con qualification y genera el dataset.
    Retorna una lista con los chats procesados.
    """
    # 1. Obtener todos los archivos JSON
    archivos = [
        f for f in sorted(os.listdir(ruta_conversaciones))
        if f.endswith(".json")
    ]
    
    if not archivos:
        print("No se encontraron archivos JSON en la carpeta de conversaciones")
        return []
    
    print(f"Se encontraron {len(archivos)} chats: {archivos}")
    
    # 2. Procesar cada chat y extraer mensajes con qualification
    chats_procesados = []
    dataset = []
    
    for archivo in archivos:
        ruta = os.path.join(ruta_conversaciones, archivo)
        
        with open(ruta, "r", encoding="utf-8") as fh:
            conversacion = json.load(fh)
        
        # Procesar mensajes del chat actual
        mensajes_chat = []
        for mensaje in conversacion:
            qualification = mensaje.get("qualification")
            text = mensaje.get("text")
            
            # Solo guardar mensajes con clasificación
            if qualification and text:
                item = {
                    "text": text,
                    "label": qualification
                }
                mensajes_chat.append(item)
                dataset.append(item)
        
        # Guardar el chat procesado
        if mensajes_chat:
            chats_procesados.append({
                "archivo": archivo,
                "mensajes": mensajes_chat,
                "total": len(mensajes_chat)
            })
    
    # 3. Guardar dataset completo
    if dataset:
        os.makedirs(os.path.dirname(ruta_dataset), exist_ok=True)
        
        with open(ruta_dataset, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, ensure_ascii=False, indent=4)
        
        print(f"Se guardaron {len(dataset)} ejemplos en {ruta_dataset}")
    else:
        print("No se encontraron mensajes con qualification")
    
    # 4. Retornar la lista completa con todo el procesamiento
    return {
        "archivos": archivos,
        "chats_procesados": chats_procesados,
        "total_mensajes": len(dataset),
        "dataset": dataset,
        "ruta_dataset": ruta_dataset
    }


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
        
        with open(ruta, "r", encoding="utf-8") as fh:
            conversacion = json.load(fh)
        
        # Buscar mensajes con qualification positive o negative
        for mensaje in conversacion:
            qualification = mensaje.get("qualification", "").lower()
            id_mensaje = mensaje.get("id")
            
            # Solo extraer si es positive o negative y tiene ID
            if qualification in ["positive", "negative"] and id_mensaje is not None:
                ids_encontrados.append({
                    "id": id_mensaje,
                    "archivo": archivo,
                    "qualification": qualification,
                    "input": mensaje.get("input", ""),
                    "output": mensaje.get("output", "")
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