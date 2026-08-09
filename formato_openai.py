import json

# ========== variables ==========
ROLES_VALIDOS = {"system", "user", "assistant"}
CLAVE_CUALIFICACION = "qualification"


# ========== deteccion de formato ==========

def es_formato_mensajes(conversacion):
    """True si la conversacion ya usa el formato OpenAI (clave 'messages')."""
    if isinstance(conversacion, dict):
        return "messages" in conversacion
    if isinstance(conversacion, list) and conversacion:
        return isinstance(conversacion[0], dict) and "role" in conversacion[0]
    return False


def es_formato_antiguo(conversacion):
    """True si la conversacion usa el formato de pares input/output."""
    if isinstance(conversacion, list) and conversacion:
        return isinstance(conversacion[0], dict) and "input" in conversacion[0]
    return False


# ========== normalizacion ==========

def _limpiar_mensaje(mensaje):
    """Elimina prefijos residuales 'user\\n'/'assistant\\n' del content."""
    limpio = dict(mensaje)
    content = limpio.get("content", "")
    if isinstance(content, str):
        content = content.lstrip("\n")
        for prefijo in ("user\n", "assistant\n"):
            if content.startswith(prefijo):
                content = content[len(prefijo):]
        limpio["content"] = content
    return limpio


def normalizar_a_mensajes(conversacion):
    """
    Convierte cualquier formato al estandar OpenAI: dict con clave 'messages'.

    Acepta:
        - dict con 'messages' (ya normalizado).
        - lista de pares {id, input, output, qualification} (formato antiguo).
        - lista de mensajes {role, content}.
    """
    if isinstance(conversacion, dict) and "messages" in conversacion:
        return conversacion

    if isinstance(conversacion, list):
        mensajes = []
        for m in conversacion:
            if not isinstance(m, dict):
                continue
            if "role" in m and "content" in m:
                mensajes.append(_limpiar_mensaje(m))
            elif "input" in m:
                cualificacion = m.get(CLAVE_CUALIFICACION)
                id_mensaje = m.get("id")
                fecha = m.get("date")
                usuario = _limpiar_mensaje({"role": "user", "content": m.get("input", "")})
                asistente = _limpiar_mensaje({"role": "assistant", "content": m.get("output", "")})
                if id_mensaje is not None:
                    usuario["id"] = id_mensaje
                    asistente["id"] = id_mensaje
                if fecha is not None:
                    usuario["date"] = fecha
                    asistente["date"] = fecha
                if cualificacion is not None:
                    asistente[CLAVE_CUALIFICACION] = cualificacion
                mensajes.append(usuario)
                mensajes.append(asistente)
        return {"messages": mensajes}

    return {"messages": []}


def leer_conversacion(ruta):
    """Lee un archivo de chat y lo devuelve en formato OpenAI messages."""
    with open(ruta, encoding="utf-8") as fh:
        data = json.load(fh)
    return normalizar_a_mensajes(data)


def guardar_conversacion(ruta, conversacion):
    """Guarda una conversacion en formato OpenAI messages."""
    data = normalizar_a_mensajes(conversacion)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)
    return ruta


# ========== acceso ==========

def obtener_mensajes(conversacion):
    """Devuelve la lista de mensajes {role, content[, qualification]}."""
    return normalizar_a_mensajes(conversacion).get("messages", [])


def mensajes_para_llm(conversacion):
    """Devuelve solo {role, content} (sin metadatos) para el LLM."""
    return [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in obtener_mensajes(conversacion)
        if m.get("content")
    ]


def pares_entrenamiento(conversacion):
    """
    Devuelve los pares user->assistant como {input, output, qualification}.

    La qualification se toma del mensaje assistant.
    """
    mensajes = obtener_mensajes(conversacion)
    pares = []
    for i in range(len(mensajes) - 1):
        actual = mensajes[i]
        siguiente = mensajes[i + 1]
        if actual.get("role") == "user" and siguiente.get("role") == "assistant":
            pares.append({
                "id": siguiente.get("id"),
                "date": siguiente.get("date"),
                "input": actual.get("content", ""),
                "output": siguiente.get("content", ""),
                "qualification": siguiente.get(CLAVE_CUALIFICACION),
            })
    return pares
