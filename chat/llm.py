import json
import os
import re
from datetime import datetime
from functools import wraps

from flask import Flask, Response, jsonify, request, stream_with_context
from threading import Thread

from transformers import GenerationConfig

import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)) + "/..")
from formato_openai import (
    leer_conversacion,
    guardar_conversacion,
    obtener_mensajes,
    mensajes_para_llm,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "LLM-base")
CONVERSACIONES_DIR = os.path.join(PROJECT_DIR, "json", "conversaciones")

os.makedirs(CONVERSACIONES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

model = None
tokenizer = None
chat_actual = None

# ========== contexto global entre chats ==========
RUTA_CONTEXTO_GLOBAL = os.path.join(PROJECT_DIR, "json", "contexto_global.json")
MAX_CONTEXTO_GLOBAL = 20
CONTEXTO_SISTEMA = (
    "Eres APACMA, un asistente que recuerda el contexto compartido entre "
    "todas las conversaciones. Responde de forma útil, natural y coherente "
    "con lo que se ha hablado antes."
)

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

@manejar_errores
def nombre_archivo(nombre):
    nombre = re.sub(r"[^A-Za-z0-9\-_]", "-", nombre) or "chat"
    return os.path.join(CONVERSACIONES_DIR, nombre + ".json")


@manejar_errores(default=[])
def listar_chats():
    chats = []
    for f in sorted(os.listdir(CONVERSACIONES_DIR)):
        if f.endswith(".json"):
            chats.append(f[:-5])
    return chats


@manejar_errores(default=[])
def cargar_chat(nombre):
    global chat_actual
    chat_actual = nombre
    ruta = nombre_archivo(nombre)
    if os.path.exists(ruta):
        try:
            return leer_conversacion(ruta)
        except Exception:
            pass
    return {"messages": []}


@manejar_errores
def guardar_chat(nombre, conversacion):
    ruta = nombre_archivo(nombre)
    return guardar_conversacion(ruta, conversacion)


# ========== contexto global entre chats ==========

@manejar_errores(default={"messages": [{"role": "system", "content": CONTEXTO_SISTEMA}]})
def cargar_contexto_global():
    """Carga el contexto global entre chats en formato OpenAI messages."""
    if os.path.exists(RUTA_CONTEXTO_GLOBAL):
        with open(RUTA_CONTEXTO_GLOBAL, encoding="utf-8") as fh:
            return json.load(fh)
    contexto = {"messages": [{"role": "system", "content": CONTEXTO_SISTEMA}]}
    guardar_contexto_global(contexto)
    return contexto


@manejar_errores
def guardar_contexto_global(contexto):
    """Guarda el contexto global entre chats en formato OpenAI messages."""
    with open(RUTA_CONTEXTO_GLOBAL, "w", encoding="utf-8") as fh:
        json.dump(contexto, fh, ensure_ascii=False, indent=4)


@manejar_errores
def agregar_a_contexto_global(usuario, respuesta):
    """Agrega un par user->assistant al contexto global y lo recorta."""
    contexto = cargar_contexto_global()
    contexto.setdefault("messages", []).extend([
        {"role": "user", "content": usuario},
        {"role": "assistant", "content": respuesta},
    ])
    contexto["messages"] = contexto["messages"][-MAX_CONTEXTO_GLOBAL:]
    guardar_contexto_global(contexto)


@manejar_errores(default=[])
def mensajes_con_contexto(mensajes):
    """Prefija el contexto global (formato OpenAI) a los mensajes actuales."""
    contexto = cargar_contexto_global()
    return obtener_mensajes(contexto) + mensajes



def generar_respuesta_stream(mensajes):
    if model is None:
        print("no hay modelo local, respondiendo con contexto global")
        yield "no hay modelo disponible en local, se mantiene el contexto global"
        return
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        ids = tokenizer.apply_chat_template(mensajes, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        config = GenerationConfig(max_new_tokens=512, do_sample=True, temperature=0.7)
        thread = Thread(target=model.generate, kwargs={"input_ids": ids["input_ids"], "generation_config": config, "streamer": streamer})
        thread.start()
        for token in streamer:
            yield token
    except Exception as e:
        print("error en generacion:", e)
        yield "no hay modelo disponible en local"


app = Flask(__name__)


@app.route("/")
def index():
    return open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read()


@app.route("/chats", methods=["GET"])
def chats():
    return jsonify({"chats": listar_chats(), "actual": chat_actual})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    nombre = data.get("nombre")
    if not nombre:
        return jsonify({"error": "falta nombre"}), 400
    conversacion = cargar_chat(nombre)
    return jsonify({"conversacion": conversacion, "actual": chat_actual})


@app.route("/nuevo", methods=["POST"])
def nuevo():
    n = 1
    while "conversacion-" + str(n) in listar_chats():
        n += 1
    nombre = "conversacion-" + str(n)
    guardar_chat(nombre, {"messages": []})
    cargar_chat(nombre)
    return jsonify({"nombre": nombre, "actual": chat_actual})


@app.route("/enviar", methods=["POST"])
def enviar():
    data = request.get_json(force=True)
    texto = data.get("mensaje", "")
    nombre = data.get("nombre") or chat_actual
    if not nombre:
        return jsonify({"error": "no hay chat activo"}), 400
    conversacion = cargar_chat(nombre)

    # Construir mensajes para el modelo con el contexto global entre chats
    mensajes = mensajes_para_llm(conversacion)
    mensajes = mensajes_con_contexto(mensajes)
    mensajes.append({"role": "user", "content": texto})

    def generar():
        respuesta = ""
        for fragmento in generar_respuesta_stream(mensajes):
            respuesta += fragmento
            yield fragmento

        nuevo_mensaje = {
            "id": len(obtener_mensajes(conversacion)) // 2 + 1,
            "date": datetime.now().isoformat(),
            "role": "user",
            "content": texto,
        }
        respuesta_mensaje = {
            "id": len(obtener_mensajes(conversacion)) // 2 + 1,
            "date": datetime.now().isoformat(),
            "role": "assistant",
            "content": respuesta,
            "qualification": "neutra",
        }
        conversacion.setdefault("messages", []).extend([nuevo_mensaje, respuesta_mensaje])
        guardar_chat(nombre, conversacion)
        agregar_a_contexto_global(texto, respuesta)

    return Response(stream_with_context(generar()), mimetype="text/plain", headers={"X-Accel-Buffering": "no"})


@app.route("/calificar", methods=["POST"])
def calificar():
    data = request.get_json(force=True)
    nombre = data.get("nombre") or chat_actual
    mid = data.get("id")
    calificacion = data.get("calificacion", "negative")
    if not nombre:
        return jsonify({"error": "no hay chat activo"}), 400
    conversacion = cargar_chat(nombre)
    for m in obtener_mensajes(conversacion):
        if m.get("role") == "assistant" and m.get("id") == mid:
            m["qualification"] = calificacion
            break
    guardar_chat(nombre, conversacion)
    return jsonify({"ok": True, "actual": nombre})


@manejar_errores
def run_server():
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODELS_DIR)
        print("modelo cargado desde " + MODELS_DIR)
    except ImportError:
        print("instala transformers con: pip install transformers torch")
    except Exception as e:
        print("no se pudo cargar el modelo local, se usara contexto global:", e)
    if not listar_chats():
        guardar_chat("conversacion-1", [])
    cargar_chat("conversacion-1")
    cargar_contexto_global()
    run_server()