import json
import os
import re
from datetime import datetime
from functools import wraps

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
import io
from threading import Thread
import queue

from transformers import GenerationConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "LLM-base")
CONVERSACIONES_DIR = os.path.join(PROJECT_DIR, "json", "conversaciones")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
API_BASE_URL = "http://localhost:11434/v1"

os.makedirs(CONVERSACIONES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

model = None
tokenizer = None
chat_actual = None

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
            with open(ruta, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return []


@manejar_errores
def guardar_chat(nombre, conversacion):
    ruta = nombre_archivo(nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(conversacion, f, ensure_ascii=False, indent=4)
    return ruta



def generar_respuesta_stream(mensajes):
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


def completar_openai_stream(mensajes):
    if not os.environ.get("OPENAI_API_KEY"):
        yield from generar_respuesta_stream(mensajes)
        return
    headers = {
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
        "Content-Type": "application/json",
    }
    payload = {"model": DEFAULT_MODEL, "messages": mensajes, "stream": True}
    r = requests.post(API_BASE_URL + "/chat/completions", json=payload, headers=headers, timeout=120, stream=True)
    r.raise_for_status()
    for linea in r.iter_lines():
        if not linea:
            continue
        linea = linea.decode("utf-8")
        if not linea.startswith("data:"):
            continue
        contenido = linea[len("data:"):].strip()
        if contenido == "[DONE]":
            break
        try:
            delta = json.loads(contenido)["choices"][0]["delta"].get("content", "")
        except Exception:
            continue
        if delta:
            yield delta


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
    guardar_chat(nombre, [])
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

    # Construir mensajes para el modelo
    mensajes = []
    for m in conversacion:
        if "input" in m:
            mensajes.append({"role": "user", "content": m["input"]})
        if "output" in m:
            mensajes.append({"role": "assistant", "content": m["output"]})
    mensajes.append({"role": "user", "content": texto})

    def generar():
        respuesta = ""
        for fragmento in completar_openai_stream(mensajes):
            respuesta += fragmento
            yield fragmento

        nuevo_registro = {
            "id": len(conversacion) + 1,
            "date": datetime.now().isoformat(),
            "input": texto,
            "output": respuesta,
            "qualification": "neutra"
        }
        conversacion.append(nuevo_registro)
        guardar_chat(nombre, conversacion)

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
    for m in conversacion:
        if m["id"] == mid:
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
        raise SystemExit(1)
    except Exception as e:
        print("no se pudo cargar el modelo:", e)
        raise SystemExit(1)
    if not listar_chats():
        guardar_chat("conversacion-1", [])
    cargar_chat("conversacion-1")
    run_server()