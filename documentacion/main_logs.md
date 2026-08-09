# main_logs.md

Archivo de documentación técnica para `main_logs.py`.

## ¿Qué hace el archivo?

`main_logs.py` es el **punto de entrada principal del proyecto APACMA**, con soporte de **logging** integrado. Su función es unificar y coordinar todos los módulos del sistema (seguridad, clasificación, fine-tuning y chat LLM), registrando cada paso en un archivo de log con fecha/hora.

- Define y verifica la ruta del modelo a usar (`models/LLM` o `models/LLM-base`).
- Configura el sistema de logs en `test/logs/` con un archivo único por ejecución.
- Registra la fecha de inicio del programa.
- Orquesta el flujo de trabajo mensual: generación de dataset, filtro de seguridad, fine-tuning y prueba de seguridad.
- Al ejecutarse como script, imprime y registra la ruta del modelo verificada y lanza el pipeline.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `test/logs/` | Directorio | Carpeta donde se guardan los logs (se crea si no existe). |
| `test/logs/main_<fecha>.log` | Archivo (salida) | Log de la ejecución, con formato `main_YYYYMMDD_HHMMSS.log`. |
| `json/entrenamiento/dataset.json` | Archivo (entrada) | Dataset usado para el entrenamiento. |
| `models/LLM` | Directorio | Modelo de lenguaje base (preferido). |
| `models/LLM-base` | Directorio | Modelo de lenguaje alternativo. |

## Módulos que importa

| Módulo | Uso |
|--------|-----|
| `fine.entrenar_fine` | Ejecuta el fine-tuning con LoRA del modelo. |
| `seguridad.filtro_seguridad` | Limpia los JSON mediante el filtro de seguridad. |
| `seguridad.prueba_seguridad_modelo` | Prueba breve de seguridad del modelo entrenado. |
| `clasificador.juntar` | Genera el dataset a partir de los datos de entrenamiento. |
| `clasificador.juntar_con_dnapan_completo` | Genera el dataset completo usando DNAPAN. |
| `chat.llm` (wildcard `*`) | Servidor web del chat y motor del LLM. |
| `datetime` | Genera la fecha/hora de inicio del programa y del nombre del log. |
| `logging` | Sistema de registro de logs (consola y archivo). |
| `os` | Manejo de rutas y creación de directorios. |
| `time` | Pausa del bucle (`time.sleep`) cuando no toca ejecutar el pipeline. |
| `functools.wraps` | Preserva los metadatos de las funciones envueltas por el decorador. |

## Tecnologías

- **Python** (lenguaje principal).
- **`logging`** (biblioteca estándar) para registro estructurado en consola y archivo.
- **Transformers / Hugging Face** (vía módulos importados: LLM, fine-tuning).
- **PyTorch** (vía módulos importados para inferencia y entrenamiento).

## Configuración de logs

La sección `# ========== configuración de logs ==========` prepara el sistema de registro:

- `DIRECTORIO_LOGS = "test/logs"` → se crea con `os.makedirs(..., exist_ok=True)`.
- `RUTA_LOG` → `test/logs/main_<YYYYMMDD_HHMMSS>.log`, único por ejecución.
- `logging.basicConfig` con nivel `INFO`, formato `%(asctime)s - %(levelname)s - %(message)s` y dos *handlers*:
  - `logging.FileHandler` → escribe en el archivo de log (UTF-8).
  - `logging.StreamHandler` → muestra en consola.
- `log = logging.getLogger("APACMA")` → *logger* del proyecto usado en todo el archivo.

## Manejo de errores

La sección `# ========== manejo de errores ==========` contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`_SIN_DEFAULT`**: centinela interno (`object()`) para distinguir cuándo no se indicó `default`.
- **`manejar_errores`**: decorador que captura excepciones, registra `[ERROR] <función>: <mensaje>` y:
  - Relanza como `ErrorAPACMA` si no se indicó `default`.
  - Devuelve el valor `default` si se indicó (`@manejar_errores(default=[])`).
  - Re-lanza directamente los `ErrorAPACMA` originales sin envolverlos.

Se aplica a `verificar_ruta_modelo` y `main`. El bloque `if __name__ == "__main__":` se envuelve en `try/except ErrorAPACMA` y `except Exception`, registrando `[ERROR FATAL]`.

## Lógica de selección de modelo

`verificar_ruta_modelo()` elige la ruta del modelo en orden de prioridad:

1. Si existe `models/LLM` → usa `models/LLM`.
2. Si no, pero existe `models/LLM-base` → usa `models/LLM-base`.
3. Si ninguno existe → `model_path` contiene `"error fatal: no se encontró la ruta del modelo"`.

## Funciones principales

| Función | Rol |
|---------|-----|
| `verificar_ruta_modelo()` | Detecta y asigna `model_path` global entre los modelos disponibles. |
| `main()` | Orquestador principal del proceso mensual. |

## Variables globales

| Variable | Valor inicial | Descripción |
|----------|---------------|-------------|
| `model_path` | `""` | Ruta al modelo base; la asigna `verificar_ruta_modelo()`. |
| `dataset_path` | `"json/entrenamiento/dataset.json"` | Ruta del dataset usado en el entrenamiento. |
| `fecha` | `datetime.now()` | Fecha de inicio del programa, usada para decidir si toca nuevo mes. |
| `DIRECTORIO_LOGS` | `"test/logs"` | Carpeta de salida de los logs. |
| `RUTA_LOG` | `"test/logs/main_<fecha>.log"` | Ruta completa del archivo de log de la ejecución. |

## Flujo de ejecución de `main()`

1. `verificar_ruta_modelo()` → asigna `model_path`.
2. Bucle `while True`:
   - Si `fecha.day == 9` (en pruebas; en producción debería ser `1`):
     - Registra `=== PASO 1: GENERAR DATASET ===` → `juntar()` y `juntar_con_dnapan_completo()`.
     - Registra `=== PASO 2: FILTRO DE SEGURIDAD DE JSON ===` → `filtro_seguridad()`.
     - Registra `=== PASO 3: ENTRENAMIENTO (FINE-TUNING) ===` → `entrenar_fine()`.
     - Registra `=== PASO 4: PRUEBA BREVE DE SEGURIDAD DEL MODELO ===` → `prueba_seguridad_modelo()`.
     - Registra `Proceso completado` y rompe el bucle.
   - Si no se cumple la condición de día: registra `Hoy no corresponde ejecutar el pipeline...` y espera 60 segundos (`time.sleep(60)`), volviendo a comprobar.

## Código a nivel de módulo

- Se calcula `fecha_inicio = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")` y se registra `Inicio del programa: {fecha_inicio}` y `Logs guardados en: {RUTA_LOG}` al importar el módulo.
- La variable `fecha` (objeto `datetime`) se conserva para la comprobación mensual en `main()`.

## Flujo de ejecución

```
inicio del script
     │
     ▼
configuración de logs  →  test/logs/main_<fecha>.log  (archivo + consola)
     │
     ▼
fecha_inicio = datetime.now(...)  →  log "Inicio del programa"
                                     log "Logs guardados en: RUTA_LOG"
     │
     ▼
if __name__ == "__main__":
     │
     ▼
main()  →  verificar_ruta_modelo()  →  detecta ruta del modelo
     │                                  │
     │                                  ▼
     │      while True:
     │         si fecha.day == 9  →  PASO 1: juntar() / juntar_con_dnapan_completo()
     │              → PASO 2: filtro_seguridad() → PASO 3: entrenar_fine()
     │              → PASO 4: prueba_seguridad_modelo() → "Proceso completado" → break
     │         sino → log "Hoy no corresponde..." → time.sleep(60)
     ▼
log "Ruta del modelo verificada: model_path"
     │
     ▼
fin
```

## Notas y advertencias

- El proceso mensual (dataset, filtro, fine-tuning y prueba de seguridad) solo se ejecuta cuando `fecha.day == 9`. El comentario del código indica que debería ser el día `1` (primer día del mes) y que actualmente se está en fase de pruebas con `9` (menciona también `10` en el comentario).
- A diferencia de `main.py`, este módulo no termina si no toca el día: **se queda en un bucle infinito** comprobando cada 60 segundos hasta que la condición de día se cumpla.
- `main_logs.py` se ejecuta como módulo principal mediante `if __name__ == "__main__":`.
- El proyecto depende de los módulos importados; si faltan dependencias (transformers, torch), el import fallará al inicio.
- `entrenar_fine` se invoca sin argumentos (usa sus variables globales), guardando el modelo en `models/LLM/`.
- El log se escribe con codificación UTF-8 para soportar mensajes en español.
