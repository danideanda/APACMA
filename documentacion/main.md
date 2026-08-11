# main.md

Archivo de documentación técnica para `main.py`.

## ¿Qué hace el archivo?

`main.py` es el **punto de entrada principal** del proyecto. Su función es unificar y coordinar todos los módulos del sistema (seguridad, clasificación, fine-tuning y chat LLM). Actualmente:

- Define y verifica la ruta del modelo a usar (`models/LLM` o `models/LLM-base`).
- Registra la fecha de inicio del programa.
- Orquesta el flujo de trabajo mensual: generación de dataset, filtro de seguridad, fine-tuning y prueba de seguridad.
- Al ejecutarse como script, imprime la ruta del modelo verificada y lanza el fine-tuning.

## Rutas, directorios y archivos

| Ruta | Tipo | Descripción |
|------|------|-------------|
| `json/entrenamiento/dataset.json` | Archivo (entrada) | Dataset usado para el entrenamiento. |
| `models/LLM` | Directorio | Modelo de lenguaje base (preferido). |
| `models/LLM-base` | Directorio | Modelo de lenguaje alternativo. |
| `models/LLM/model.json` | Archivo (entrada) | Registro del último entrenamiento: version, fecha y modelo base. Usado por `ya_entrenado_hoy()`. |

## Módulos que importa

| Módulo | Uso |
|--------|-----|
| `fine.entrenar_fine` | Ejecuta el fine-tuning con LoRA del modelo. |
| `fine.ya_entrenado_hoy` | Comprueba si el modelo ya fue entrenado hoy leyendo `models/LLM/model.json`. |
| `seguridad.filtro_seguridad` | Limpia los JSON mediante el filtro de seguridad. |
| `seguridad.prueba_seguridad_modelo` | Prueba breve de seguridad del modelo entrenado. |
| `clasificador.juntar` | Genera el dataset a partir de los datos de entrenamiento. |
| `clasificador.juntar_con_dnapan_completo` | Genera el dataset completo usando DNAPAN. |
| `chat.llm` (wildcard `*`) | Servidor web del chat y motor del LLM. |
| `datetime` | Genera la fecha/hora de inicio del programa. |

## Tecnologías

- **Python** (lenguaje principal).
- **Transformers / Hugging Face** (vía módulos importados: LLM, fine-tuning).
- **PyTorch** (vía módulos importados para inferencia y entrenamiento).

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

## Manejo de errores

Se define la sección `# ========== manejo de errores ==========` (entre variables y funciones) que contiene:

- **`ErrorAPACMA(Exception)`**: excepción base del proyecto.
- **`manejar_errores`**: decorador que captura excepciones, imprime `[ERROR] <función>: <mensaje>` y relanza como `ErrorAPACMA`, o devuelve un `default` si se indica (`@manejar_errores(default=[])`).

Se aplica a `verificar_ruta_modelo` y `main`. El bloque `if __name__ == "__main__":` se envuelve en `try/except ErrorAPACMA` y `except Exception`, imprimiendo `[ERROR FATAL]`.

## Variables globales

| Variable | Valor inicial | Descripción |
|----------|---------------|-------------|
| `model_path` | `""` | Ruta al modelo base; la asigna `verificar_ruta_modelo()`. |
| `dataset_path` | `"json/entrenamiento/dataset.json"` | Ruta del dataset usado en el entrenamiento. |
| `fecha` | `datetime.now()` | Fecha de inicio del programa, usada para decidir si toca nuevo mes. |

## Flujo de ejecución de `main()`

1. `verificar_ruta_modelo()` → asigna `model_path`.
2. Si `fecha.day == 1` (primer día del mes):
   - Si `ya_entrenado_hoy()` devuelve `True` (el modelo ya fue entrenado hoy según `models/LLM/model.json`) → imprime `El modelo ya fue entrenado hoy. No se dispara el proceso programado.` y pasa a la siguiente iteración del bucle. El pipeline NO se ejecuta.
   - En caso contrario (no entrenado hoy), ejecuta:
     - Genera el dataset: `juntar()` y `juntar_con_dnapan_completo()`.
     - Imprime `=== FILTRO DE SEGURIDAD DE JSON ===` y ejecuta `filtro_seguridad()`.
     - Ejecuta `entrenar_fine()`.
     - Imprime `=== PRUEBA BREVE DE SEGURIDAD DEL MODELO ===` y ejecuta `prueba_seguridad_modelo()`.
     - Imprime `Proceso completado`.

## Código a nivel de módulo

- Se calcula `fecha_inicio = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")` y se imprime `Inicio del programa: {fecha_inicio}` al importar el módulo. La variable `fecha` (objeto `datetime`) se conserva para la comprobación mensual en `main()`.

## Flujo de ejecución

```
inicio del script
     │
     ▼
fecha_inicio = datetime.now(...)  →  print "Inicio del programa"
     │
     ▼
if __name__ == "__main__":
     │
     ▼
main()  →  verificar_ruta_modelo()  →  detecta ruta del modelo
     │                                  │
     │                                  ▼
     │         si fecha.day == 1  →  juntar() / juntar_con_dnapan_completo()
     │              → filtro_seguridad() → entrenar_fine() → prueba_seguridad_modelo()
     ▼
print("Ruta del modelo verificada:", model_path)
     │
     ▼
entrenar_fine()  →  fine-tuning del LLM
     │
     ▼
fin
```

## Notas y advertencias

- El proceso mensual (dataset, filtro, fine-tuning y prueba de seguridad) solo se ejecuta cuando `fecha.day == 1`, y adicionalmente solo si el modelo no fue ya entrenado hoy (controlado por `ya_entrenado_hoy()` leyendo `models/LLM/model.json`).
- `main.py` se ejecuta como módulo principal mediante `if __name__ == "__main__":`.
- El proyecto depende de los módulos importados; si faltan dependencias (transformers, torch), el import fallará al inicio.
- `entrenar_fine` se invoca sin argumentos (usa sus variables globales), guardando el modelo en `models/LLM/`.
