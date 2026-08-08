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

## Módulos que importa

| Módulo | Uso |
|--------|-----|
| `fine.entrenar_fine` | Ejecuta el fine-tuning con LoRA del modelo. |
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

## Variables globales

| Variable | Valor inicial | Descripción |
|----------|---------------|-------------|
| `model_path` | `""` | Ruta al modelo base; la asigna `verificar_ruta_modelo()`. |
| `dataset_path` | `"json/entrenamiento/dataset.json"` | Ruta del dataset usado en el entrenamiento. |
| `fecha` | `datetime.now()` | Fecha de inicio del programa, usada para decidir si toca nuevo mes. |

## Flujo de ejecución de `main()`

1. `verificar_ruta_modelo()` → asigna `model_path`.
2. Si `fecha.day() == 1` (primer día del mes):
   - Genera el dataset: `juntar()` y `juntar_con_dnapan_completo()`.
   - Imprime `=== FILTRO DE SEGURIDAD DE JSON ===` y ejecuta `filtro_seguridad()`.
   - Ejecuta `entrenar_fine()`.
   - Imprime `=== PRUEBA BREVE DE SEGURIDAD DEL MODELO ===` y ejecuta `prueba_seguridad_modelo()`.
   - Imprime `Proceso completado`.

## Código a nivel de módulo

- Se calcula `fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")` y se imprime `Inicio del programa: {fecha}` al importar el módulo.

## Flujo de ejecución

```
inicio del script
     │
     ▼
fecha = datetime.now(...)  →  print "Inicio del programa"
     │
     ▼
if __name__ == "__main__":
     │
     ▼
main()  →  verificar_ruta_modelo()  →  detecta ruta del modelo
     │                                  │
     │                                  ▼
     │         si fecha.day() == 1  →  juntar() / juntar_con_dnapan_completo()
     │              → filtro_seguridad() → entrenar_fine() → prueba_seguridad_modelo()
     ▼
print("Ruta del modelo verificada:", model_path)
     │
     ▼
entrenar_fine(modelo_path=model_path)  →  fine-tuning del LLM
     │
     ▼
fin
```

## Notas y advertencias

- El proceso mensual (dataset, filtro, fine-tuning y prueba de seguridad) solo se ejecuta cuando `fecha.day() == 1`; en cualquier otro día `main()` no hace nada.
- `main.py` se ejecuta como módulo principal mediante `if __name__ == "__main__":`.
- El proyecto depende de los módulos importados; si faltan dependencias (transformers, torch), el import fallará al inicio.
- `entrenar_fine` se invoca con `modelo_path=model_path`, guardando el modelo en `models/LLM/`.
- `fecha.day()` se llama como método en `main()`; si `datetime` se importa como módulo y `fecha` es un objeto `datetime`, `day` es un atributo (sin paréntesis). Revisar si el paréntesis extra provoca un error en tiempo de ejecución.
