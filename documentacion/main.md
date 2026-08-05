# main.md

Archivo de documentación técnica para `main.py`.

## ¿Qué hace el archivo?

`main.py` es el **punto de entrada principal** del proyecto. Su función es unificar y coordinar todos los módulos del sistema (backend, frontend, modelos, seguridad y clasificación). Actualmente:

- Define y verifica la ruta del modelo a usar (`models/LLM` o `models/LLM-base`).
- Orquesta el flujo de trabajo, llamando al entrenamiento de fine-tuning al final (`entrenar_fine`).
- Es la pieza que integra los distintos subsistemas: DNAPAN, seguridad, clasificación y chat LLM.

> ⚠️ **Estado:** el cuerpo de la función `main()` está en desarrollo. Hay un `return` sin lógica funcional definida todavía.

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
| `DNAPAN` (wildcard `*`) | Inferencia de agrado positivo/negativo. |
| `seguridad` (wildcard `*`) | Gestión de seguridad y ejecución. |
| `clasificador` (wildcard `*`) | Clasificación y construcción de datasets. |
| `chat.llm` (wildcard `*`) | Servidor web del chat y motor del LLM. |

## Tecnologías

- **Python** (lenguaje principal).
- **Transformers / Hugging Face** (vía módulos importados: LLM, DNAPAN, fine-tuning).
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
| `main()` | Orquestador principal (en desarrollo). |

## Flujo de ejecución

```
inicio del script
     │
     ▼
verificar_ruta_modelo()  →  detecta ruta del modelo
     │
     ▼
main()
     │
     └─ (desarrollo: return sin lógica)
     │
     ▼
entrenar_fine(modelo_path=model_path)  →  fine-tuning del LLM
     │
     ▼
fin
```

## Notas y advertencias

- La línea `return` dentro de `main()` impide que haya lógica ejecutable por el momento.
- `main.py` se ejecuta como módulo principal mediante `if __name__ == "__main__":`.
- El proyecto depende de los módulos importados; si faltan dependencias (transformers, torch), el import fallará al inicio.