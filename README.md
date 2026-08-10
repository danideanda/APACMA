# APACMA: Arquitectura de Personalización Continua Mediante Adaptadores

## Resumen Ejecutivo

APACMA (Arquitectura de Personalización Continua Mediante Adaptadores) es una arquitectura de IA diseñada para ofrecer personalización profunda y permanente por usuario **sin reentrenar ni duplicar el modelo base compartido**. En lugar de inyectar las preferencias del usuario en cada llamada (costo recurrente de tokens) o de mantener un modelo completo por cliente (costo inviable de infraestructura), APACMA almacena el conocimiento de cada usuario en un **dataset personalizado** y materializa ese conocimiento en un **adaptador LoRA** ligero y regenerable.

Para un CTO, el valor se resume en tres frentes:

1. **Control de costos**: la personalización no escala con el costo por token ni con el costo de entrenar modelos completos; el incremento marginal por usuario es un adaptador pequeño entrenado de forma asíncrona.
2. **Escalabilidad arquitectónica**: un solo modelo base sirve a toda la base de usuarios; la separación entre conocimiento general y conocimiento específico permite actualizar el modelo base sin tocar la personalización.
3. **Gobernanza y auditoría**: el conocimiento de cada usuario vive en un dataset versionable y auditable, no embebido de forma opaca en pesos de un modelo.

## Decisión Técnica: ¿Por Qué Esta Arquitectura?

Los enfoques dominantes de personalización presentan limitaciones que una decisión técnica debe evaluar:

- **Dependencia de la ventana de contexto**: repetir las preferencias del usuario en cada conversación consume tokens valiosos y acota la información realmente útil por llamada. El costo crece linealmente con el tráfico, sin retorno de inversión.
- **Fine-tuning completo por usuario**: multiplica infraestructura y mantenimiento por el número de usuarios, un modelo inviable fuera de unos pocos clientes enterprise.
- **Pérdida de personalización entre sesiones**: sin persistencia entrenable, cada nueva conversación parte de cero, degradando la experiencia percibida y la retención.

APACMA resuelve estos problemas separando dos tipos de información:

| Tipo de información | Dónde vive | Ciclo de vida |
|---------------------|------------|---------------|
| **Temporal** (conversación actual, documentos abiertos, instrucciones recientes) | Ventana de contexto | Se descarta al terminar la sesión |
| **Permanente** (idioma, nivel técnico, tono, formato, ejemplos positivos/negativos) | Dataset personalizado → adaptador | Persiste y evoluciona |

## Componentes de la Arquitectura

| Componente | Rol | Propiedad técnica |
|------------|-----|-------------------|
| **Modelo base** | Conocimiento general, razonamiento y capacidades del sistema | Compartido por todos los usuarios; nunca se personaliza directamente |
| **Memoria estructurada** | Hechos relevantes del usuario (idioma, estilo, temas) | Actualizable en caliente |
| **Dataset personalizado** | Ejemplos de entrenamiento reales (input, output, retroalimentación) | **El centro de la arquitectura**: el conocimiento del usuario es este dataset, no el adaptador |
| **Adaptador LoRA** | Representación entrenable del dataset | Pequeño, regenerable, independiente del modelo base |
| **Modelo revisor (DNAPAN)** | Evalúa si las respuestas son relevantes y las incorpora al dataset | Modelo de clasificación independiente |
| **Modelo de seguridad** | Revisa entradas y salidas contra violencia, agresión, ilegalidad y contenido poco ético | Capa de cumplimiento y protección |

## Pipeline de Aprendizaje

El sistema convierte la interacción natural en datos de entrenamiento:

1. Cada conversación se analiza y clasifica (aprobación explícita, rechazo, corrección, preferencia).
2. Retroalimentación en lenguaje natural ("Perfecto", "Hazlo más corto", "No entendiste") se traduce a ejemplos **positivos** y **negativos**.
3. Los ejemplos se acumulan en el dataset personalizado.
4. El adaptador se regenera de forma **asíncrona** (lotes, horario de baja demanda o capacidad ociosa), generando una nueva versión sin interrumpir el servicio.

Esta operación en segundo plano es clave para la operación: el entrenamiento no ocurre en el camino crítico de la conversación.

## Manejo de Ejemplos Negativos

APACMA incorpora el aprendizaje a partir de retroalimentación negativa mediante estrategias combinables:

- **Unlikelihood loss**: reduce activamente la probabilidad de respuestas indeseadas (`-log(1 - p)`) durante el entrenamiento del adaptador.
- **Ponderación de tokens**: baja el peso de términos específicos marcados como indeseables.
- **Corrección explícita**: la versión corregida se guarda como ejemplo positivo y la original como negativo, aprendiendo de ambos.

## Operación y Mantenimiento

### Actualización del modelo base

El dataset es **independiente del modelo**. Al lanzar una nueva versión del modelo base, el proceso es: tomar el mismo dataset personalizado, reentrenar, generar un nuevo adaptador. No existe migración entre adaptadores: el adaptador es una derivada regenerable del dataset. Esto desacopla el roadmap del modelo base de la experiencia de personalización.

### Gestión del tamaño del dataset

El dataset está sujeto a un **presupuesto máximo** (tokens, ejemplos o almacenamiento). Al alcanzarlo, se aplican políticas de prioridad: eliminar redundancias, fusionar ejemplos similares, resumir conversaciones repetitivas, conservar ejemplos representativos o priorizar el mayor valor de aprendizaje.

## Ventajas para la Decisión de Inversión

- **Costo marginal predecible y bajo**: adaptadores pequeños entrenados en recursos ociosos, sin reentrenar modelos completos.
- **Actualización sin fricción**: el cambio de modelo base no rompe la personalización (dataset desacoplado).
- **Eficiencia de cómputo**: se entrena el adaptador, no el modelo.
- **Clara separación de responsabilidades**: conocimiento general vs. conocimiento personalizado.
- **Reutilización**: el dataset sirve para regenerar adaptadores, evaluar nuevas versiones del modelo y experimentar con otras técnicas de adaptación sin perder el historial.

## Implementación

La arquitectura se implementa en módulos Python (en la raíz del proyecto):

- `chat/llm.py`: servidor web del chat, calificación de respuestas y persistencia de conversaciones.
- `formato_openai.py`: normalización de conversaciones al formato estándar OpenAI `messages`.
- `json_script.py`: extracción de mensajes calificados para el dataset.
- `clasificador.py`: construcción del dataset de entrenamiento.
- `DNAPAN.py`: modelo que revisa las respuestas para identificar si son relevantes (calificación).
- `DNAPAN/model.py`: entrenamiento del clasificador DNAPAN (fórmula de épocas y mínimo de 5M de parámetros).
- `seguridad.py`: modelo de revisión que verifica que las respuestas no contengan violencia, agresión, contenido ilegal o poco ético.
- `fine.py`: entrenamiento del adaptador (LoRA) con SFT y unlikelihood para ejemplos negativos.
- `limites.py`: control del presupuesto de tokens del dataset.

Las conversaciones se guardan en formato estándar OpenAI `messages` (misma estructura que usan OpenAI y la industria para chats y datasets de fine-tuning), lo que facilita interoperabilidad y portabilidad del dato.

---

**creado por: daniel de anda**
<br>

**documentacion tecnica: carpeta /documentacion**
<br>

**licencia: archivo LICENCE**
