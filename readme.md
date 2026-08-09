# APACMA: Arquitectura de Personalización Continua Mediante Adaptadores

## Resumen

APACMA (Arquitectura de Personalización Continua Mediante Adaptadores) es una arquitectura de inteligencia artificial diseñada para lograr una personalización profunda y permanente de cada usuario sin modificar el modelo base compartido.

La idea central consiste en separar el conocimiento general del modelo de la información específica de cada usuario. En lugar de almacenar toda la personalización dentro de la ventana de contexto o reentrenar el modelo completo, APACMA mantiene un dataset personalizado por usuario que sirve como fuente de conocimiento para generar un adaptador mediante técnicas como LoRA. El adaptador modifica únicamente el comportamiento del modelo respecto a ese usuario, permitiendo que el sistema evolucione continuamente sin duplicar el modelo completo para cada persona.

## Problemas que Resuelve

1.APACMA aborda varios problemas presentes en los sistemas actuales:

2.Dependencia excesiva de la ventana de contexto: Los modelos actuales requieren que las preferencias del usuario se repitan constantemente dentro de la conversación, consumiendo tokens valiosos y limitando la cantidad de información relevante que puede procesarse.

3.Repetición constante de preferencias: El usuario debe recordar y repetir sus preferencias en cada conversación, lo que resulta en una experiencia frustrante y poco natural.

4.Pérdida de personalización entre conversaciones: Cuando una sesión termina, la personalización se pierde y el usuario debe  comenzar de nuevo en la siguiente interacción.

5.modelos mas humanos: hace que los modelos de inteligencia artificial actuen como humanos y evolucionen con el usuario

## Principio Fundamental

El principio central de APACMA establece que la personalización permanente debe almacenarse como conocimiento entrenable del usuario y no únicamente como contexto textual. Esto implica separar dos tipos de información:

La información temporal corresponde a la conversación actual, incluyendo preguntas recientes, documentos abiertos, instrucciones momentáneas y referencias a mensajes anteriores. Esta información pertenece a la ventana de contexto y puede desaparecer cuando termina la conversación.

La información permanente corresponde a características propias del usuario, como el idioma preferido, nivel técnico, tono favorito, estilo de respuestas, preferencias de formato, patrones de corrección y ejemplos positivos y negativos. Esta información pertenece al sistema de personalización y no necesita volver a escribirse en cada conversación.

## Componentes de APACMA

La arquitectura se compone de 6 elementos principales.

El modelo base es el modelo compartido entre todos los usuarios. Contiene conocimiento general, razonamiento, comprensión del lenguaje y capacidades principales. Nunca se personaliza directamente y todos los usuarios utilizan exactamente el mismo modelo base.

La memoria estructurada es una representación organizada de información importante sobre el usuario. No almacena conversaciones completas, sino únicamente hechos relevantes como el idioma, nivel técnico, preferencias de estilo y temas de interés. Esta memoria puede actualizarse continuamente.

El dataset personalizado es el componente central de APACMA. No es el adaptador ni el modelo, sino el verdadero conocimiento del usuario. Contiene ejemplos de entrenamiento obtenidos mediante interacción real, cada uno con su correspondiente entrada, salida y retroalimentación. Con el tiempo, este dataset representa cómo espera el usuario que responda la IA. Es independiente del modelo utilizado.

El adaptador personalizado, típicamente implementado mediante LoRA, es una representación matemática entrenada utilizando el dataset personalizado. No contiene el conocimiento completo del usuario, sino que es simplemente una adaptación del modelo base para reflejar ese conocimiento. Puede regenerarse cuando sea necesario.

un modelo independiente que se encarga de revizar las respuestas para identificar si son relevantes y guardarlas en el dataset.

otro modelo de ai independiente encargado de la revision de las respuestas y entradas de el modelo con el objetivo de que revize que las respuestas no contengan mensajes de violencia, agrecion, cosas ilegales o poco eticas.

## Flujo de Aprendizaje

Cada conversación genera nueva información. El sistema analiza cada interacción y puede detectar aprobación explícita, rechazo, correcciones, preferencias y cambios de estilo. Por ejemplo, cuando un usuario dice "Me gustó esta explicación", el sistema interpreta esa interacción como un ejemplo positivo y lo incorpora al dataset para ser utilizado posteriormente en el entrenamiento del adaptador.

La retroalimentación no depende únicamente de botones. APACMA también puede interpretar lenguaje natural. Expresiones como "Perfecto", "Exactamente así" o "Muy bien explicado" se interpretan como ejemplos positivos, mientras que frases como "Hazlo más corto", "No entendiste" o "Quiero más detalle" se transforman en ejemplos negativos que guían el aprendizaje.

El entrenamiento ocurre de forma periódica, no necesariamente después de cada mensaje. Puede ejecutarse al acumular suficientes ejemplos, cuando el sistema está inactivo, durante la noche o cuando existe capacidad de procesamiento disponible. El entrenamiento genera una nueva versión del adaptador.

## El Papel de la Ventana de Contexto

En APACMA, la ventana de contexto cambia de función. Su objetivo principal deja de ser recordar permanentemente al usuario y se dedica a información temporal como la conversación actual, documentos abiertos, tareas en ejecución e instrucciones recientes. Las preferencias permanentes ya no necesitan repetirse continuamente, lo que permite dedicar una mayor parte del contexto al problema actual. Esto no elimina la necesidad del contexto, pero reduce significativamente el uso de tokens para información estable ademas de la cantidad de prosesamiento necesario.

## El Centro de la Arquitectura

En la mayoría de arquitecturas actuales, el centro suele ser el modelo entrenado o el propio adaptador. En APACMA, el elemento más importante es el dataset personalizado del usuario. Todo lo demás puede reconstruirse: si desaparece el adaptador, se vuelve a entrenar; si cambia el modelo base, se vuelve a entrenar; si aparece una nueva técnica de adaptación, se vuelve a entrenar. Mientras el dataset permanezca intacto, la personalización continúa existiendo.

## Actualización del Modelo Base

Uno de los problemas más importantes en el sistema de IA es cómo actualizar el modelo sin perder la personalización. para solucionarlo APACMA desacopla completamente el dataset del usuario respecto al modelo base. Cuando aparece una nueva versión del modelo, el proceso es simple: se toma el mismo dataset personalizado, se realiza un nuevo entrenamiento y se genera un nuevo adaptador. No existe migración directa entre adaptadores porque el adaptador simplemente se vuelve a generar desde cero utilizando el dataset.

## Gestión del Tamaño del Dataset

El dataset no puede crecer indefinidamente. APACMA propone establecer un presupuesto máximo basado en número de tokens, número de ejemplos o tamaño de almacenamiento. Cuando se alcanza el límite, el sistema puede eliminar ejemplos redundantes, fusionar ejemplos similares, resumir conversaciones repetitivas, conservar únicamente ejemplos representativos o priorizar ejemplos con mayor valor de aprendizaje. De esta forma, el conocimiento importante permanece mientras el tamaño permanece controlado a traves de el modelo correspondiente.

## Manejo de Ejemplos Negativos

Cuando un usuario proporciona retroalimentación negativa, APACMA puede manejar esta información de varias formas. El enfoque más directo consiste en aplicar un peso negativo en la función de pérdida durante el entrenamiento, lo que hace que el modelo aprenda activamente a evitar esa respuesta. También es posible reducir el peso de tokens específicos que el usuario ha indicado como indeseables, haciendo que esas palabras sean menos probables en respuestas futuras. Otra alternativa es el aprendizaje contrastivo, donde las respuestas buenas se acercan en el espacio de representación y las malas se alejan. Cuando el usuario proporciona una corrección explícita, el sistema guarda la versión corregida como ejemplo positivo y la versión original como ejemplo negativo, permitiendo al modelo aprender de ambos casos.

## Ventajas de la Arquitectura

APACMA ofrece personalización continua y adaptación permanente sin necesidad de reentrenar modelos completos. Reduce la dependencia del contexto para recordar preferencias estables, permitiendo que los tokens disponibles se dediquen al contenido relevante. La actualización del modelo base resulta sencilla porque el dataset es independiente del modelo. El entrenamiento es eficiente porque utiliza adaptadores pequeños en lugar de modelos completos. La arquitectura mantiene una separación clara entre conocimiento general y conocimiento personalizado, y el dataset puede reutilizarse ante nuevas versiones del modelo o cambios en la técnica de adaptación.

## Posibilidades Futuras

La arquitectura es deliberadamente independiente del método de adaptación. Aunque inicialmente utilice LoRA, el diseño permite sustituir esa técnica por otras sin modificar el núcleo de APACMA. El dataset personalizado puede servir para regenerar adaptadores tras actualizar el modelo base, evaluar la calidad de nuevas versiones del modelo con los mismos ejemplos del usuario, experimentar con distintas técnicas de personalización sin perder el historial de aprendizaje, o combinar memoria estructurada, ejemplos de entrenamiento y adaptadores para lograr una personalización más robusta.

## Principio Arquitectónico Final

El conocimiento personalizado del usuario no reside en el modelo base ni en el adaptador. Reside en un dataset personalizado, construido de forma continua mediante la interacción y la retroalimentación del usuario. El adaptador es una representación matemática regenerable de ese conocimiento, mientras que el modelo base aporta las capacidades generales del sistema. La ventana de contexto queda reservada principalmente para la información temporal de la conversación, separando así el conocimiento permanente del conocimiento transitorio y facilitando la evolución continua del sistema sin perder la personalización.

Esta separación de responsabilidades es el núcleo conceptual de APACMA y la característica que distingue la arquitectura. La definición anterior sirve como fundamento sobre el que desarrollar los algoritmos específicos, las políticas de actualización y las métricas para evaluar si realmente mejora la personalización respecto a otros enfoques.

## Implementación

La arquitectura se implementa en los siguientes módulos de Python (en la raíz del proyecto):

- `chat/llm.py`: servidor web del chat, calificación de respuestas y persistencia de conversaciones.
- `formato_openai.py`: normalización de conversaciones al formato estándar OpenAI `messages`.
- `json_script.py`: extracción de mensajes calificados para el dataset.
- `clasificador.py`: construcción del dataset de entrenamiento.
- `DNAPAN.py`: modelo que revisa las respuestas para identificar si son relevantes (calificación).
- `seguridad.py`: modelo de revisión que verifica que las respuestas no contengan violencia, agresión, contenido ilegal o poco ético.
- `fine.py`: entrenamiento del adaptador (LoRA) con SFT y unlikelihood para ejemplos negativos.
- `limites.py`: control del presupuesto de tokens del dataset.

Las conversaciones se guardan en formato estándar OpenAI `messages` (misma estructura que usan OpenAI y la industria para chats y datasets de fine-tuning).

**creado por: daniel de anda**
<br>

**documentacion tecnica: carpeta /documentacion**
<br>

**licencia: archivo LICENCE**