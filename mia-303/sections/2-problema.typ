= Planteamiento del Problema
En el contexto de la Industria 4.0 y la transformación digital, el volumen de información técnica disponible en las operaciones industriales ha crecido exponencialmente. En sectores críticos como la minería, y específicamente en las *plantas concentradoras de cobre*, la documentación técnica que incluye manuales de instalación, operación y mantenimiento (IOM), catálogos de partes, hojas de especificaciones, filosofías de control, arreglos generales y reportes de calidad— constituye la base fundamental para la toma de decisiones en mantenimiento. Estos activos de información son vitales para garantizar la continuidad operativa; sin embargo, suelen presentarse en formatos heterogéneos y no estructurados (PDFs escaneados, imágenes, planos y diagramas), lo que dificulta su gestión y accesibilidad inmediata.\

Actualmente, la recuperación de información específica dentro de estos documentos presenta limitaciones significativas. Los técnicos, ingenieros y planificadores dependen de técnicas tradicionales como la lectura secuencial, el uso de índices estáticos (tablas de contenido), la búsqueda por palabras clave (_keyword search_) y búsqueda secuencial. Estos métodos resultan ineficientes ante la complejidad de los manuales modernos y antiguos, los cuales suelen caracterizarse por su gran extensión, terminología técnica ambigua y barreras idiomáticas. Más aún, existe una *desconexión semántica* en la búsqueda: la información crítica a menudo reside en formatos multimodales (como un plano de despiece o un diagrama de flujo) que los motores de búsqueda de texto tradicionales no pueden interpretar ni relacionar con las instrucciones escritas.\

Esta ineficiencia en la gestión del conocimiento impacta directamente en la gestión del mantenimiento. Una planificación efectiva requiere identificar con precisión el procedimiento de cambio, los repuestos exactos (ubicados en listas o planos de partes), las herramientas especiales y los tiempos estimados. La demora o el error en la localización de estos datos no solo consume horas-hombre valiosas de ingeniería, sino que incrementa el riesgo de errores en la ejecución ("mantenimiento incorrecto") o retrasos en la intervención. En una planta concentradora, donde la disponibilidad de los equipos es crítica, esta latencia en el acceso a la información puede traducirse en paradas de planta prolongadas y pérdidas significativas en la producción de cobre.

== Diagnóstico
En el área de planificación y mantenimiento de las plantas concentradoras de cobre, se observa una gestión documental dispersa y poco funcional. Actualmente, los manuales de equipos críticos y no críticos (como molinos SAG, chancadoras, bombas y celdas de flotación) se almacenan en repositorios digitales masivos (SharePoint, servidores locales) sin una indexación semántica adecuada, el almacenamiento actual esta basado en como se adquirieron en la etapa de proyecto y no por equipos, procesos o clase de equipo, flota o grupos de equipos (Ejemplo: Todas las bombas de la planta están agrupadas en una sola carpeta ya que todas las bombas de lodo fueron compradas a un mismo proveedor, caso similar con las celdas de flotación todas están en una sola carpeta y con un solo pdf por todos los tipos de celdas).\

Se evidencia que, ante una falla o una parada programada, los planificadores/programadores de mantenimiento *invierten un tiempo excesivo navegando manualmente* entre carpetas y archivos PDF extensos —algunos de los cuales son documentos escaneados ("imágenes de texto")— lo que impide el uso de herramientas de búsqueda convencionales (Ctrl+F). Un síntoma recurrente es la dificultad para correlacionar la información visual con la textual; por ejemplo, el técnico encuentra el procedimiento de desmontaje en la página 50, pero el plano de despiece con los códigos de repuestos está en un anexo al final del documento o en un archivo separado, obligando a una validación manual cruzada propensa a errores. Además, la existencia de manuales en inglés técnico complejo genera barreras de comprensión inmediata por parte del personal operativo, retrasando la ejecución de las órdenes de trabajo.\

Se evidencia que los técnicos no realizan la búsqueda de especificaciones técnicas en manuales, generando que 6 de cada 10 condiciones de equipos reportados carecen de información suficiente para la gestión de planificación del mantenimiento, especialmente en la identificación de los repuestos a cambiar.\


== Identificación y Diagnóstico del Problema de Estudio

El problema central identificado no es la inexistencia de información, sino la incapacidad de los sistemas de búsqueda actuales para procesar y relacionar información multimodal (texto e imagen) contenida en documentos técnicos no estructurados.\

Las técnicas de búsqueda tradicionales (basadas por palabras clave, lectura secuencial o búsqueda por tabla de contenidos) resultan insuficientes para interpretar consultas complejas de mantenimiento que requieren contexto, como "procedimiento de cambio de liner considerando el torque especificado en el plano A". Existe una brecha tecnológica entre la naturaleza heterogénea de los manuales IOM (que combinan diagramas, tablas de especificaciones y narrativas técnicas) y los mecanismos de recuperación de información disponibles, los cuales tratan el texto y la imagen como entidades desconectadas.\

Esta limitación tecnológica deriva en una baja precisión y retrabajo  en las consultas técnicas, lo que impacta negativamente en el tiempo medio de reparación (MTTR) y en la confiabilidad de la planificación de mantenimiento. Por lo tanto, el problema de estudio se define como la ineficiencia en la recuperación de información técnica contextualizada debido a la falta de integración semántica entre los datos textuales y visuales en los repositorios de mantenimiento de plantas mineras.\

=== Antecedentes bibliográficos

=== Formulación del Problema

==== Formulación del Problema General
¿Puede la implementación de un Sistema de Consultas Multimodal basada en Arquitectura Rag y fine tuning (LoRA) reducir el tiempo de busqueda de información técnica de manuales de de mantenimiento de equipos críticos de una planta concentradora de cobre?\
¿De qué manera la implementación de un Sistema de Consultas Multimodal basado en Arquitectura RAG (Retrieval-Augmented Generation) reduce el tiempo de busqueda de información técnica en los manuales de mantenimiento de plantas concentradoras de cobre, en comparación con los métodos de búsqueda tradicionales?

==== Formulación de los Problemas Específicos
+ ¿Cómo influye la integración semántica de datos multimodales (texto, planos, diagramas e imágenes) en la capacidad del sistema para responder consultas técnicas contextuales que requieren interpretación visual, a diferencia de la búsqueda puramente textual?

+ ¿Cuál es la mejora en la precisión y exhaustividad (recall) de la información recuperada al utilizar técnicas de embedding y re-ranking vectorial frente a la búsqueda léxica (palabras clave) en manuales con terminología heterogénea?

+ ¿En qué medida se reduce el tiempo de búsqueda de información crítica (procedimientos, especificaciones de repuestos y herramientas) para la planificación de mantenimiento al utilizar el asistente conversacional basado en RAG?

=== Justificación y Alcances
==== Justificación
La presenta investigación se justifica ante la creciente complejidad de la documentación técnica en entornos industriales y de ingeniería, donde la información crítica reside en manuales técnicos con información heterogénea (texto, imagenes, planos, etc.) esta información crítica escencial para toma de decisiones toma tiempo largo de extraer con metodos convencionales y los sistemas RAG(Retrieval-Augmented Generation) multimodales tienen la capacidad de entender y responder de forma rapida reduciendo esta brecha de tiempo entre la información y la toma de decisiones o acciones realies en la empresa industrial.
Desde una perspectiva técnica y operativa, el proyecto contribuye en el tiempo de busqueda de información lo cual genera un incremento positivo en el % wrench time del equipo de mantenimiento como lo describe @palmer_maintenance_2019 en su libro el wrench time(tiempo efectivo de mantenimiento) representa en el mejor de los casos del 55% del tiempo total disponible por el equipo técnico de mantenimiento, donde el 45% de tiempo restante corresponde a actividades de traslado, demoras entre sub procesos y *busqueda de información técnica* para ejecutar la actividad de mantenimiento, por lo que el proyecto va a contribuir en la eficiencia y utilización de las HHs del personal técnico de mantenimiento.
Desde una perspectiva economica la reducción de tiempo de busqueda de información va a ayudar en una rapida y mejor toma de decisiones lo cual contribuye en una respuesta mas rapida ante emergencias contribuyendo positivamente en la disponibilidad global de las plantas concentradora y con ello la producción asociada.

==== Alcances
El proyecto pretende generar un sistema RAG multimodal y Fine-Tuning (LoRa) para manuales de mantenimiento en planta concentradora con manuales facilitados por el stakeholder el cual pretende tener un anonimato y que la información tenga un estandar de confidencialidad, por lo que el stakeholder va a brindar los manuales que se requieren para este proyecto:  
 Limitaciones:
 - Solo se van a procesar PDF como tipo de información (ingesta de datos).
 - Base de conocimiento (corpus) es estricta a los manuales facilidados por el stakehokder.
 - El sistema no pretende generar nuevos diagramas, planos o imagenes.
 - Base de datos y sistemas va a ser proporsionada por el stakeholder solo en la etapa de producción, la etapa de prototipo se va a realizar un proceso interno de validación de financiamiento.

== Objetivo General
Desarrollar un Sistema de Consultas Multimodal basado en arquitectura RAG (Retrieval-Augmented Generation) para optimizar la eficiencia y precisión en la recuperación de información técnica de los manuales de mantenimiento en plantas concentradoras de cobre.
== Objetivo Especifico

Estos objetivos representan los pasos técnicos y validaciones necesarias para alcanzar el objetivo general. Están alineados 1 a 1 con tus problemas específicos:

+ Diseñar e implementar un pipeline de procesamiento de datos multimodal que permita la extracción, vectorización e indexación conjunta de texto no estructurado y esquemas visuales (planos de partes, diagramas procedimientos, imágenes) contenidos en los manuales de mantenimiento.

+ Evaluar el desempeño del motor de recuperación mediante métricas de relevancia (Precision y Recall) BERTscore y ROUGE L(Recall Oriented Understudy for Gisting Evaluation).

+ Validar la utilidad del sistema en un entorno operativo, cuantificando la reducción del tiempo empleado por los planificadores en la búsqueda de información crítica y atención de reportes de condición.
