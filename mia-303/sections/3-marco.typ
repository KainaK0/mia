= Marco Teórico y Estado del Arte
== Bases Teóricas
=== Fundamentos de IA:
El desarrollo de LLMs en el mundo de creación de sistemas capaces de razonar y responder consultas ha ido en constante desarrollo, actualmente se tiene fundamentos donde se desarrollaron la capacidad de incrementar la información de LLMs con el fin de adaptar los estos a nuevos contextos en especifico y con el fin de este proyecto responder a consultas sobre docuemntos técnicos de mantenimiento para ellos surgen conceptos claves como son los siguientes:  

+ *Grandes Modelos de Lenguaje (LLMs)*\
	Los Grandes Modelos de Lenguaje (LLMs), especialmente las variantes basadas en la arquitectura Transformer, son sistemas de inteligencia artificial que han logrado un éxito extraordinario en diversas tareas lingüísticas. Estos modelos se someten a un pre-entrenamiento con conjuntos de datos de instrucciones extensos y de alta calidad, lo que les permite aprender una amplia gama de patrones lingüísticos, estructuras y conocimientos factuales. Gracias a esto, los LLMs pueden generar texto similar al humano con altos grados de fluidez y coherencia, demostrando una gran capacidad para comprender y responder a una variedad de consultas. Sin embargo, presentan limitaciones significativas, como la generación de respuestas especulativas o fabricadas (alucinaciones) cuando se enfrentan a dominios altamente especializados o información que requiere actualizaciones en tiempo real, ya que su conocimiento es estático y limitado a sus datos de entrenamiento @yin_survey_2024

+ *Generación Aumentada por Recuperación (RAG)*\
	La Generación Aumentada por Recuperación (RAG) surge como una solución efectiva para mitigar las alucinaciones de los LLMs, mejorando sus capacidades de generación mediante la recuperación de conocimiento externo relevante. Este sistema opera típicamente en un proceso de dos pasos: recuperación y generación. En la etapa de recuperación, el sistema localiza rápidamente conocimiento semánticamente similar a la consulta del usuario dentro de una colección de documentos a gran escala. Posteriormente, en la etapa de generación, los fragmentos de documentos recuperados se combinan con la consulta original para formar una entrada aumentada, proporcionando al LLM un contexto rico en información externa. Esto permite que el modelo fundamente sus respuestas en información factual y actualizada dinámicamente, mejorando la precisión y reduciendo la incidencia de respuestas inventadas @gao_retrieval-augmented_2024.
+  *RAG Multimodal (mRAG)*\
	El RAG Multimodal (mRAG) representa una evolución significativa del marco RAG tradicional, extendiendo sus capacidades para procesar y gestionar datos de diversas modalidades, como texto, imágenes, audio y video. A diferencia de los sistemas tradicionales limitados a texto plano, el mRAG integra datos multimodales tanto en los procesos de recuperación como en los de generación, permitiendo respuestas más completas y contextualmente relevantes. En este enfoque, la recuperación implica localizar e integrar conocimientos de diferentes tipos de datos, mientras que la generación utiliza Grandes Modelos de Lenguaje Multimodales (MLLMs) para producir respuestas que incorporan información de múltiples modalidades. Se ha demostrado que el mRAG supera al RAG de solo texto, especialmente en escenarios donde la información visual y textual es crítica para comprender y responder consultas @gao_retrieval-augmented_2024 @yin_survey_2024
-  *Low-Rank Adaptation (LoRA)*\
	Aunque el documento se centra en mRAG, menciona la Adaptación de Bajo Rango (LoRA) en el contexto de la eficiencia y el ajuste fino de modelos. LoRA se presenta como una técnica utilizada para la adaptación eficiente de modelos grandes, permitiendo el ajuste fino de LLMs para tareas específicas como la compresión de contextos o la codificación, reduciendo la carga computacional. También se emplea en arquitecturas como "mezcla de expertos" (Mixture of Experts) para mitigar conflictos de datos durante el ajuste de instrucciones en MLLMs. En esencia, permite adaptar modelos masivos a nuevas tareas o dominios sin la necesidad de re-entrenar todos sus parámetros, facilitando la eficiencia en la inferencia y el entrenamiento @hu_lora_2021
 - *Agentes basados en LLM*\
	Los agentes basados en LLM son sistemas avanzados que utilizan modelos de lenguaje de gran tamaño como núcleo para planificar y ejecutar tareas complejas. Estos agentes pueden descomponer preguntas multimodales complicadas en cadenas de sub-preguntas y acciones de recuperación, adaptando sus siguientes pasos basándose en el estado de la resolución del problema y el contenido recuperado. Un ejemplo destacado es HuggingGPT, un agente impulsado por LLM que conecta varios modelos de IA para resolver tareas sofisticadas; utiliza ChatGPT para realizar la planificación de tareas al recibir una solicitud del usuario, selecciona los modelos apropiados y resume la respuesta según los resultados de la ejecución. Estos agentes permiten abordar una amplia gama de tareas de IA que abarcan diferentes modalidades y dominios.
-  *Multimodal Document Parsing and Indexing*\
	Este concepto se refiere al proceso fundamental de procesar documentos multimodales (como PDFs, HTMLs o diapositivas) para hacerlos buscables. El objetivo es analizar y estructurar elementos como texto, imágenes, tablas y videos provenientes de documentos no estructurados o semi-estructurados. Existen dos enfoques principales:
	1. *Basado en extracción:* Extrae información multimodal y la convierte en descripciones textuales o las procesa por separado (por ejemplo, usando OCR para texto y modelos de _captioning_ para imágenes).
	2. *Basado en representación:* Utiliza capturas de pantalla de los documentos directamente para la indexación, preservando el diseño visual y la estructura original para evitar la pérdida de información durante la extracción. El resultado final es la creación de un índice vectorial que almacena representaciones de estos datos para su recuperación posterior.
+ *Multimodal Retrieval*\
	La recuperación multimodal es el componente encargado de identificar y obtener documentos o fragmentos de información relevantes desde una base de conocimiento externa, utilizando consultas que pueden combinar diferentes modalidades. Este proceso supera la búsqueda de texto simple al permitir búsquedas cruzadas, como encontrar una imagen relevante usando una consulta de texto (text-to-image) o viceversa. La tecnología detrás de esto incluye el uso de "retrievers" (recuperadores) que codifican datos en espacios vectoriales compartidos para medir similitud, y componentes de "reranking" (re-clasificación) que refinan el orden de los resultados basándose en interacciones más profundas entre la consulta y los documentos multimodales recuperados.
- *Multimodal Search Planning*\
	La planificación de búsqueda multimodal se refiere a las estrategias inteligentes empleadas por los sistemas mRAG para gestionar consultas complejas que requieren información de múltiples fuentes o modalidades. En lugar de seguir una tubería fija, los sistemas avanzados utilizan una planificación adaptativa que descompone una consulta compleja (por ejemplo, una pregunta que requiere razonamiento visual y textual) en sub-tareas. Este módulo decide dinámicamente qué tipo de recuperación realizar (por ejemplo, si buscar una imagen o un texto) y puede reformular la consulta original para mejorar la precisión de la búsqueda, integrando pistas visuales y textuales. Su objetivo es optimizar la adquisición de información, minimizando búsquedas innecesarias y maximizando la relevancia del contenido recuperado.
- *Multimodal Generation*\
	La generación multimodal es la fase final donde el sistema sintetiza una respuesta coherente integrando la consulta del usuario y la información recuperada, abarcando múltiples modalidades. Gracias a los MLLMs, este proceso no solo produce texto, sino que puede generar respuestas mixtas que entrelazan texto, imágenes, audio y video de manera fluida. Esto permite escenarios donde "una imagen vale más que mil palabras", respondiendo directamente con datos visuales, o escenarios donde la inclusión de medios multimodales mejora la precisión y riqueza de una explicación textual (como en guías paso a paso). El sistema debe identificar inteligentemente dónde insertar estos elementos multimodales dentro de la narrativa para asegurar la coherencia y mejorar la experiencia del usuario.
+ *Métricas y evaluación: ROUGE-L*\
  Es una métrica de comparación entre un texto candidato y una referencia que se basa en la subsecuencia común más larga para estimar cobertura y respeto del orden relativo de las palabras. La idea central es medir cuánto del contenido de la referencia aparece en el candidato manteniendo la secuencia de aparición, aunque no sea de forma contigua. A partir de la longitud de esa subsecuencia se obtienen medidas de cobertura sobre la referencia, precisión sobre el candidato y una combinación de ambas. ROUGE-L se usa ampliamente en tareas de resumen y en evaluación de generación porque captura de forma simple la presencia y el orden de unidades léxicas relevantes, y se puede agregar a nivel de documento o corpus manteniendo un procedimiento de cálculo transparente.
- *Metodologías: DSR, MLOps*\




== Definición de términos
- Glosario de términos y de abreviaturas o siglas
+ Glosario Técnico:
+ Terminología de Inteligencia Artificial y RAG:
  - Embedding (Incrustación Vectorial): Representación matemática de datos (texto o imagen) como vectores en un espacio multidimensional continuo. La proximidad entre vectores indica similitud semántica.
  - Fine-Tuning (Ajuste Fino): Proceso de entrenamiento adicional de un modelo pre-entrenado (Foundation Model) con un conjunto de datos específico del dominio para especializar sus capacidades en una tarea concreta.
  - Hallucination (Alucinación): Fenómeno en el cual un modelo generativo produce contenido que es sintácticamente coherente y seguro, pero factualmente incorrecto o no fundamentado en los datos de entrada. 
  - Knowledge Graph (Grafo de Conocimiento): Estructura de datos que representa entidades (nodos) y sus relaciones (aristas) de manera explícita. En RAG avanzado (GraphRAG), se utiliza para capturar la conectividad entre equipos (ej. Bomba A -> alimenta a -> Tanque B) que los embeddings vectoriales pueden perder.   
  - LoRA (Low-Rank Adaptation): Técnica de PEFT (Parameter-Efficient Fine-Tuning) que permite adaptar LLMs gigantescos con recursos computacionales limitados, modificando solo matrices de bajo rango inyectadas en la red.
  - RAG (Retrieval-Augmented Generation): Paradigma arquitectónico que mejora la salida de un LLM al proporcionarle información externa recuperada en tiempo de ejecución, combinando la vastedad de conocimiento del modelo con la precisión de datos propietarios.
  - Vector Database (Base de Datos Vectorial): Sistema de almacenamiento optimizado para guardar y consultar vectores de alta dimensión (embeddings). Utiliza algoritmos de búsqueda de vecinos más cercanos aproximados (ANN) como HNSW para una recuperación ultrarrápida.

+ Métricas de Validación Experimental
  - BERTScore:
    A diferencia de las métricas tradicionales basadas en n-gramas (que buscan coincidencia exacta de palabras), BERTScore evalúa la similitud semántica utilizando embeddings contextuales.Fundamento: Calcula la similitud del coseno entre los embeddings de cada token en la respuesta generada ($x$) y los tokens en la respuesta de referencia ($y$), utilizando una alineación voraz (greedy matching) para maximizar la puntuación.4Relevancia Minera: Es crucial porque en minería existen múltiples formas de referirse a un mismo concepto (ej. "Liner", "Revestimiento", "Blindaje"). Una métrica exacta penalizaría estas variaciones, mientras que BERTScore captura su equivalencia semántica. Nam et al. reportaron una mejora de 3.0 puntos porcentuales en esta métrica usando su arquitectura.
    Formula:
      $ R_"BERT" = 1 / (|x|) sum_(x_i in x) max_(y_j in y) x_i^T y_j $

    
  - ROUGE-L(Recall-Oriented Understudy for Gisting Evaluation): 
  Se centra en la estructura y la secuencia, midiendo la subsecuencia común más larga (Longest Common Subsequence - LCS) entre la generación y la referencia.   
  
  Fundamento: Evalúa la capacidad del modelo para preservar el orden de las palabras y la estructura de la oración.

  Fórmula:
    $ F_"lcs" = ((1 + beta^2) R_"lcs" P_"lcs") / (R_"lcs" + beta^2 P_"lcs") $

  - Evaluación Cualitativa (Human-in-the-loop):
  Validación realizada por expertos del dominio (técnicos/ingenieros) mediante escalas Likert para medir satisfacción, claridad y utilidad.
- Principios de validación experimental y métricas clave.
- Marco normativo (leyes, normas técnicas, reglamentos)

+ *Taxonomía de Datos de Confiabilidad (ISO 14224)*
  Para que un Modelo de Lenguaje Grande (LLM) pueda razonar eficazmente sobre mantenimiento, debe "entender" la estructura jerárquica de los equipos industriales. La norma ISO 14224 proporciona el estándar de facto para la recolección e intercambio de datos de confiabilidad y mantenimiento. Aunque originada en la industria del petróleo y gas, su adopción en la minería es generalizada para estructurar los sistemas ERP (como SAP PM) y CMMS.   
  
  La norma establece una jerarquía taxonómica que permite descomponer un activo complejo en unidades manejables. Esta estructura es vital para el diseño de la base de datos vectorial del sistema RAG, ya que permite la creación de metadatos precisos para el filtrado de búsquedas (Metadata Filtering).
  
  #table(
    columns: (1.5fr, 2fr, 2fr),
    inset: 10pt,
    align: horizon,
    stroke: 0.5pt + gray,
    fill: (col, row) => if row == 0 { rgb("#2d5a27").lighten(90%) }, // Color suave para el encabezado
    
    [*Nivel Taxonómico (ISO 14224)*], 
    [*Ejemplo en Planta Concentradora*], 
    [*Aplicación en Sistema RAG*],
  
    [Nivel 3: Instalación], 
    [Planta Concentradora de Cobre], 
    [Contexto global del sistema.],
  
    [Nivel 4: Sistema], 
    [Circuito de Molienda SAG], 
    [Delimitación del alcance de la consulta.],
  
    [Nivel 5: Sub-sistema], 
    [Sistema de Lubricación de Chumaceras], 
    [Agrupación funcional de documentos.],
  
    [Nivel 6: Equipo (Unit)], 
    [Unidad de Bombeo de Alta Presión], 
    [Entidad principal de consulta (Subject).],
  
    [Nivel 7: Sub-unidad], 
    [Bomba de desplazamiento positivo], 
    [Componente específico de falla.],
  
    [Nivel 8: Pieza (Part)], 
    [Sello Mecánico / Empaquetadura], 
    [Objeto de la instrucción de recambio.],
  
    [Nivel 9: Ítem Mantenible], 
    [Anillo tórico (O-ring)], 
    [Detalle granular para repuestos.],
  )


== Estado del Arte

+ Taxonomía de métodos en IA aplicada al Mantenimiento y Documentación
  La literatura actual (2023-2025) permite clasificar las soluciones de IA para gestión de conocimiento técnico en tres generaciones evolutivas :
  
  - Sistemas de Recuperación Basados en Palabras Clave (Lexical Search) :
  
  - Método: Utilizan algoritmos como TF-IDF o BM25.
  
  - Aplicación: Motores de búsqueda tradicionales en gestores documentales (DMS).
  
  - Limitación: No capturan el contexto semántico; fallan ante sinónimos o consultas naturales complejas. 

+ Sistemas RAG Naive (Ingenuos) Unimodales :

  - Método: Indexación vectorial de texto plano + recuperación semántica + generación con LLM genérico.
  
  - Aplicación: Chatbots de primera generación para manuales simples.
  
  - Limitación: Sufren del problema de "Lost in the Middle", alucinaciones frecuentes y ceguera ante imágenes/tablas.
+ Sistemas RAG Modulares y Multimodales (MM-RAG) - Enfoque de la Tesis :
  
  - Método: Integran módulos de re-ranking, grafos de conocimiento (GraphRAG) y procesamiento de visión (Vision Encoders) para ingerir texto, imágenes y diagramas conjuntamente.
  
  - Tecnologías: Modelos de Embeddings densos (ej. BAAI-bge-m3), Bases de Datos Vectoriales, y adaptación de dominio vía PEFT/LoRA.
  
  - Aplicación: Estado del arte para documentación técnica compleja (automotriz, aeroespacial, y ahora propuesto para minería). 

- Revisión comparativa: fortalezas y debilidades.

- Vacíos y oportunidades de investigación.
