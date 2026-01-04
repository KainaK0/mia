= Marco Teórico y Estado del Arte
== Bases Teóricas
=== Fundamentos de IA:
El desarrollo de LLMs en el mundo de creación de sistemas capaces de razonar y responder consultas ha ido en constante desarrollo, actualmente se tiene fundamentos donde se desarrollaron la capacidad de incrementar la información de LLMs con el fin de adaptar los estos a nuevos contextos en especifico y con el fin de este proyecto responder a consultas sobre docuemntos técnicos de mantenimiento para ellos surgen conceptos claves como son los siguientes:  

+ *Grandes Modelos de Lenguaje (LLMs)*\
  Los Grandes Modelos de Lenguaje (LLMs) representan un cambio de paradigma en el procesamiento del lenguaje natural, caracterizados por su capacidad de aprendizaje few-shot (pocos ejemplos) sin necesidad de actualizaciones de gradiente para tareas específicas. @brown_language_2020 demuestran que al escalar el tamaño del modelo y los datos de entrenamiento, los modelos desarrollan la capacidad de adaptarse a nuevas tareas simplemente a través de instrucciones textuales o demostraciones en el contexto (in-context learning). A pesar de su capacidad para almacenar conocimiento factual en sus parámetros, los LLMs enfrentan desafíos significativos con el conocimiento de "larga cola" (información que aparece raramente en los datos de entrenamiento). @kandpal_large_2023 establecen una relación causal y correlacional entre la capacidad de un modelo para responder preguntas factuales y el número de documentos relevantes vistos durante el pre-entrenamiento, indicando que incluso modelos masivos luchan por retener información de baja frecuencia.

+ *Generación Aumentada por Recuperación (RAG)*\
	La Generación Aumentada por Recuperación (RAG) combina una memoria paramétrica (el modelo seq2seq pre-entrenado) con una memoria no paramétrica (un índice vectorial denso de documentos, como Wikipedia) @lewis_retrieval-augmented_2021. Este enfoque mitiga las alucinaciones y permite la actualización del conocimiento sin reentrenar el modelo base. @gao_retrieval-augmented_2024 categorizan la evolución de RAG en tres paradigmas:

  - Naive RAG: El proceso tradicional de indexación, recuperación y generación.

  - Advanced RAG: Introduce estrategias de pre-recuperación y post-recuperación para mejorar la calidad de los documentos seleccionados.

  - Modular RAG: Ofrece mayor adaptabilidad mediante la incorporación de módulos especializados y la reconfiguración del flujo de trabajo.

  Además, @cheng_ragtrace_2025 destacan la importancia de comprender la dinámica interna entre la recuperación y la generación, proponiendo métodos para rastrear y diagnosticar errores en estos flujos de trabajo opacos.

+  *RAG Multimodal (mRAG)*\
	El RAG Multimodal (mRAG) extiende el marco RAG para integrar datos multimodales (texto, imágenes, video) tanto en los procesos de recuperación como de generación @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025 . A diferencia del RAG tradicional basado solo en texto, mRAG aborda la limitación de aprovechar información rica contenida en formatos no textuales. La evolución de mRAG desde sistemas que convierten datos multimodales a texto (pseudo-MRAG) hasta sistemas end-to-end que preservan los datos multimodales originales y utilizan Modelos de Lenguaje Grande Multimodales (MLLMs) para la generación.

+  *Low-Rank Adaptation (LoRA)*\
	La Adaptación de Bajo Rango (LoRA) en el contexto de la eficiencia y el ajuste fino de modelos. LoRA se presenta como una técnica utilizada para la adaptación eficiente de modelos grandes, permitiendo el ajuste fino de LLMs para tareas específicas como la compresión de contextos o la codificación, reduciendo la carga computacional. También se emplea en arquitecturas como "mezcla de expertos" (Mixture of Experts) para mitigar conflictos de datos durante el ajuste de instrucciones en MLLMs. En esencia, permite adaptar modelos masivos a nuevas tareas o dominios sin la necesidad de re-entrenar todos sus parámetros, facilitando la eficiencia en la inferencia y el entrenamiento @hu_lora_2021
+ *Agentes basados en LLM*\
	Los agentes basados en LLM representan una evolución hacia sistemas autónomos capaces de razonamiento y uso de herramientas. @zhang_mm-llms_2024 clasifican a los MLLMs en variantes de "Uso de Herramientas" (Tool-using), donde el LLM actúa como un controlador que invoca herramientas externas (como expertos en visión o API de búsqueda) para realizar tareas multimodales, en lugar de realizar todo el procesamiento end-to-end. @gao_retrieval-augmented_2024 discuten el papel de los agentes dentro del paradigma "Modular RAG", donde módulos funcionales como búsqueda, memoria y predicción son orquestados dinámicamente para resolver problemas complejos, permitiendo flujos de trabajo iterativos y adaptativos en lugar de lineales.

+  *Multimodal Document Parsing and Indexing*\
	Este concepto se refiere al proceso fundamental de procesar documentos multimodales (como PDFs, HTMLs o diapositivas) para hacerlos buscables. El objetivo es analizar y estructurar elementos como texto, imágenes, tablas y videos provenientes de documentos no estructurados o semi-estructurados. Existen dos enfoques principales:
	1. *Basado en extracción:* Extrae información multimodal y la convierte en descripciones textuales o las procesa por separado (por ejemplo, usando OCR para texto y modelos de _captioning_ para imágenes).
	2. *Basado en representación:* Utiliza capturas de pantalla de los documentos directamente para la indexación, preservando el diseño visual y la estructura original para evitar la pérdida de información durante la extracción. El resultado final es la creación de un índice vectorial que almacena representaciones de estos datos para su recuperación posterior.
+ *Multimodal Retrieval*\
	La recuperación multimodal es el componente encargado de identificar y obtener documentos o fragmentos de información relevantes desde una base de conocimiento externa, utilizando consultas que pueden combinar diferentes modalidades. Este proceso supera la búsqueda de texto simple al permitir búsquedas cruzadas, como encontrar una imagen relevante usando una consulta de texto (text-to-image) o viceversa. La tecnología detrás de esto incluye el uso de "retrievers" (recuperadores) que codifican datos en espacios vectoriales compartidos para medir similitud, y componentes de "reranking" (re-clasificación) que refinan el orden de los resultados basándose en interacciones más profundas entre la consulta y los documentos multimodales recuperados.
+ *Multimodal Search Planning*\
	La planificación de búsqueda multimodal se refiere a las estrategias inteligentes empleadas por los sistemas mRAG para gestionar consultas complejas que requieren información de múltiples fuentes o modalidades. En lugar de seguir una tubería fija, los sistemas avanzados utilizan una planificación adaptativa que descompone una consulta compleja (por ejemplo, una pregunta que requiere razonamiento visual y textual) en sub-tareas. Este módulo decide dinámicamente qué tipo de recuperación realizar (por ejemplo, si buscar una imagen o un texto) y puede reformular la consulta original para mejorar la precisión de la búsqueda, integrando pistas visuales y textuales. Su objetivo es optimizar la adquisición de información, minimizando búsquedas innecesarias y maximizando la relevancia del contenido recuperado.
+ *Multimodal Generation*\
	La generación multimodal es la fase final donde el sistema sintetiza una respuesta coherente integrando la consulta del usuario y la información recuperada, abarcando múltiples modalidades. Gracias a los MLLMs, este proceso no solo produce texto, sino que puede generar respuestas mixtas que entrelazan texto, imágenes, audio y video de manera fluida. Esto permite escenarios donde "una imagen vale más que mil palabras", respondiendo directamente con datos visuales, o escenarios donde la inclusión de medios multimodales mejora la precisión y riqueza de una explicación textual (como en guías paso a paso). El sistema debe identificar inteligentemente dónde insertar estos elementos multimodales dentro de la narrativa para asegurar la coherencia y mejorar la experiencia del usuario.

+ *Métricas y evaluación:* 
- *ROUGE-L(Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)*\
  ROUGE es un marco de métricas ampliamente utilizado para evaluar tareas de generación de texto y resumen automático. Aunque la familia ROUGE incluye variantes como ROUGE-N (que mide la superposición de N-gramas), ROUGE-L se utiliza específicamente para evaluar la calidad de la generación basándose en la subsecuencia común más larga entre el texto generado y la referencia. Este enfoque evalúa qué tan bien el texto generado captura el contenido esencial de la referencia, con un fuerte énfasis en la recuperación (recall) de la información @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025.

- *BERTScore*
  Definición y Fundamento: BERTScore es una métrica de evaluación que utiliza incrustaciones contextuales (contextual embeddings) provenientes de modelos pre-entrenados como BERT para medir la similitud semántica entre el texto generado y el texto de referencia. A diferencia de las métricas basadas en n-gramas (como ROUGE o BLEU) que dependen de coincidencias exactas de palabras, BERTScore alinea los tokens basándose en sus representaciones vectoriales, lo que le permite capturar relaciones semánticas más profundas y matices de significado que no son evidentes en la superficie léxica @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025.

=== Metodologías de investigación tecnológicas

+ *Agile+IA* \

  La metodología Agile+IA no es una metodología rígida única, sino la adaptación de los principios del Manifiesto Ágil (comúnmente usando marcos como Scrum o Kanban) a la naturaleza experimental e incierta de los proyectos de Inteligencia Artificial.
  A diferencia del desarrollo de software tradicional (que es determinista), la IA es probabilística y requiere mucha experimentación, lo que obliga a adaptar el enfoque ágil de la siguiente manera: 
  - Principio 1: Entrega Temprana de Valor 
    En lugar de esperar meses para tener un "modelo perfecto", el objetivo es reducir el Time-to-Market mediante incrementos funcionales.

  - Principio 2: Feedback Continuo de Stakeholders
    La validación en IA es más compleja que en software tradicional porque los modelos pueden fallar de formas impredecibles (alucinaciones, sesgos).

  - Principio 3: Priorizar Desempeño y Calidad
    El Backlog del producto debe tratar los requisitos no funcionales (rendimiento del modelo) con la misma urgencia que las nuevas funcionalidades.

  - Principio 4: Colaboración Multidisciplinaria
    Se incentiva la participación de varias tipos de diciplinas Data scientist, dev, DevOps y experto de dominio trabajan codo a codo en cada iteración.


+ *CRISP-DM* 
  Es el estándar más utilizado en la industria para proyectos de minería de datos y ciencia de datos. A diferencia de Agile (que es una metodología de gestión de proyectos), CRISP-DM es un modelo de proceso específico para el ciclo de vida de los datos.
  Se estructura en 6 fases secuenciales pero iterativas, lo que significa que es común retroceder entre fases (por ejemplo, volver a la preparación de datos tras un modelado fallido).
  Las 6 Fases de CRISP-DM:

  + Comprensión del Negocio (Business Understanding):\   Es la fase crítica donde se definen los objetivos del proyecto desde la perspectiva empresarial y se traducen en problemas técnicos de minería de datos. Se establecen los criterios de éxito.
  + Comprensión de los Datos (Data Understanding):\   Implica la recolección inicial, descripción y exploración de los datos. Se busca identificar problemas de calidad (datos faltantes, errores) y descubrir primeros patrones o hipótesis.
  + Preparación de los Datos (Data Preparation):\   Generalmente es la fase que consume más tiempo (60-80% del proyecto). Incluye la limpieza, selección, integración y transformación de datos (Feature Engineering) para crear el conjunto final que se usará en el modelado.
  + Modelado (Modeling):\ Se seleccionan y aplican las técnicas de modelado (algoritmos de IA/ML). Se calibran los parámetros para optimizar resultados. A menudo se requiere volver a la fase de preparación para ajustar los datos a las necesidades del modelo específico.
  + Evaluación (Evaluation):\   No solo se evalúa la precisión técnica del modelo, sino su eficacia para resolver el problema de negocio planteado en la fase 1. Se decide si el modelo es apto para ser desplegado o si requiere revisión.
  + Despliegue / Implementación (Deployment):\   El conocimiento obtenido se presenta de forma útil para el usuario final. Puede ir desde generar un reporte simple hasta la implementación de un modelo predictivo en tiempo real integrado en una aplicación de software. Incluye planes de monitoreo y mantenimiento.



== Definición de términos
+ Glosario de términos y de abreviaturas o siglas
  - Embedding (Incrustación Vectorial): Representación matemática de datos (texto o imagen) como vectores en un espacio multidimensional continuo. La proximidad entre vectores indica similitud semántica.
  - Fine-Tuning (Ajuste Fino): Proceso de entrenamiento adicional de un modelo pre-entrenado (Foundation Model) con un conjunto de datos específico del dominio para especializar sus capacidades en una tarea concreta.
  - Hallucination (Alucinación): Fenómeno en el cual un modelo generativo produce contenido que es sintácticamente coherente y seguro, pero factualmente incorrecto o no fundamentado en los datos de entrada. 
  - Knowledge Graph (Grafo de Conocimiento): Estructura de datos que representa entidades (nodos) y sus relaciones (aristas) de manera explícita. En RAG avanzado (GraphRAG), se utiliza para capturar la conectividad entre equipos (ej. Bomba A -> alimenta a -> Tanque B) que los embeddings vectoriales pueden perder.   
  - LoRA (Low-Rank Adaptation): Técnica de PEFT (Parameter-Efficient Fine-Tuning) que permite adaptar LLMs gigantescos con recursos computacionales limitados, modificando solo matrices de bajo rango inyectadas en la red.
  - RAG (Retrieval-Augmented Generation): Paradigma arquitectónico que mejora la salida de un LLM al proporcionarle información externa recuperada en tiempo de ejecución, combinando la vastedad de conocimiento del modelo con la precisión de datos propietarios.
  - Vector Database (Base de Datos Vectorial): Sistema de almacenamiento optimizado para guardar y consultar vectores de alta dimensión (embeddings). Utiliza algoritmos de búsqueda de vecinos más cercanos aproximados (ANN) como HNSW para una recuperación ultrarrápida.
  - IOM (Manuales de Instalación, Operación y Mantenimiento): Es el documento técnico rector proporcionado por el fabricante (OEM) o desarrollado internamente, que contiene las especificaciones, procedimientos e intervalos necesarios para conservar la función de un activo.
  - Búsqueda de Información Secuencial: Es el método de recuperación de información que sigue un orden lineal o cronológico preestablecido en el documento.
  - Búsqueda de Información por Palabra Clave: Método de recuperación basado en la coincidencia léxica exacta (keyword matching) de términos específicos dentro de un índice o documento digital.
  - Búsqueda por Tabla de Contenido: Método de navegación jerárquica que utiliza la estructura lógica del documento (Capítulos, Secciones, Subsecciones) para localizar información.
  - Despiece de Partes de Equipos: Es una representación técnica gráfica (plano o diagrama) que muestra los componentes de un ensamblaje ligeramente separados por una distancia, indicando su orden de montaje y relación espacial.
  - Procedimientos de Actividad de Mantenimiento de Equipo: Son documentos instruccionales estandarizados que describen paso a paso cómo ejecutar una tarea específica de mantenimiento para asegurar consistencia, seguridad y calidad.
  - Documento Técnico: Término paraguas que engloba toda la información escrita o gráfica que describe la funcionalidad, arquitectura y manejo de un producto técnico o sistema.
  - Sistema Pregunta-Respuesta (Question-Answering System): Es una aplicación de Inteligencia Artificial diseñada para responder automáticamente a preguntas formuladas por humanos en lenguaje natural.
  - Maintenance Domain (Dominio de mantenimiento): Se refiere al campo de conocimiento especializado y al contexto operativo relacionado con la gestión, preservación y restauración de activos físicos industriales. Se distingue por un vocabulario técnico altamente específico (ej. "cavitación", "holgura", "termografía"), una baja tolerancia al error (por riesgos de seguridad) y una gran dependencia de datos multimodales (sonidos de vibración, imágenes de desgaste, manuales PDF).

+ Principios de validación experimental y métricas clave
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




== Estado del Arte

+ Taxonomía de métodos en IA aplicada
Hacia la Inteligencia Industrial: Evolución del RAG Multimodal y el Ajuste Fino en el Mantenimiento
La gestión del mantenimiento en plantas concentradoras enfrenta un desafío endémico: la desconexión entre el volumen masivo de documentación técnica (manuales, diagramas P&ID, reportes de fallas) y la necesidad de respuestas precisas en tiempo real. La literatura reciente narra una evolución tecnológica que va más allá de la simple búsqueda de texto, integrando la visión por computadora y el razonamiento estructurado para resolver este problema.

1. El Primer Obstáculo: La Precisión en la Recuperación Técnica
La historia comienza con una limitación fundamental de los modelos de lenguaje generales: no entienden la jerga técnica profunda ni los códigos de equipos específicos. En el contexto de la documentación técnica industrial, confiar únicamente en la búsqueda vectorial (embeddings densos) a menudo falla.

Investigaciones recientes en el dominio de la producción automotriz alemana demostraron que los modelos de incrustación estándar luchan significativamente con textos técnicos especializados que no están en inglés. Para mitigar esto, se ha propuesto un enfoque de recuperación híbrida, que combina la búsqueda vectorial semántica con la búsqueda de texto completo (BM25). Este enfoque híbrido ha demostrado mejorar el rendimiento de recuperación en un 20% en documentos técnicos, superando las deficiencias de los modelos que solo usan vectores. De manera similar, en el dominio médico, se ha validado que la fusión adaptativa de recuperadores dispersos y densos es crucial para mejorar la precisión del contexto recuperado.
+3

2. Estructurando el Conocimiento: De Texto Plano a Grafos de Fallas
Una vez que el sistema puede "encontrar" las palabras clave, el siguiente desafío es "entender" la causalidad de una falla, algo vital para los activos críticos de una planta concentradora (como molinos o chancadoras).

El estudio de fallas en elevadores de minas (equipos críticos en minería subterránea) ha revelado que las relaciones simples son insuficientes para capturar la complejidad de los diagnósticos industriales. Para ello, se propone el uso de Grafos de Conocimiento Hiper-Relacionales, construidos con la ayuda de Grandes Modelos de Lenguaje (LLMs). A diferencia de los grafos tradicionales, estos capturan atributos multidimensionales de una falla (causa, impacto, acción correctiva) en una sola estructura lógica.
+2

Complementando esta visión estructural, la metodología Document GraphRAG sugiere que, en lugar de dividir los manuales en fragmentos arbitrarios, se debe utilizar la estructura intrínseca del documento (capítulos, secciones) para crear un grafo de conocimiento. Esto permite al sistema realizar un razonamiento de "múltiples saltos" (multi-hop), cruzando información de diferentes secciones del manual para responder preguntas complejas.
+1

3. La Dimensión Visual: RAG Multimodal y Fine-Tuning (LoRA)
Un técnico de mantenimiento en una planta no solo lee; interpreta diagramas, tablas de carga y videos de procedimientos. La narrativa actual del estado del arte marca un giro decisivo hacia la multimodalidad combinada con el ajuste fino (Fine-Tuning).

Un avance significativo se presenta en el desarrollo de sistemas para manuales de mantenimiento de vehículos complejos (Hyundai Staria), donde se implementó un sistema RAG multimodal ajustado mediante LoRA (Low-Rank Adaptation). Este sistema no solo procesa texto, sino que alinea semánticamente las imágenes de los manuales con las instrucciones textuales. El uso de LoRA permitió un ajuste fino eficiente de los parámetros del modelo (bLLossom-8B) y del modelo de incrustación (BAAI-bge-m3), logrando mejoras sustanciales en métricas como BERTScore y ROUGE-L.
+2

Llevando la multimodalidad al extremo, se ha propuesto la arquitectura mRAG (Multimodal RAG) unificada, capaz de procesar cuatro modalidades: texto, imagen, tabla y video. Utilizando modelos como Claude 3.5 Sonnet para enriquecer semánticamente las tablas y describir fotogramas clave de videos, este enfoque permite que un técnico reciba respuestas que sintetizan datos de un video tutorial y una tabla de especificaciones simultáneamente.
+1

4. Integración y Aprendizaje Continuo en el Flujo de Trabajo
Finalmente, la literatura aborda cómo estos sistemas aprenden y se gestionan dentro del ciclo de vida industrial.

Para gestionar los cambios constantes en la documentación técnica (como actualizaciones de firmware en sistemas de control), se han desarrollado sistemas como ReqRAG, que ayudan a los ingenieros a rastrear cambios en documentos de arquitectura y notas de lanzamiento. Esto es análogo a la gestión de cambios en los sistemas de control de una planta.

Además, para resolver la escasez de datos etiquetados en plantas industriales, se propone la metodología LU-RAG, que combina el aprendizaje en contexto (usando ejemplos etiquetados de reportes anteriores) con la recuperación de conocimiento externo no etiquetado (manuales). Este enfoque híbrido permite aprovechar lo "mejor de ambos mundos", mejorando la clasificación de textos y la verificación de hechos en entornos donde la data estructurada es limitada.
+1

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

#import "@preview/tdtr:0.4.3" : *
#tidy-tree-graph(compact: true, draw-node: horizontal-draw-node)[
  - Retrieval Augmented Generation (RAG)
    - Hybrid Retrieval
      - '@knollmeyer_hybrid_2025'
      - '@wang_retrieval_2025'
      - '@santra_curious_2025'
    - Grapth RAG
      - '@knollmeyer_document_2025'
      - '@shu_utilizing_2024'
    - Multimodal RAG
      - '@nam_lora-tuned_2025'
      - '@drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa'
    - In-Context Learning(ICL)/Prompting
      - '@ibtasham_reqrag_nodate'
      - '@santra_curious_2025'
    - Fine-Tuning
      - '@nam_lora-tuned_2025'
]


