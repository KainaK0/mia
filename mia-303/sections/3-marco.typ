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
	La recuperación multimodal es el componente encargado de identificar y obtener documentos o fragmentos de información relevantes desde una base de conocimiento externa, utilizando consultas que pueden combinar diferentes modalidades. Este proceso supera la búsqueda de texto simple al permitir búsquedas cruzadas, como encontrar una imagen relevante usando una consulta de texto (text-to-image) o viceversa. La tecnología detrás de esto incluye el uso de "retrievers" (recuperadores) que codifican datos en espacios vectoriales compartidos para medir similitud, y componentes de "reranking" (re-clasificación) que refinan el orden de los resultados basándose en interacciones más profundas entre la consulta y los documentos multimodales recuperados @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025.
+ *Multimodal Search Planning*\
	La planificación de búsqueda multimodal se refiere a las estrategias inteligentes empleadas por los sistemas mRAG para gestionar consultas complejas que requieren información de múltiples fuentes o modalidades. En lugar de seguir una tubería fija, los sistemas avanzados utilizan una planificación adaptativa que descompone una consulta compleja (por ejemplo, una pregunta que requiere razonamiento visual y textual) en sub-tareas. Este módulo decide dinámicamente qué tipo de recuperación realizar (por ejemplo, si buscar una imagen o un texto) y puede reformular la consulta original para mejorar la precisión de la búsqueda, integrando pistas visuales y textuales. Su objetivo es optimizar la adquisición de información, minimizando búsquedas innecesarias y maximizando la relevancia del contenido recuperado @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025.
+ *Multimodal Generation*\
	La generación multimodal es la fase final donde el sistema sintetiza una respuesta coherente integrando la consulta del usuario y la información recuperada, abarcando múltiples modalidades. Gracias a los MLLMs, este proceso no solo produce texto, sino que puede generar respuestas mixtas que entrelazan texto, imágenes, audio y video de manera fluida. Esto permite escenarios donde "una imagen vale más que mil palabras", respondiendo directamente con datos visuales, o escenarios donde la inclusión de medios multimodales mejora la precisión y riqueza de una explicación textual (como en guías paso a paso). El sistema debe identificar inteligentemente dónde insertar estos elementos multimodales dentro de la narrativa para asegurar la coherencia y mejorar la experiencia del usuario @mei.etalSurveyMultimodalRetrievalAugmentedGeneration2025.

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
=== Taxonomía de métodos en IA aplicada \

  La Generación Aumentada por Recuperación (RAG) ha emergido como un paradigma fundamental para mitigar las alucinaciones y limitaciones de conocimiento en los Grandes Modelos de Lenguaje (LLMs). La investigación reciente ha trascendido las implementaciones "ingenuas" (Naive RAG), avanzando hacia arquitecturas especializadas que integran datos multimodales, estructuras de grafos de conocimiento y estrategias de recuperación híbrida para satisfacer las demandas de precisión en sectores industriales, automotrices y médicos. \
  // Aplicaciones en Dominios Industriales y Técnicos
  La implementación de RAG en entornos industriales enfrenta desafíos únicos debido a la complejidad de la documentación técnica y la necesidad de precisión operativa.
  En el sector automotriz, @nam_lora-tuned_2025 desarrollaron un sistema RAG multimodal adaptado al dominio para manuales técnicos de vehículos (caso de estudio Hyundai Staria). Su enfoque utiliza el ajuste fino eficiente en parámetros (LoRA) sobre el modelo bLLossom-8B y embeddings BAAI-bge-m3, logrando mejoras significativas en métricas como BERTScore y ROUGE-L al integrar texto e imágenes para escenarios de resolución de problemas. Simultáneamente, @knollmeyer_hybrid_2025 abordaron la gestión de conocimientos en la planificación de producción (caso de estudio Audi). Identificaron que los modelos de embedding multilingües estándar tienen un rendimiento inferior en documentos técnicos en alemán en comparación con el inglés. Su solución propuesta es un enfoque de recuperación híbrida que combina búsqueda vectorial densa con búsqueda de texto completo, mejorando la precisión de recuperación en un 20% para documentos en alemán.

  En el ámbito de la ingeniería de software y ferroviaria, @ibtasham_reqrag_nodate propusieron "ReqRAG", un chatbot diseñado para la gestión de lanzamientos de software en Alstom. Este sistema utiliza documentos técnicos (notas de lanzamiento, arquitectura) para responder consultas sobre trazabilidad de requisitos, demostrando que el 70% de las respuestas generadas fueron consideradas adecuadas y útiles por expertos industriales.

  En la minería, @shu_utilizing_2024 presentaron un marco para la construcción de Grafos de Conocimiento Hiper-Relacionales destinados al análisis de fallas en montacargas de minas. Su metodología utiliza LLMs (GPT) para extraer entidades y relaciones complejas, optimizando los datos mediante predicción de enlaces para superar la escasez de datos en manuales de mantenimiento.

  // 3. Estrategias de Recuperación Híbrida y Optimización del Contexto
  La literatura actual destaca que la recuperación puramente vectorial es insuficiente para capturar matices semánticos específicos o terminología exacta en dominios especializados.

  @santra_curious_2025 introdujeron el concepto de "LU-RAG", un enfoque híbrido que combina el aprendizaje en contexto (ICL) utilizando datos etiquetados con RAG basado en datos no etiquetados. Su metodología recalcula dinámicamente las puntuaciones de instancias etiquetadas y pasajes no etiquetados, demostrando que esta fusión supera a los enfoques aislados en tareas de verificación de hechos y clasificación de sentimientos.

  En el dominio médico, @wang_retrieval_2025 propusieron una optimización basada en RAG para la comprensión y razonamiento de conocimiento médico. Su enfoque innovador incluye una fusión adaptativa de recuperadores dispersos (TF-IDF) y densos (Transformer), junto con ingeniería de prompts y limpieza de datos rigurosa, para mitigar alucinaciones y mejorar la precisión en el conjunto de datos CCKS-TCMBench.

  Esta tendencia hacia la hibridación también es respaldada por @knollmeyer_hybrid_2025, quienes validaron que un enfoque equilibrado (30/70) entre búsqueda vectorial y búsqueda de texto completo ofrece la recuperación más robusta para corpus técnicos bilingües, superando las limitaciones de los modelos de embedding en lenguajes específicos.

  // 4. Integración de Grafos de Conocimiento (GraphRAG)
  Para superar las limitaciones de razonamiento multi-salto (multi-hop) y fragmentación de contexto en RAG convencional, la integración de Grafos de Conocimiento (KGs) ha ganado tracción.

  @knollmeyer_document_2025 introdujeron "Document GraphRAG", un marco que estructura documentos técnicos en un Grafo de Conocimiento de Documentos (DKG). Este sistema preserva la estructura jerárquica (capítulos, secciones) y utiliza enlaces semánticos basados en palabras clave para mejorar la recuperación. La evaluación demostró que GraphRAG supera a las líneas base de RAG ingenuo, especialmente en preguntas que requieren razonamiento complejo a través de múltiples documentos. Similarmente, el trabajo de Shu et al. con grafos hiper-relacionales permite capturar relaciones multidimensionales en diagnósticos de fallas, lo cual es superior a las representaciones de relaciones binarias tradicionales.

  // 5. Avances en RAG Multimodal
  La capacidad de procesar información más allá del texto es crucial para la aplicabilidad en el mundo real.
  Además del trabajo de Nam et al. con texto e imágenes, @drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa propusieron un sistema "mRAG" unificado capaz de procesar texto, tablas, imágenes y video. Su tubería (pipeline) ingesta datos utilizando herramientas como Amazon Textract y Transcribe, y emplea LLMs multimodales (Claude 3.5 Sonnet) para el enriquecimiento semántico. Aunque el rendimiento en consultas de video e imagen aún presenta desafíos comparado con texto y tablas, su marco demuestra la viabilidad de pipelines unificados para datos diversos.

  El estado del arte actual en RAG se caracteriza por un alejamiento de soluciones genéricas hacia arquitecturas altamente especializadas. La evidencia sugiere que la combinación de recuperación híbrida, la estructuración mediante grafos de conocimiento y la integración multimodal como se ve en la taxonomía de la @fig:tree_rag son estrategias esenciales para desplegar sistemas de QA confiables en entornos críticos como el mantenimiento de equipos de mineria de plantas concentradoras.

#import "@preview/tdtr:0.4.3" : *
#figure(
  tidy-tree-graph(compact: true, draw-node: horizontal-draw-node)[
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
  ],
  caption: [Taxonomía por tipo de RAGs de lecturas revisadas]
)<fig:tree_rag>

=== Revisión comparativa: fortalezas y debilidades.  

1. *Sistema RAG Multimodal con Ajuste Fino (LoRA)*\
		@nam_lora-tuned_2025 presentan una solución robusta para la industria automotriz, específicamente para la gestión de manuales técnicos del vehículo Hyundai Staria. La técnica empleada combina un sistema RAG multimodal con el ajuste fino eficiente de parámetros (PEFT) mediante LoRA (Low-Rank Adaptation) sobre el modelo de lenguaje bLLossom-8B y el modelo de embedding BAAI-bge-m32. El dominio es estrictamente técnico-automotriz, utilizando datos extraídos de manuales en PDF que contienen tanto texto como diagramas.
		- *Resultados y Conclusiones:* El sistema logró mejoras notables frente a líneas base, con un incremento del 18.0% en ROUGE-L(27.12%-9.09%) y un 3.0% en similitud de coseno(78.11%-75.81%), destacando en la precisión de respuestas guiadas por imágenes. Los autores concluyen que la integración multimodal es esencial para la resolución de problemas técnicos complejos.
		- *Fortaleza:* Su mayor virtud es la precisión procedimental, al integrar anotaciones de similitud a nivel de oración y pares texto-imagen, el sistema ofrece instrucciones paso a paso muy superiores a la búsqueda simple.
		- *Debilidad:* La escalabilidad es su principal limitación. El mapeo imagen-texto se realizó manualmente (solo 200 pares), lo que representa un cuello de botella significativo para expandir el sistema a bibliotecas de manuales más extensas sin automatización avanzada.
	2. *Recuperación Híbrida para Documentos en Alemán*\
		@knollmeyer_hybrid_2025 abordan un problema lingüístico y técnico en la planificación de producción automotriz (caso Audi). La técnica empleada es una recuperación híbrida que fusiona la búsqueda vectorial densa (usando modelos multilingües como Cohere y Titan) con la búsqueda de texto completo dispersa (BM25F #footnote[@rajaraman_mining_2011] /TF-IDF #footnote[@robertson_probabilistic_2009]). Los datos consisten en normas industriales (VDA) bilingües, permitiendo una comparación directa entre el rendimiento en inglés y alemán.
		- *Resultados y Conclusiones:* El estudio demuestra que los modelos de embedding multilingües tienen un rendimiento inferior en textos técnicos en alemán. El enfoque híbrido mejoró la precisión de recuperación en un 10% para documentos en alemán, igualando el rendimiento obtenido en documentos en inglés (English dataset Precisión: 78.9%, MMR: 0.64; German dataset Precisión: 79.3%, MMR: 0.64).
		- *Fortaleza:* Su pragmatismo y eficiencia destacan al resolver la deficiencia de los modelos de embedding en idiomas distintos al inglés mediante la reincorporación de algoritmos clásicos (BM25), logrando una solución robusta sin el alto costo computacional del ajuste fino.
		- *Debilidad:* El estudio depende de pares de preguntas y respuestas generados por LLMs para la evaluación, lo que introduce un sesgo potencial donde las preguntas podrían ser "demasiado fáciles" o artificiales, no reflejando completamente la complejidad de la consulta humana real.
	 3. *ReqRAG: Gestión de Lanzamientos de Software*\
		@ibtasham_reqrag_nodate (Fecha inferida: 2025) proponen "ReqRAG" para el sector ferroviario en Alstom. La técnica empleada es un pipeline RAG especializado que utiliza modelos OCR (YOLOX, Detectron2) para la extracción de datos de PDFs complejos y compara varios LLMs (Phi-3, Llama-3.2) para la generación. El dominio es la gestión de requisitos y lanzamientos de software, utilizando datos de documentos de arquitectura y notas de lanzamiento.
		- *Resultados y Conclusiones:* La evaluación humana indicó que el 70% de las respuestas cumplieron el críterio adequacy avg=3.69, usefulness avg=3.44, Relevance avg=3.32. Se concluye que la combinación de embeddings 'stella_v5' con OCR Detectron2 ofrece una buena recuperación.
		- *Fortaleza:* La validación industrial real es su punto fuerte; al incluir una evaluación cualitativa con expertos de dominio, el estudio trasciende las métricas sintéticas y prueba su utilidad en un entorno crítico cercano al real.
		- *Debilidad:* El volumen de datos es bajo (solo 7 documentos técnicos), lo que plantea dudas sobre la generalización de los resultados a repositorios documentales masivos. Además, el uso de modelos de código abierto plantea interrogantes sobre la privacidad de datos propietarios.
	4. *Grafos de Conocimiento Hiper-Relacionales en Minería*\
		@shu_utilizing_2024 se enfocan en el análisis de fallas para sistemas de izaje para minas. La técnica empleada es la construcción de Grafos de Conocimiento Hiper-Relacionales (HKG) utilizando LLMs (GPT) para la extracción de tuplas complejas y algoritmos de predicción de enlaces para completar datos faltantes. El dominio es el mantenimiento de sistema de izajes mineros, utilizando datos de informes de inspección y registros de mantenimiento.
		- *Resultados y Conclusiones:* El modelo optimizado (MHSD) logró una mejora de 0.008 en la métrica MRR (Mean Reciprocal Rank) sobre datos no optimizados y superó al modelo KICGPT. Se concluye que la representación hiper-relacional captura mejor la complejidad de las fallas mecánicas de sistemas de izaje mineros.
		- *Fortaleza:* La capacidad de modelado complejo es superior; al utilizar grafos hiper-relacionales, el sistema puede representar matices (condiciones, causas, consecuencias) que se pierden en los grafos de conocimiento binarios tradicionales.
		- *Debilidad:* Existe una fuerte dependencia de la ingeniería de prompts para la generación de datos y la extracción de relaciones, lo que puede introducir inconsistencias si el modelo generativo alucina o malinterpreta la terminología técnica sin una supervisión estricta.
	5. *Optimización RAG para Conocimiento Médico*\
		@wang_retrieval_2025 proponen una optimización algorítmica para el sector salud. La técnica empleada incluye una fusión adaptativa de recuperadores dispersos (TF-IDF) y densos (Transformer), junto con una rigurosa limpieza de datos e ingeniería de prompts. El dominio es la medicina clínica y el razonamiento semántico, utilizando datos del benchmark CCKS-TCMBench (exámenes médicos y casos clínicos).
		- *Resultados y Conclusiones:* El modelo optimizado superó a las líneas base (incluyendo GPT-4 y ChatGLM3) en métricas de precisión y razonamiento, con un aumento promedio del 3.86% en métricas integrales. Se concluye que la fusión de recuperadores y la limpieza de datos son críticas para reducir alucinaciones médicas.
		- *Fortaleza:* La robustez metodológica en el preprocesamiento y la fusión de recuperadores permite mitigar el riesgo de alucinaciones, un aspecto crítico y no negociable en aplicaciones médicas.
		- *Debilidad:* La evaluación se basa principalmente en datos de competiciones, que aunque estandarizados, pueden no capturar la diversidad de los datos clínicos del mundo real (historias clínicas no estructuradas).
	6. *Document GraphRAG en Manufactura*
		@knollmeyer_document_2025 introducen "Document GraphRAG" para entornos de manufactura. La técnica empleada estructura documentos en un Grafo de Conocimiento de Documentos (DKG) que preserva la jerarquía (capítulos, secciones) y utiliza enlaces basados en palabras clave para la recuperación. El dominio es la manufactura automotriz, evaluado con datos públicos (SQUAD, HotpotQA) y un conjunto de datos interno de planificación de producción.
		- *Resultados y Conclusiones:* GraphRAG superó consistentemente a RAG ingenuo en métricas de recuperación y generación, mostrando beneficios notables en preguntas de razonamiento multi-salto (multi-hop). Se concluye que preservar la estructura del documento es vital para consultas complejas. 
		- *Fortaleza:* Su capacidad para el razonamiento estructural; al mapear explícitamente la jerarquía del documento en el grafo, el sistema puede responder preguntas complejas que requieren entender el contexto de secciones completas, no solo fragmentos aislados.
		- *Debilidad:* El costo de latencia es alto; el tiempo de respuesta total fue aproximadamente 5 veces mayor que el de un sistema naive RAG (9.8s vs 1.7s), lo que podría afectar la experiencia de usuario en aplicaciones de tiempo real.
	7. *LU-RAG: Fusión de Datos Etiquetados y No Etiquetados*\
		@santra_curious_2025 presentan un enfoque teórico-práctico denominado LU-RAG. La técnica empleada es un marco híbrido que combina el aprendizaje en contexto (ICL) usando ejemplos etiquetados con RAG basado en documentos no etiquetados, utilizando una combinación lineal de puntuaciones. El dominio es general (NLP), aplicado a tareas de verificación de hechos y clasificación de sentimientos, usando datos de FEVER y SST.
		- *Resultados y Conclusiones:* LU-RAG superó tanto a ICL puro como a RAG puro, logrando, por ejemplo, una mejora del 19.45% en F1-score para verificación de hechos frente a líneas base supervisadas. Se concluye que equilibrar datos etiquetados y no etiquetados ofrece "lo mejor de ambos mundos".
		- *Fortaleza:* La innovación algorítmica al fusionar paradigmas; demuestra que la combinación dinámica de ejemplos de entrenamiento (few-shot) con conocimiento externo (retrieval) es superior a usarlos por separado.
		- *Debilidad:* La sensibilidad de hiperparámetros; el rendimiento depende crucialmente del parámetro $alpha$ (proporción de mezcla), que varía según la tarea y es difícil de generalizar sin un ajuste fino específico para cada nuevo conjunto de datos.
	8. *RAG Multimodal Unificado (mRAG)*\
		@drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa proponen un sistema unificado para procesar múltiples tipos de medios. La técnica empleada es un pipeline "mRAG" que ingesta texto, tablas, imágenes y video utilizando servicios de AWS (Textract, Transcribe) y LLMs multimodales (Claude 3.5 Sonnet) para generar descripciones semánticas45. El dominio es el soporte técnico de servidores (manuales Dell), usando datos de PDFs y videos instruccionales.
    - *Resultados y Conclusiones:* El sistema mejoró la relevancia contextual y redujo alucinaciones. Sin embargo, el rendimiento en consultas de video e imagen fue inferior al de texto y tablas. Se concluye que un pipeline unificado es viable pero requiere mejoras en la interpretación de datos visuales no estructurados.
    - *Fortaleza:* La arquitectura unificada; es uno de los pocos enfoques que integra nativamente el video como una modalidad de recuperación junto con texto y tablas, abriendo la puerta a asistentes técnicos verdaderamente completos.
    - *Debilidad:* El rendimiento desigual entre modalidades. La precisión en la recuperación de video e imagen sigue siendo baja comparada con el texto, lo que limita su fiabilidad en escenarios donde la información visual es crítica.

=== Vacíos y oportunidades de investigación.
En general el principal vacío de los papers revisados son información concreta de la latencia de sus propuestas, no se explican el tiempo que le toma a sus sistemas RAG en responder, a excepción de @knollmeyer_document_2025 quien menciona esta limitante.
Otro vacío es la falta de técnicas Parameter-Efficient Fine-Tuning (PEFT) el cual solo en el paper @nam_lora-tuned_2025 se aplica y explica, el resto de paper no tocan esta técnica por lo que no se puede comparar o afirmar que es efectiva en mas tipos de data o dominio.



// 1. Permitir que las figuras se rompan entre páginas
#show figure: set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
// // 2. Índice de Tablas
// #heading("Índice de Tablas")
// #outline(title: none, target: figure.where(kind: table))
// #pagebreak()

// 3. Tabla Corregida
#figure(
  table(
    columns: (10%, 18%, 18%, 14%, 20%, 20%),
    inset: 1.5pt,
    align: left + top,
    stroke: 0.5pt + luma(180),
    
    // --- CORRECCIÓN AQUÍ ---
    // El color de fondo se define en la tabla, no en el header.
    // "Si la fila (y) es 0, usa gris claro, si no, nada".
    fill: (x, y) => if y == 0 { luma(230) },

    table.header(
      // Ya no ponemos 'fill' aquí dentro
      [*Trabajo*], [*Método*], [*Datos/Dominio*], [*Métrica clave*], [*Fortalezas*], [*Debilidades*]
    ),

    // --- DATOS ---
    [@nam_lora-tuned_2025 Hyundai],
    [mRAG+LoRA Fine Tuning (0.1%)],
    [Manuales Hyundai Staria PDF. \ \ simple QA dataset, \ multi-turn QA dataset, \ RAG QA dataset],
    [ROUGE-L: 27.12% \ BERT: 78.11% \ Encuesta: 4.4/5],
    [
      - Integra texto e imágenes.
      - Similitud semántica en el dominio automotriz.
    ],
    [
      - Escalabilidad limitada por las anotaciones manuales.
      - Dominio restringido al mantenimiento automotriz.
      - No aplica técnicas de optimización.
    ],

    [@knollmeyer_hybrid_2025 (2025) hybrid],
    [Hybrid Retrieval 30/70 \ (búsqueda vectorial Amazon Titan + búsqueda texto completo BM25F/TF-IDF)],
    [18 normas y estándares de VDA. \ Corpus idénticos en alemán e inglés. \ Data sintética QA (Claude Sonnet 3.5)],
    [Precisión: 0.79 \ MMR: 0.64],
    [
      - Dominio en idioma alemán.
      - Técnica escalable y simple.
      - Alta precisión.
    ],
    [
      - Se centra únicamente en texto.
      - Evaluación con datos sintéticos.
      - Falta de análisis de latencia.
    ],

    [@ibtasham_reqrag_nodate Software],
    [ReqRAG \ OCR (YOLOX) \ Embeddings: \ mxbai-embeb-large-v1 \ stella_en_400M_v5],
    [Gestión de Releases de Software Ferroviario (Alstom). \ 7 docs técnicos (TCMS), 27 queries reales.],
    [R$@$3 (Recall): 0.90 \ Adequacy (Humana): 3.69/5],
    [
      - Capacidad de procesar tablas y diagramas.
      - Alta precisión en el dominio industrial.
    ],
    [
      - Dataset de evaluación pequeño.
      - Dependencia crítica del OCR.
      - Ventana de contexto limitada para documentos extensos.
    ],

    [@shu_utilizing_2024  Knowledge Graph],
    [LLM-HKGCF (GPT o1 preview) \ Embeddings: UltraFastBERT],
    [MHSD (Mine Hoist System Dataset) \ Open Dataset (JF17K, WD50K)],
    [MRR: 0.715],
    [
      - Alta automatización en extracción de conocimiento.
      - Representación de contextos multidimensionales.
      - Reducción de costos de construcción manual.
    ],
    [
      - Problemas de interpretabilidad de fallas.
      - Uso de LLMs costosos.
      - Limitado al dominio del dataset.
      - No soporta tablas/figuras.
    ],

    [@wang_retrieval_2025  Medical],
    [RAG optimizado \ (sparse TF-IDF y dense retrievers basado en Transformers)],
    [Examen de licenciatura médica de China (9,788 preguntas examen, 5,473 práctica)],
    [Comprensión: \ Precisión: 0.83 \ Rouge-L: 0.23 \ BertScore: 0.71 \  Razonamiento: \ Precisión: 0.91],
    [
      - Reducción de alucinaciones.
      - Eficiencia de recursos (no fine-tuning).
      - *Trazabilidad:* permite rastrear la fuente original.
    ],
    [
      - Representatividad de los datos.
      - Falta de validación clínica en entornos reales.
      - Complejidad en resolución de ambigüedades.
    ],

    [@knollmeyer_document_2025 \ Graph RAG],
    [*Document GraphRAG* \ Estrategias búsqueda: \ 1. ICS \ 2. IKS \ 3. UKS],
    [(AUDI AG) \ 17 documentos técnicos, +5,500 páginas. \ Planificación y estándares en alemán.],
    [Recuperación: \ Recall$@$700: 0.79 \ MRR$@$1300: 0.63 \  Generación: \ Faithfulness: 0.94],
    [
      - Escalabilidad en creación de grafos.
      - Robustez a pérdida de rendimiento.
      - Calidad de respuestas.
      - Pipeline modular.
    ],
    [
      - Costoso (APIs externas).
      - Latencia elevada.
      - Susceptible a ruido por redundancia.
      - No soporta tablas/imágenes.
    ],

    [@liu_hm-rag_2025 \ text-image],
    [*HM-RAG* \ (Vectorial, grafos y web)],
    [ScienceQ \ CrisisMMD],
    [Accuracy: 93.73],
    [
      - Modularidad y escalabilidad.
      - Soporta datos heterogéneos.
      - Reducción de alucinaciones.
    ],
    [
      - Latencia en búsqueda web.
      - Dependencia APIs externas.
      - Complejidad de infraestructura.
      - Costo computacional alto.
    ],

    [@drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa \ table, image],
    [mRAG (Multimodal RAG)],
    [36 documentos servidores Dell. \ 82 manuales video. \ 116 preguntas (texto, tablas, imágenes, videos)],
    [Contextual Precision: 0.35 \ Contextual Recall: 0.69],
    [
      - Procesa 4 modalidades en un pipeline.
      - Buena calidad en tablas y texto estructurado.
    ],
    [
      - Bajo rendimiento en video.
      - Datos no estructurados difíciles.
      - Dependencia de AWS.
    ]
  ),
  caption: [Comparativa del Estado del Arte en Sistemas RAG y Multimodales],
  kind: table,
  supplement: [Tabla]
) <tabla_sota>

