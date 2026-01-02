= Marco Teórico y Estado del Arte
== Bases Teóricas
=== Fundamentos de IA:
El desarrollo de LLMs en el mundo de creación de sistemas capaces de razonar y responder consultas ha ido en constante desarrollo, actualmente se tiene fundamentos donde se desarrollaron la capacidad de incrementar la información de LLMs con el fin de adaptar los estos a nuevos contextos en especifico y con el fin de este proyecto responder a consultas sobre docuemntos técnicos de mantenimiento para ellos surgen conceptos claves como son los siguientes:  

// + *Procesos de planta concentradoras*
//   La comprensión del dominio minero es esencial para evaluar la relevancia de las respuestas generadas. Los manuales técnicos en este sector cubren procesos con físicas y modos de falla distintos:
  
//   Conminución (Chancado y Molienda): Es la etapa más intensiva en energía. Los manuales de Chancadores Giratorios y Cónicos (ej. Metso MP, FLSmidth) contienen procedimientos críticos de ajuste del setting (CSS) y cambio de revestimientos (mantles/bowls), tareas que involucran manipulación de componentes de varias toneladas. Los Molinos SAG y de Bolas requieren mantenimiento especializado en sus sistemas de transmisión (coronas, piñones) y lubricación hidrostática, donde un error en la interpretación de las tolerancias de presión de aceite puede fundir una chumacera.   
  
//   Flotación: Involucra Celdas de Flotación (mecánicas, neumáticas, columnas) donde el mantenimiento se centra en los mecanismos de agitación (rotores, estatores) y sistemas de instrumentación. La documentación técnica aquí es rica en diagramas de lazos de control y esquemas de distribución de aire.   
  
//   Gestión de Fluidos y Relaves: Los Espesadores y Bombas de Relaves son críticos para la continuidad hídrica. Los manuales detallan el mantenimiento de rastras hidráulicas y sistemas de accionamiento (drives) con altos torques. La interpretación correcta de las curvas de operación de las bombas es vital para evitar cavitación o arenamiento de líneas.

+ *Grandes Modelos de Lenguaje (LLMs) y sus Limitaciones*
  Los LLMs, fundamentados en la arquitectura Transformer, han revolucionado el Procesamiento de Lenguaje Natural (NLP). Modelos como la serie GPT, Llama, o bLLossom (utilizado por Nam et al. en el estudio base) poseen una capacidad semántica profunda. Sin embargo, en el dominio de mantenimiento industrial, presentan limitaciones estructurales conocidas como la "tríada de la inviabilidad":   
  
    - Alucinaciones: La tendencia a generar información plausible pero factualmente incorrecta. En minería, un LLM podría "inventar" un procedimiento de bloqueo de energía basándose en datos generales de internet, lo cual es inaceptable bajo normativas de seguridad.   
  
    - Obsolescencia del Conocimiento (Cut-off Date): El conocimiento paramétrico de un LLM es estático. No puede conocer las actualizaciones recientes de un manual de fabricante o los cambios en los PETS (Procedimientos Escritos de Trabajo Seguro) de la mina, que son documentos vivos.   
  
    - Falta de Acceso a Datos Propietarios: Los manuales detallados de plantas concentradoras (ej. planos as-built de Cerro Verde o Antamina) son propiedad intelectual privada y no forman parte de los corpus de entrenamiento públicos. 
    
+ *Generación Aumentada por Recuperación (RAG)*
  RAG es un enfoque de generación que combina un motor de recuperación de información con un modelo de lenguaje para producir respuestas basadas en evidencia externa. El proceso se organiza en tres momentos: preparación del conocimiento, recuperación y generación. En la preparación, los documentos se limpian, se segmentan en fragmentos manejables, se enriquecen con metadatos y se indexan usando representaciones textuales que permiten buscarlos de manera eficiente. En la recuperación, ante una consulta, el sistema localiza los fragmentos más pertinentes mediante búsqueda léxica, búsqueda semántica o una mezcla de ambas, y opcionalmente reordena los resultados con modelos más precisos. En la generación, el modelo de lenguaje redacta una respuesta condicionada por la consulta y por los fragmentos recuperados, manteniendo la trazabilidad hacia las fuentes. RAG se entiende como una arquitectura donde el conocimiento principal se mantiene fuera del modelo y puede actualizarse sin re-entrenamiento, mientras el modelo actúa como redactor que integra y explica la evidencia encontrada.
  - Fase de Ingesta (Indexing): Descomposición de documentos PDF técnicos en fragmentos (chunks). A diferencia del texto plano, los manuales técnicos requieren estrategias de chunking que respeten la estructura del documento (encabezados, tablas), preservando el contexto semántico.29 Estos fragmentos se convierten en vectores densos (embeddings) mediante modelos como BAAI-bge-m3.4
  - Fase de Recuperación (Retrieval): Ante una consulta del usuario (ej. "¿Cuál es el torque de los pernos del revestimiento del Molino SAG?"), el sistema busca en la base de datos vectorial los $k$ fragmentos más similares semánticamente utilizando métricas de distancia (similitud del coseno o producto punto).
  - Fase de Generación (Generation): El LLM recibe un prompt enriquecido que incluye la consulta del usuario y los fragmentos recuperados como "contexto de verdad". El modelo es instruido para responder basándose exclusivamente en este contexto, citando las fuentes.
  
+ *RAG multimodal*
  RAG multimodal extiende el principio anterior a fuentes de información heterogéneas como texto, imágenes, diagramas, tablas, audio transcrito y documentos con maquetación compleja. La definición abarca una cadena de procesamiento que inicia con la ingesta y normalización de los datos (incluyendo reconocimiento óptico de caracteres, extracción de tablas y detección de figuras), continúa con la creación de representaciones comparables entre modalidades para poder indexarlas y consultarlas en un espacio común, y culmina con la generación condicionada por evidencias de distinta naturaleza. En este marco, una consulta puede ser textual, visual o mixta; la respuesta puede incluir texto anclado a regiones de una imagen, referencias a celdas de una tabla o a secciones específicas de un documento. El sistema prioriza conservar el contexto estructural del origen, de modo que el usuario pueda verificar rápidamente la procedencia de cada afirmación.
  - Alineación Semántica Manual/Híbrida: @nam_lora-tuned_2025. proponen un mapeo explícito donde las imágenes se vinculan a los párrafos de texto correspondientes antes de la vectorización. Esto asegura que, al recuperar un procedimiento textual, el sistema también recupere la imagen asociada para presentarla al usuario final.   

+ *Low-Rank Adaptation (LoRA)*
  LoRA es una técnica de adaptación eficiente para modelos de lenguaje grandes que incorpora módulos adicionales de bajo costo computacional dentro de capas ya existentes del modelo. En lugar de modificar de manera completa los parámetros preentrenados, LoRA introduce pequeños componentes entrenables que actúan como correcciones y permiten especializar el comportamiento del modelo hacia un dominio, un estilo de respuesta o una tarea concreta. Estos componentes se insertan típicamente en proyecciones de atención y en capas internas del transformador y se entrenan manteniendo congelados los parámetros originales. La definición práctica de LoRA incluye la selección de dónde insertar los módulos, la configuración de su tamaño y la posibilidad de combinar varios adaptadores para distintas tareas sin interferencias, conservando compatibilidad con el flujo normal de inferencia @hu_lora_2021.\
  El vocabulario minero es altamente específico (ej. "chancado", "hidrociclón", "relave"). Los LLMs generalistas pueden no interpretar correctamente estos términos. El reentrenamiento completo (Full Fine-Tuning) es costoso y computacionalmente inviable para muchas operaciones. La técnica LoRA (Low-Rank Adaptation), empleada en el estudio base, congela los pesos del modelo pre-entrenado e introduce matrices de bajo rango entrenables en las capas de atención del Transformer. Esto permite adaptar el modelo al lenguaje técnico y al estilo de respuesta ("instruccional") requerido en mantenimiento, modificando menos del 1% de los parámetros totales, lo que facilita su despliegue en infraestructura local (on-premise) típica de faenas mineras.
  
+ *Agentes basados en LLM*
  Un agente basado en un modelo de lenguaje es un sistema que percibe un estado del entorno, razona en lenguaje natural para planificar pasos, ejecuta acciones mediante herramientas externas y mantiene memoria a lo largo de múltiples interacciones. Se define por un ciclo continuo: interpretar la situación y el objetivo, decidir la siguiente acción, llamar a una herramienta si es necesario (por ejemplo, un buscador, una base de datos, un extractor de tablas o un ejecutor de código), integrar el resultado al contexto y actualizar su plan. La memoria del agente puede registrar episodios de conversación, hechos persistentes y procedimientos reutilizables. La orquestación suele expresarse como flujos o grafos de estados que indican cuándo razonar, cuándo recuperar información, cuándo verificar y cuándo responder. De este modo, el agente no solo produce texto, sino que coordina recursos de información y cómputo para alcanzar metas definidas por el usuario.
  
+ *Métricas y evaluación: ROUGE-L*
  Es una métrica de comparación entre un texto candidato y una referencia que se basa en la subsecuencia común más larga para estimar cobertura y respeto del orden relativo de las palabras. La idea central es medir cuánto del contenido de la referencia aparece en el candidato manteniendo la secuencia de aparición, aunque no sea de forma contigua. A partir de la longitud de esa subsecuencia se obtienen medidas de cobertura sobre la referencia, precisión sobre el candidato y una combinación de ambas. ROUGE-L se usa ampliamente en tareas de resumen y en evaluación de generación porque captura de forma simple la presencia y el orden de unidades léxicas relevantes, y se puede agregar a nivel de documento o corpus manteniendo un procedimiento de cálculo transparente.
- *Metodologías: DSR, MLOps*.



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
  La literatura actual (2023-2025) permite clasificar las soluciones de IA para gestión de conocimiento técnico en tres generaciones evolutivas @gao_retrieval-augmented_2024:
  
  - Sistemas de Recuperación Basados en Palabras Clave (Lexical Search) :
  
  - Método: Utilizan algoritmos como TF-IDF o BM25.
  
  - Aplicación: Motores de búsqueda tradicionales en gestores documentales (DMS).
  
  - Limitación: No capturan el contexto semántico; fallan ante sinónimos o consultas naturales complejas. 

+ Sistemas RAG Naive (Ingenuos) Unimodales @gao_retrieval-augmented_2024:

  - Método: Indexación vectorial de texto plano + recuperación semántica + generación con LLM genérico.
  
  - Aplicación: Chatbots de primera generación para manuales simples.
  
  - Limitación: Sufren del problema de "Lost in the Middle", alucinaciones frecuentes y ceguera ante imágenes/tablas.
+ Sistemas RAG Modulares y Multimodales (MM-RAG) - Enfoque de la Tesis @knollmeyer_document_2025:
  
  - Método: Integran módulos de re-ranking, grafos de conocimiento (GraphRAG) y procesamiento de visión (Vision Encoders) para ingerir texto, imágenes y diagramas conjuntamente.
  
  - Tecnologías: Modelos de Embeddings densos (ej. BAAI-bge-m3), Bases de Datos Vectoriales, y adaptación de dominio vía PEFT/LoRA.
  
  - Aplicación: Estado del arte para documentación técnica compleja (automotriz, aeroespacial, y ahora propuesto para minería). 

- Revisión comparativa: fortalezas y debilidades.

- Vacíos y oportunidades de investigación.
