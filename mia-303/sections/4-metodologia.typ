= Metodología de Investigación

== Enfoque Metodológico
- El proyecto sigue una metodología *Agile+IA & CRISP-DM* combinada con el fin de poder adaptarse a las exigencias del stakeholder que solicita avances continuos y ser robusto en la etapa del procesamiento de la datos hereogeneos texto, imagenes, tablas de los Manuales tecnicos de mantenimiento. La metodología CRISP-DM (Cross-Industry Standard Process for Data Mining) contribuye en reforzar la parte de comprensión tanto del negocio como de los datos y su procesamiento, cabe mencionar que la metodología SEMMA no se considero ya que su etapa de muestreo puede ser contra producente ya que la data (pdfs) tienen estructuras diversas las cuales tienen que ser tomadas en su totalidad desde el principio y así tener comparaciones reales.
//descartamos la metodología SEMMA ya que su etapa de muestreo(sampler) es contraproducente debido a que este puede generar mala interpretación de los pdfs son difenretes variando de quipos y fabricantes. 

- *Agile+IA* 
Como metodologia plantea los siguientes sprints para el desarrollo, implementación del proyecto.
Ciclo de investigación: 
  - *Sprint 0 - Revisión de papers y baselane - Comprensión del negocio (CRISP-DM): * <sprint_0>
    Como parte inicial se va a revisar literatura relacionada con papers relevantes y analizarlos de forma que podamos entender el problema y las propuestas de solución entendiendo tanto las fortalizas, debilidades y oportunidades de mejora que se tienen en los papers, como principal entregable se tienen los Controles de Lectura que detallan el análisis realizado.
    - Objetivo: \
      - Desarrollar el estado del arte del proyecto.
      - Elaboración de Controles de Lecturas como apoyo para la elaboración del estado del arte.
      - Se va a comenzar a reunir con el stakeholer con el fin de entender y adecuar el estado del arte a la necesidad o problema a resolver.
  - *Sprint 1 - Preparación de datos(CRISP-DM):* <sprint_1>
    Se va a desarrollar un pipeline de ingesta de los manuales técnicos, estos van a ser procesados y analisados con una herramienta que tenga la capacidad de procesar texto, tablas e imagenes:
    Objetivos:\
      - Extracción de texto: se va a emplear la libería de python PyMuPDF, como se uso en el paper @nam_lora-tuned_2025 quien logro buenos resultados.
      - Extracción de tablas e imagenes, en este caso se tiene la propuesta de usar Gemini-2.5-flash siendo una variante a la propuesta en el paper @drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa, quien utiliza CLaude 3.5 Sonnet.
      - Embedding and Indexing: esta sub etapa consiste en vectorizar tanto el texto, tablas e imagenes y almacenarlo en un gestor de base de datos de embeddings, metadata y almacenamiento de documentos, como propuesta se tiene a Chroma como primera opción.
      - EDA Rápido: estadistivas y visualizaciones de la calidad de los documentos técnicos y las técnicas de extracción propuestas.

      #figure(image("../assets/rag_pipeline.png"), caption:[Pipeline de extracción de información de PDFs @drushchak.etalMultimodalRetrievalAugmentedGenerationUnifiedInformationa ])

  - *Sprint 2 - Modelado(CRISP-DM):* <sprint_2>
    En esta etapa pretendemos implementar un naive RAG (RAG básico) como se propone en el estado del arte, con el cual se pretende tener un punto de partida con el cual se pueda demostrar la mejora en cada iteración, y poder realizar los primeros analisis y comenzar con la iteración de pruebas con indicadores o metricas propuestas como son ROUGE-L y BertScore.
    Objetivos:\
    - naive RAG: crear el chat inteligente con RAG de manuales técnicos, se propone utilizar LangChain como plataforma para desarrollar el RAG.
    - EDA Rápido: metricas del naive RAG y registro de resultados.
  
  - *Sprint 3 - Generación de Demo:* <sprint_3>
    En esta etapa pretendemos demostrar el naive RAG y generar una plataforma donde el stakeholder y usuarios puedan usar el artefacto y brindar feedback.
    Objetivo:\
    - Creación de plataforma de interacción 1 a 1 (1 usuario a la vez con el artefacto), se propone implementar con el paquete de python Gradio por sus ventajas y facilidad para el prototipado con fines de chats inteligentes.
    - Informe con feedback (EDA) de usuarios seleccionados por el stakeholder. 

  - *Sprint 4 - Integración multimodal del RAG:* <sprint_4>
    Se va a tener un spring para la integración de los 3 tipos de datos o datasets (texto, tablas e imagenes), para ellos se va a usar la base de datos Chroma integrada con LangChain y poder cumplir con este objetivo..
    - Creación de Multimodal RAG he implementación en la plataforma Gradio.
    - Informe con feedback de usurios (EDA).
  
  - *Sprint 5 - Fine tuning del modelo base (aplicación de LoRA):* <sprint_5>
    En este spring se pretende realizar el fine tuning de modelo base y lograr un modelo con el dominio de mantenimiento de equipos de plantas concentradoras.
    Objetivo:
    - Creación de modelo fine-tuned: Para lograr esta implementación se apoya en la librearía PEFT de huggingface. 
    - Documento Feedback del nuevo modelo y sus resultados.
    - Integración del feedback desde la plataforma Gradio. 

  - *Sprint 6 - Revisión de feedback y refinamiento del sistema:* <sprint_6>
    Ene sta etapa se tiene como objetivo levantar observaciones respecto al sistema integrado y poder mejorar factores claves de las metricas elegidas.
    Obejtivos:
    - Generar un sistema mRAG Fined-tuned con metricas esperadas por el stake holder.

diseño, implementación, validación, comunicación.

== Diseño Experimental
  + *Pipeline de datos*: 
    - Fuente de datos: PDF de manuales de mantenimientos de equipos críticos
    - Ingesta: Se va a implementar 3 tipo de procesamiento y generación de 3 tipos de dataset:
      + PDF a texto: Dataset de QA texto-texto, con PyMuPDF.
      + PDF a imagen-texto: Dataset de texto-imagen, con un llm(Gemini) para generar tanto los bounding boxes y la descripción semantica de la imagen.
      + PDF a tablas-texto: Dataset de tablas-tablas, con un llm(Gemini) para generar tanto los bounding boxes y la descripción semantica de la tabla, tambien se va a extraer información de la tabla con PyMuPDF/OCR (alternativa a evaluar).
    - Preprocesado y Feature engineering: 
      + Con la herramienta LangChain se va a realizar el enriquecimiento semantico y extracción de palabras claves mediante Gemini como se detalla en el punto de ingesta.
      + Se va a generar la base de datos vectorial y de documentos con ChromaDB donde se va a almacenar los embeddings.\
    - Entrenamiento.
      Se va a implementar PEFT(Parameter-Efficient Fine-Tuning) para dotar o enriqueser el dominio en mantenimiento de equipos de planta concentradora mediante LoRA utilizando librerías como Hugging Face PEFT para adaptar LLMs.

  + *Validación*: 
    - Se va a implementar el framwork RAGAS para poder realizar una evaluación automatizada como es el caso de @knollmeyer_hybrid_2025.

  + *Métricas*:
    - Se va a usar LangSmith libreria de langchain para poder aplicar las métricas ROUGE-L y BERTScore.

== Interacción con Stakeholders
- Plan de consultas y retroalimentación a partir del spring 3, .
