= Administración del plan de tesis

== Cronograma

#figure(image("../assets/cronograma.png"), caption:[Cronograma de implementación de proyecto])

== Presupuesto

#figure(image("../assets/budget.png"), caption:[Presupuesto de implementación de proyecto])


== Financiamiento
El stakeholder va a sustentar al área de finanzas en un proceso de caso de negocio para poder financiar el proyecto, sin embargo de cumplir con las espectativas de los usuarios el área de TI se compromete a crear un espacio para la implementación en el sistema de la compañia (Azure).

// Desactivamos numeración para lo final
#set heading(numbering: none)

// = Referencias Bibliográficas
// Listado de referencias según normas APA o IEEE.

= Anexos:

== Sprint 1 - Modelo Baseline

=== Resumen Ejecutivo
EDA del sprint 1 donde se analizaron los documentos descargados del Aconex esta descarga consiste en *18.7Gb comprimido*(20.24 Gb descomprido) de documentos PDFs y otros comprimidos que no consideran unicamente manuales de mantenimiento, por lo que se realizo una revisión manual para identificar que documentos son efectivamente manuales, esto resulto en la reducción a *24 pdf que efectivamente son manuales de mantenimiento*, los cuales fueron procesados para la extracción de texto y posterior análisis.

=== Sprint Preparación de datos
*Objetivo del Sprint:* El principal objetivo del del sprint 01 es la extracción y analisis de los PDFs de manuales de mantenimiento de equipos críticos.

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 10pt,
  align: horizon,
  fill: (x, y) => if y == 0 { gray.lighten(80%) },
  [*ID*], [*Historia / Tarea*], [*Responsable*], [*Estado*],
  [0], [Revisión de papers y baseline (CRISP-DM)], [Johan Callomamani], [Hecho],
  [1], [Preparación de datos (CRISP-DM)], [Johan Callomamani], [Hecho],
  [2], [Modelado (PDF/Imágenes)], [Johan Callomamani], [Por Empezar],
  [3], [Generación de Demo ], [Johan Callomamani], [Por Empezar],
  [4], [Integración multimodal del RAG], [Johan Callomamani], [Por Empezar],
  [5], [Fine tuning del modelo base (Aplicación de LoRA)], [Johan Callomamani], [Por Empezar],
  [6], [Revisión de feedback y refinamiento], [Johan Callomamani], [Por Empezar],
)

=== Data Pipeline Básico
+ *Descripción:* 
  - Procesamiento de transmittals descargados del Aconex (18.7 Gb de documentos de todo tipo no solo manuales de mantenimiento).
  - Se identifica y clasifica los archivos que son manuales de mantenimiento.
  - Se realiza un análisis de los archivos generando una archivo csv con características de los archivos y graficas donde se analisan los pdf.
+ *Entregable:* 
   - Lista de archivos transmittal `aconex_file.zip`
   - notebook de analisis de los pdfs `exploratory_analysis.ipynb`
   - `list_files.csv`
   - Graficas de análisis de los pdfs.
+ *Comentarios/Problemas:* Los documentos no son solo manuales, se tienen documentos adicionales, de no interés para el proyecto:
  - Información de ordenes de compra.
  - Solicitudes de transporte de componentes.
  - Documentos de aceptación de reportes y aceptación de proyectos.
  - Gantts de inspección/QA.
  - Invoices.
  - Monthly reports (avances de la etapa de proyecto).

=== EDA Rápido
+ *Hallazgos:*
  - Total de documentos 3607 archivos con un peso total 20.24 Gb (ver @grafico_01-eda).
  - Alta densidad de documentos PDF con un total 78.53% (ver @grafico_01-eda).
  - Se encontró que no todos los archivos son manuales, el 22% son pagos y solo el 60% corresponde a información técnica relevante.
  - Para una configuración de Size: 1000, Overlap=200 para realizar la partición de los documentos tenemos como resultado Total chunks (vectors): 56,625, avg chunks per document: 2359.4 (ver @grafico_03-eda).
  - Se tiene archivos mayoritariamente con paginas de 100 a 1800 paginas, sin embargo hay 2 archivos con cantidad superior a las 5000 paginas (ver @grafico_02-eda)
  - Se valida que se tiene una relación directa entre cantidad de paginas y el tamaño del archivo (ver @grafico_02-eda), lo cual indica que normalmente todas las paginas tiene contenido y no son paginas en blanco.
  - Considerar que el ruido en el texto (simbolos, caracteres especiales) es en su mayoria menor al 0.15 lo que indica que la calidad de los documentos es buena (ver @grafico_04-eda), sin embargo hauy 1 archivo que supera por poco y requiere una revisión adicional.
  - Considerar que los documentos de menor cantidad de palabras son los que tiene mayor diversidad de vocabulario (ver @grafico_04-eda), lo que indica que son documentos con mayor cantidad de paginas/palabras reduce su vocabulario diverso.
  - Los archivos son predominantemente manuales del año 2019 (ver @grafico_04-eda).
  - Se requiere revisar y retirar terminología que se repite constantemente en todas las hojas, textos de los foot headers, numeración de paginas, logos de empresas, etc (ver @grafico_04-eda), ya que al ser predominandtes en los pdfs puede afectar en la implementación del RAG.

+ *Visualizaciones:*.

#figure(image("../assets/image06.png", width: 100%), caption: [Análisis del total de archivos *sección 01* Distribusión del peso de los archivos por cantidad de estos, *sección 02* Cantidad de archivos por tipo de archivos, *sección 03* Peso de archivos por tipo de archivos, *sección 04* Boxplot del peso de los archivos ]) <grafico_01-eda>


#figure(image("../assets/image01.png", width: 100%), caption: [Análisis de 24 pdf Manuales de mantenimiento de equipos críticos *sección 01:* Cantidad de archivos por cantidad de paginas de los archivos, *sección 02:* Cantidad de archios por cantidad de palabras, *sección 03:* tamaño de los archivos por cantidad de paginas, *sección 04:* Boxplot de la cantidad de caracteres]) <grafico_02-eda>

#figure(image("../assets/image02.png", width: 100%), caption: [Análisis de estimación de Chunks por archivos]) <grafico_03-eda>

#figure(image("../assets/image03.png", width: 100%), caption: [Análisis del contenido de los archivos de mantenimiento]) <grafico_04-eda>

#figure(image("../assets/image05.png", width: 100%), caption: [Mayor cantidad de terminas en los archivos de mantenimiento]) <grafico_05-eda>


- *Entregables:*
  - Informe EDA.
  - Piline de procesamiento de PDFs.
  - Graficas(@grafico_01-eda, @grafico_02-eda, @grafico_03-eda, @grafico_04-eda y @grafico_05-eda).

  
// == Modelo Baseline\
// *{En Proceso}*\
// + *Tipo:* RAG Multimodal, LoRA-Tuned.
// + *Configuración:* 03 Datasets:
//   - Multi-turn QA
//   - Simple QA
//   - visual-text QA
// + *Entregable:* `notebooks/02_baseline_model.ipynb`

// == Pruebas Iniciales\
// *{PENDIENTE DE REALIZAR}*\
// + *Protocolo:* Hold-out (80/20) sobre el set de preguntas y respuestas sintéticas.
// + Resultados por fold (si aplica) y media ± std. Ejemplo:
//   #table(
//     columns: (auto, auto, auto, auto, auto, auto, auto),
//     inset: 7pt,
//     align: center,
//     fill: (x, y) => if x == 6 { blue.lighten(90%) },
//     [*Métrica*], [*F1*], [*F2*], [*F3*], [*F4*], [*F5*], [*Media ± Std*],
//     [Accuracy], [0.78], [0.81], [0.80], [0.79], [0.82], [$0.80 plus.minus 0.015$],
//     [F1-score], [0.75], [0.80], [0.79], [0.77], [0.81], [$0.78 plus.minus 0.017$],
//   )
// + *Entregables:* Tabla de resultados, boxplot de la distribución de scores.
// + *Observaciones:* Identificación de outliers o folds problemáticos.


// == Avance de la Demo Interna\
// *{PENDIENTE DE REALIZAR}*\
// - *Qué se mostró:* Interfaz básica de consulta sobre un manual de chancadora.
// - *Feedback:*
//   - *Puntos fuertes:* Velocidad de respuesta.
//   - *Puntos a mejorar:* Precisión en la recuperación de códigos de parte en tablas.

// = Plan para siguiente semana

// + Revisión he implementación de algoritmo para detectar archivos que son manuales y documentos técnicos relevantes.
// + Revisión de integridad de archivos, se detecto archivos corructos que no se pueden abrir.
// + Iniciar con el planteamiento y estrategia de extracción de información con herramienta OCR.