= Anexos:

== Sprint 1 - Modelo Baseline
===  Resumen Ejecutivo
En el sigueinte reporte dddd
Breve descripción del objetivo del sprint y el alcance alcanzado en el desarrollo del sistema RAG multimodal.

Para la implementación de Sistema de consulta Multimodal basado en RAG para Manuales de Mantenimiento en Plantas Concentradoras de Cobre, se realiza la solicitud y descarga de documentos provistos por el sponsor de manuales e información técnica.

===  Sprint Planning
*Objetivo del Sprint:* Establecer el flujo de ingesta y un modelo de recuperación base.

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 10pt,
  align: horizon,
  fill: (x, y) => if y == 0 { gray.lighten(80%) },
  [*ID*], [*Historia / Tarea*], [*Responsable*], [*Estado*],
  [1], [Solicitar la data de manuales de equipos - Planta Sulfuros], [Johan Callomamani], [Hecho],
  [2], [Descarga de manuales vía Aconex - Transmittals (5TB)], [Johan Callomamani], [En Proceso],
  [4], [Limpieza inicial de manuales (PDF/Imágenes)], [Johan Callomamani], [Por Empezar],
  [5], [Análisis de la data y vectorización ], [Johan Callomamani], [Por Empezar],
  [6], [Implementación de Baseline RAG], [Johan Callomamani], [Por Empezar],
)

== Data Pipeline Básico
+ *Descripción:* 
  - Procesamiento de transmittals descargados del Aconex.
  - Se realiza un análisis de los archivos generando una archivo csv con características de los archivos.
+ *Entregable:* 
   - Lista de archivos transmittal `aconex_file.zip`
   - `list_files.csv`
+ *Comentarios/Problemas:* Los documentos no son solo manuales, se tienen documentos adicionales, de no interés para el proyecto:
  - Información de ordenes de compra.
  - Solicitudes de transporte de componentes.
  - Documentos de aceptación de reportes y aceptación de proyectos.
  - Gantts de inspección/QA.
  - Invoices.
  - Monthly reports (avances de la etapa de proyecto).

== EDA Rápido
- *Hallazgos:*
  - Total de documentos 139474 archivos con un peso total 3.18 TB.
  - Alta densidad de documentos PDF con un total 78.53%.
  - Se encontró que no todos los archivos son manuales, el 22% son pagos y solo el 60% corresponde a información técnica relevante.

- *Visualizaciones:*.

// #figure(image("assets/eda_plots/Picture1.png", width: 80%), caption: [Distribución de tokens por manual]) <grafico-eda>
// #figure(image("assets/eda_plots/Picture2.png", width: 80%), caption: [Distribución de tokens por manual]) <grafico-eda>
- *Entregables:*
  - Graficas(Figura 1 y Figura 2).
  
== Modelo Baseline\
*{En Proceso}*\
+ *Tipo:* RAG Multimodal, LoRA-Tuned.
+ *Configuración:* 03 Datasets:
  - Multi-turn QA
  - Simple QA
  - visual-text QA
+ *Entregable:* `notebooks/02_baseline_model.ipynb`

== Pruebas Iniciales\
*{PENDIENTE DE REALIZAR}*\
+ *Protocolo:* Hold-out (80/20) sobre el set de preguntas y respuestas sintéticas.
+ Resultados por fold (si aplica) y media ± std. Ejemplo:
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    inset: 7pt,
    align: center,
    fill: (x, y) => if x == 6 { blue.lighten(90%) },
    [*Métrica*], [*F1*], [*F2*], [*F3*], [*F4*], [*F5*], [*Media ± Std*],
    [Accuracy], [0.78], [0.81], [0.80], [0.79], [0.82], [$0.80 plus.minus 0.015$],
    [F1-score], [0.75], [0.80], [0.79], [0.77], [0.81], [$0.78 plus.minus 0.017$],
  )
+ *Entregables:* Tabla de resultados, boxplot de la distribución de scores.
+ *Observaciones:* Identificación de outliers o folds problemáticos.


== Avance de la Demo Interna\
*{PENDIENTE DE REALIZAR}*\
- *Qué se mostró:* Interfaz básica de consulta sobre un manual de chancadora.
- *Feedback:*
  - *Puntos fuertes:* Velocidad de respuesta.
  - *Puntos a mejorar:* Precisión en la recuperación de códigos de parte en tablas.

= Plan para siguiente semana

+ Revisión he implementación de algoritmo para detectar archivos que son manuales y documentos técnicos relevantes.
+ Revisión de integridad de archivos, se detecto archivos corructos que no se pueden abrir.
+ Iniciar con el planteamiento y estrategia de extracción de información con herramienta OCR.