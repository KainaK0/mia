= Administración del plan de tesis

== Cronograma

#let x = align(center + horizon)[*X*]

#align(center)[
  #text(1.5em, weight: "bold")[Cronograma de Actividades 2025-2026]
  #v(1em)
  
  #table(
    columns: (auto, ..range(14).map(_ => 1fr)), // 1 columna auto + 14 columnas iguales
    align: (col, row) => (
      if col == 0 { left + horizon } else { center + horizon }
    ),
    stroke: 0.5pt + black,
    inset: 8pt,

    // --- Encabezados ---
    // Fila 1: Años
    [], 
    table.cell(colspan: 2, fill: luma(240))[2025], 
    table.cell(colspan: 12, fill: luma(230))[2026],

    // Fila 2: Meses
    table.cell(fill: luma(220))[*Actividades*],
    [Nov], [Dic], // 2025
    [Ene], [Feb], [Mar], [Abr], [May], [Jun], [Jul], [Ago], [Set], [Oct], [Nov], [Dic], // 2026

    // --- Datos / Marcas ---
    
    // Actividad 1 (Nov, Dic)
    [actividad 1 asfadfasdfa sd], x, x, [], [], [], [], [], [], [], [], [], [], [], [],

    // Actividad 2 (Dic, Ene, Feb, Mar)
    [actividad 2 asdasdfasdddd dddddd], [], x, x, x, x, [], [], [], [], [], [], [], [], [],

    // Actividad 3 (Feb, Mar, Abr)
    [actividad 3 adfasdfasd fasdfasd fasd], [], [], [], x, x, x, [], [], [], [], [], [], [], [],

    // Actividad 4 (Mar, Abr, May, Jun)
    [actividad 4 adfasdf fasdfasdf afadfasdf], [], [], [], [], x, x, x, x, [], [], [], [], [], [],

    // Actividad 5 (May, Jun, Jul, Ago)
    [actividad 5], [], [], [], [], [], [], x, x, x, x, [], [], [], [],

    // Actividad 6 (Jun, Jul, Ago, Set)
    [actividad 6], [], [], [], [], [], [], [], x, x, x, x, [], [], [],

    // Actividad 7 (Ago, Set, Oct)
    [actividad 7], [], [], [], [], [], [], [], [], [], x, x, x, [], [],

    // Actividad 8 (Set, Oct, Nov, Dic)
    [actividad 8], [], [], [], [], [], [], [], [], [], [], x, x, x, x,
  )
]

== Presupuesto


== Financiamiento

// Desactivamos numeración para lo final
#set heading(numbering: none)

// = Referencias Bibliográficas
// Listado de referencias según normas APA o IEEE.

= Anexos
Matriz de consistencia