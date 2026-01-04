// // IMPORTANTE: Como main.typ está en "sections/", debemos salir (..) para buscar lib y toml
// #import "../lib.typ": project
// #let meta = toml("../metadata.toml")

// #show: doc => project(meta,doc)

// // CORRECCIÓN: Quitamos "sections/" porque ya estamos dentro de esa carpeta
// // #include "1-general.typ"
// #pagebreak()
// #include "2-problema.typ"
// #include "3-marco.typ"
// #include "4-metodologia.typ"
// #include "5-admin.typ"
// 
#let meta = toml("../metadata.toml")
*DATOS GENERALES*

+ *TITULO DEL PLAN DE TESIS:*\ #meta.info.title\
+ *NOMBRE DE AUTOR:*\ #meta.info.student\
+ *NOMBRE DEL ASESOR O ASESORES:*\ "..."\
+ *AREA INVOLUCRADA: UNIDAD DE POSGRADO FIIS:*\ #meta.institution.faculty\
+ *LUGAR DONDE SE DESARROLLA EL PROYECTO:*\ #meta.project.place\
+ *DURACION ESTIMADA:*\ #meta.project.duration

#pagebreak()