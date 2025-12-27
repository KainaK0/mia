// IMPORTANTE: Como main.typ está en "sections/", debemos salir (..) para buscar lib y toml
#import "../lib.typ": project
#let meta = toml("../metadata.toml")

#show: doc => project(meta,doc)

// CORRECCIÓN: Quitamos "sections/" porque ya estamos dentro de esa carpeta
// #include "1-general.typ"
#pagebreak()
#include "2-problema.typ"
#include "3-marco.typ"
#include "4-metodologia.typ"
#include "5-admin.typ"