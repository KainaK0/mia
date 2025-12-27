#import "lib.typ": project

// Importamos los metadatos desde el archivo TOML
#let meta = toml("metadata.toml")

// Aplicamos la plantilla
#show: doc => project(meta, doc)

// Importamos las secciones en orden
// #include "sections/1-general.typ"
// #pagebreak()
#include "sections/2-problema.typ"
#include "sections/3-marco.typ"
#include "sections/4-metodologia.typ"
#include "sections/5-admin.typ"
#include "sections/6-referencias.typ"

