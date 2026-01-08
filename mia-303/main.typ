#import "lib.typ": project

// Importamos los metadatos desde el archivo TOML
#let meta = toml("metadata.toml")

// Aplicamos la plantilla
#show: doc => project(meta, doc)
 
// --- AQUÍ SE AGREGA LA TABLA DE CONTENIDO ---
#outline(
  title: "Índice General", // Título personalizado
  indent: auto,            // Indenta subsecciones (1.1, 1.1.1)
  depth: 4                 // (Opcional) Qué tan profundo mostrar niveles
)
#pagebreak()               // Salto de página después del índice
                           // 
// Importamos las secciones en orden
// #include "sections/1-general.typ"
// #pagebreak()
#include "sections/1-general.typ"
#include "sections/2-problema.typ"
#include "sections/3-marco.typ"
#include "sections/4-metodologia.typ"
#include "sections/5-admin.typ"
// #include "sections/7-anexo.typ"
#include "sections/6-referencias.typ"

