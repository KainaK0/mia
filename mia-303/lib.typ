#let project(meta, body) = {
  // 1. Configuración global
  set document(title: meta.info.title, author: meta.info.student)
  set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 3cm, right: 2.5cm),
  )
  
  // CORRECCIÓN DE FUENTE: Linux Libertine viene incluida en Typst y es igual a Times.
  set text(
    font: "Times New Roman", 
    size: 12pt,
    lang: "es"
  )
  
  set par(justify: true)

  // 2. Generación de la Portada
  align(center)[
    #text(weight: "bold", size: 14pt)[#meta.institution.name] \
    #v(0.5em)
    #text(size: 12pt)[#meta.institution.faculty]
    
    #v(1cm)
    
    // CORRECCIÓN DE LOGO:
    // Solo mostramos la imagen si la variable 'has_logo' en metadata.toml es true.
    #if meta.project.has_logo == true {
      image("assets/logo_uni.svg", width: 7cm)
    } else {
      v(4cm) // Espacio en blanco si no hay logo
    }
    
    #v(1cm)
    
    #text(weight: "bold", size: 16pt)[PLAN DE TESIS]
    #v(1cm)
    #text(weight: "bold", size: 14pt)[“#meta.info.title”]
    #v(1.5cm)
    #text(size: 12pt)[PARA OBTENER EL GRADO ACADEMICO DE #meta.info.degree]
    
    #v(2cm)
    
    #align(center)[
      #text(weight: "bold")[ELABORADA POR:] \
      #meta.info.student
    ]
    
    #v(1cm)
    
    #align(center)[
      #text(weight: "bold")[ASESOR:] #meta.info.advisor
    ]
    
    #v(1fr)
    
    #meta.info.location \
    #meta.info.year
  ]
  
  pagebreak()

  // 3. Configuración de Capítulos
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => [
    #block(above: 2em, below: 1em)[
      Capítulo #counter(heading).display(): #it.body
    ]
  ]

  body
}