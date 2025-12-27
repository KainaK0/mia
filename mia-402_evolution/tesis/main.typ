#set page(paper: "a4", margin: 2cm)
#set text(font: "Times New Roman", size: 11pt)

= 1. Introduction
Typst is a _markup-based_ typesetting system that is *powerful* and easy to learn.

== 1.1 Math Mode
Unlike LaTeX, you don't need backslashes for everything.
Inline math looks like this: $A = pi r^2$.

Block math is centered automatically if you add spaces:
// Add a space between e^(-x) and dx

$ F(x) = integral_0^infinity e^(-x) dif x $

== 1.2 Scripting
You can use variables and functions using the `#` symbol.

#let name = "Typst User"
Hello, #name!

// A simple loop
#for i in range(3) [
  - Item #i
]

== 1.3 New 