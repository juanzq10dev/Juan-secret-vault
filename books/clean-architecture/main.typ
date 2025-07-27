#let title = "Clean architecture"

#set page(
  paper: "us-letter",
  header: align(right, title),
  numbering: "1",
  columns: 1,
)

#set text(lang: "en")
#set par(justify: true)

#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
  #align(center, text(25pt)[
    *#title*
  ])

  #grid(
    columns: 1fr,
    align(center)[
      Juan Manuel Zurita Quinteros \
      #link("mailto:juanzq10dev@gmail.com")
    ]
  )

  #align(center)[
    #set par(justify: false)
    *Abstract* \
    This is a summary of the book Clean Architecture by Robert C. Martin.

  ]

]
#outline()

#show raw.where(block: false): box.with(fill: rgb("0909092f"), inset: 1pt)
#show raw.where(block: true): box.with(fill: rgb("0909092f"), inset: 5pt, width: 100%)

#pagebreak()
#include "00-introduction/page.typ"
#pagebreak()
#include "01/page.typ"
#pagebreak()
#include "02/page.typ"
#pagebreak()
#include "03/page.typ"