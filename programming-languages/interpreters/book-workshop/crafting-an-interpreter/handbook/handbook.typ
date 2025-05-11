#set page(
  paper: "us-letter",
  header: align(right, "Crafting interpreters (notes)"),
  numbering: "1",
  columns: 2,
)

#set text(lang: "en")
#set par(justify: true)

#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
  #align(
    center,
    text(25pt)[
      *Crafting Interpreters (notes)*
    ],
  )

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
    This is a set of notes from book 'Crafting an interpreter' by Robert Nystrom

    #outline()
  ]

]



#show raw.where(block: false): box.with(fill: rgb("0909092f"), inset: 1pt)
#show raw.where(block: true): box.with(fill: rgb("0909092f"), inset: 5pt, width: 100%)

#include "02/page.typ"

#include "03/page.typ"

#bibliography("works.bib", full: true)
