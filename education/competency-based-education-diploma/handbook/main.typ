#let title = "Competency-Based Education Diploma"

#set page(
  paper: "us-letter",
  header: align(right, title),
  numbering: "1",
  columns: 2,
  flipped: true,
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
    This is a set of notes and documentation for competency-based education diploma

  ]

]
#outline()

#show raw.where(block: false): box.with(fill: rgb("0909092f"), inset: 1pt)
#show raw.where(block: true): box.with(fill: rgb("0909092f"), inset: 5pt, width: 100%)

#pagebreak()
#include "sections/module-01.typ"

// #cite(<promptForGenAI>, form: "prose"), 2020)
// #bibliography("works.bib", full: true)
