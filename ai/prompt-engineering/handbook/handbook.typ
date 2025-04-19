#align(center, text(25pt)[
  *Prompt Engineering*
])

#grid(
  columns: (1fr),
  align(center)[
    Juan Manuel Zurita Quinteros \
    #link("mailto:juanzq10dev@gmail.com")
  ]
)

#align(center)[
  #set par(justify: false)
  *Abstract* \
  This is a set of notes in order to learn about prompt engineering
]

#show raw.where(block: false): box.with(fill: rgb("0909092f"), inset: 1pt)
#show raw.where(block: true): box.with(fill: rgb("0909092f"), inset: 5pt, width: 100%)

#include "01-principles/page.typ"

#include "02-chunking-text/page.typ"

// #cite(<promptForGenAI>, form: "prose"), 2020)
#bibliography("works.bib", full: true)
