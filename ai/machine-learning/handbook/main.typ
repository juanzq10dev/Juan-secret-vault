#let title = "Machine Learning"

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
    This is a set of notes in order to learn about machine learning

  ]

]
#outline()

#show raw.where(block: false): box.with(fill: rgb("0909092f"), inset: 1pt)
#show raw.where(block: true): box.with(fill: rgb("0909092f"), inset: 5pt, width: 100%)

#include "01-introduction/page.typ"
#include "02-univariate-regression-model/page.typ"
#include "03-multiple-linear-regression/page.typ"

#include "04-classification/page.typ"
#include "05-overfitting/page.typ"

#pagebreak()
#include "06-neural-networks/page.typ"

#pagebreak()
#include "07-multiclass-classification/page.typ"

#pagebreak()
#include "08-testing-ml-models/page.typ"

#pagebreak()
#include "09-ml-development-process/page.typ"

#pagebreak()
#include "10-skewed-dataset/page.typ"

#pagebreak()
#include "11-binary-trees/page.typ"

#pagebreak()
#include "12-clustering/page.typ"

#pagebreak()
#include "13-anomaly-detection/page.typ"

#pagebreak()
#include "14-recommender-systems/page.typ"


// #cite(<promptForGenAI>, form: "prose"), 2020)
// #bibliography("works.bib", full: true)
