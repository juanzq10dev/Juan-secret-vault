= Chunking text
- Chunking is the process of breaking down large pieces of text into smaller, more manageable units or chunks.

== Benefits of chunking
- Fitting the given context length.
- Improved performance.
- Increased flexibility.

== When to chunk
- Large documents. 
- Complex analysis.
- Multi-topic document.

== Disadvantages of chunking
Poor chunking may lead to:
- Loss of context.
- Increased processing load.
- Difficulty understanding main ideas or themes of the text. 
- Struggling to generate accurate summaries or translations. 

== Chunking strategies
#figure(
  caption: [Six chunking strategies highlighting their advantages and disadvantages. 
  
  Source: #cite(<promptForGenAI>, form: "prose")],
)[
#table(
  columns: 3,
  [*Splitting strategy*], [*Advantages*], [*Disadvantages*],

  [By sentence ], [Preserves context, suitable for various tasks], [May not be efficient for very long context],
  [By paragraph], [Handles longer content, focuses on cohesive units], [Less granularity, may miss subtitle connections], 
  [By topic], [Identifies main themes, better for classification], [Requires topic identification, may miss fine details],
  [By complexity], [Groups similar complexity levels, adaptive], [Requires complexity measurement, not suitable for all tasks],
  [By length], [Manages very long content, efficient processing], [Loss of context, may require more processing steps],
  [By tokens], [Accurate token counts, which helps avoiding LLM prompt token limits], [Requires tokenization, may increase computational complexity],
)]

#box(
  inset: 5pt,
  stroke: stroke(paint: blue, thickness: 3pt),
  fill: rgb("0000ff20"),
  width: 100%,
  radius: 25%
)[*Information:* Some strategies imply sentence detection using NLP, to improve chunk accuracy. ]

== Sliding Window Chunking
A technique used to process long sequences by moving a fixed size window over the data, analyzing or transforming one segment at a time.

Given: 
- A sequences (tokens, words, or time steps).
- A fixed window size *W*.
- A stride *S*

The sliding window moves across the sequence from left to right, extracting overlapping windows.

#raw("Example: (Window Size = 3, Stride = 1):
  Sequence: (For example chunks of a long file)
    [A, B, C, D, E]

  Windows:
    [A, B, C]
    [B, C, D]
    [C, D, E]
", block: true)


#box(
  stroke: stroke(paint: orange, thickness: 3pt), 
  fill: (rgb("ff000020")),
  radius: 10%,
  inset: 4pt
  )[
*Be careful:*
Window size may be increased or decreased:
- *Short window size* minimizes loss of context, but requires a lot of processing load: There is a lot of data repeated across windows. 
- *Long window size* may cause loss of context, but reduces the number of requests to the LLM.
]