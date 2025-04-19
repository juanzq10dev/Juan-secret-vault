= Introduction
- Prompt engineering is the process of discovering prompts that reliably yield useful or desired results. 
- The *prompt* serves as a set of instructions the model uses to predict the desired response. 

= Principles of prompt engineering
+ *Give direction:*
  - If possible, include what context a human might need for this task. 
  - Use role playing:
  #raw("You are senior software developer your task is to... ", block: true)
  - Use prewarming (also called internal retrieval). #footnote[Technique of priming a language model with context or examples before the actual task prompt is given] 
  #box(
    inset: 5pt,
    stroke: stroke(paint: red, thickness: 3pt),
    fill: rgb("ff000030"),
    width: 100%,
    radius: 25%
  )[*Warning:* Adding to much directions may lead to a conflictive combination the model cannot resolve   ]

+ *Specify format.*

+ *Provide examples.*
  - Three to five examples are good enough. 
+ *Evaluate quality.*
+ *Divide labor.*
  - Divide a task into multiple steps. 

