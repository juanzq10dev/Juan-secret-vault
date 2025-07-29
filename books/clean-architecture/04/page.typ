= Structured Programming
Structured programming was born by Dijkstra:

- Dijkstra wanted to have proofs from programming (as there are proofs on maths).
- Dijkstra realized that some `goto` statements did not allow code to be breakable in testable units of code.
- Dijkstra considered `goto` statements could be replaced by loops and if sentences based on Böhm and Jacopini discovery #footnote("Look at key quotes / passages sections").

He wrote a paper "Go to statement considered harmful" that talks about that, and he had the reason: nowadays structured programming is the standard and modern languages do not include `goto` statements.

Dijkstra wanted programming to be like math: in math we proof that something is true. But software now is more like science: in science we proof that something is feasible by proving it is not false, but we cannot ensure that something is true for all cases.

*Key concepts:*
- *Structured programming:* it is a programming paradigm that aimed to improve code by using well-defined control structures (functions, loops, conditionals, etc.).

*Key quotes / passages:*
- "Böhm and Jacopini proved that programming can be constructed from just three structures: sequence, selection and iteration"
- "Software is like a science, we show correctness by failing to prove incorrectness".
- "Testing shows the presence, not the absence of bugs" (Dijkstra).
