= Parts of a language:

== Scanning
- Also known as *lexing* or *lexical analysis*.
- Chunk stream of characters in series of tokens.

#figure(caption: "Example of tokens")[
  #image("image/tokens.png")
]

== Parsing
- Build Abstract Syntax Tree from tokens.
- Where syntax gets a grammar.
- Syntax error appear here.

== Static Analysis
- Analysis of source code.
  - Binding or resolution:
    - Find definition for identifiers.
    - Resolve declaration with entities.
  #raw(
    "def greet():
    message = \"Hello\"
    print(message) # Bind message -> Hello

  greet() # Resolve greet to function",
    block: true,
    lang: "py",
  )
  - Type checking.
  - Control flow analysis. #footnote("This way compilers know code that will never execute.")
- Analysis may be stored in different ways:
  - In attributes of the AST.
  - Apart on a look up table (usually key-value)

== Intermediate representations (IR)
- Internal code used by compilers
- Usually hidden from the final user.
- Agnostic to both: source and destination.
- Facilitates portability to multiple architecture.
  #box(fill: rgb("#c9c9c9"), inset: 4pt)[
    *Example:* Write compilers from Pascal, C and Fortran targeting x86, ARM and SPARC.
    - Without IR: $3 text("langs") * 3 text("arch") = 9$ compilers
    - With IR: $3$ front ends to IR + 3 IR back end to architecture = 6 compilers.
  ]
- Also used for optimization.

== Optimization
- Constant folding.
- There are other techniques (not explained in the book btw...)

== Code generation
- Two ways to generate CPU instructions:
  + Create real CPU code:
    - Faster, but hard to do.
    - Tied to specific architecture.
  + Bytecode: #footnote("Named like this because each instruction is often a single byte long.")
    - Code for a hypothetical VM. (javac, pyc)
    - A more portable solution.
    - Machines do not read bytecode:
      - Build a VM.
      - Build simple mini compiler for each architecture.

== Runtime
- Some services are needed (Garbage collector, Hoisting, etc..)
- Runtime services to be stored somewhere.
  + In the VM.
  + In the executable code.

== Other parts of a language
=== Single pass compilers
- Mix parsing, analysis, and code generation in a single step.
- No allocation of AST or IR. #footnote("Pascal and C were designed around this limitations. This is why in C you can't call a function above the code, unless a forward declaration.")

=== Three walk interpreters

- Execute code after parsing of AST.
- Common for student projects.
- Not widely used, too slow.

=== Transpilers
- Produce string valid source code to other language

=== Just-in-time compilation (JIT)
- An advanced technique.
- Compile code to native code just before execution.

=== Compilers and interpreters
- Compiling:
  - Translating source language to other
  - User runs executable themselves.
- Interpreting:
  - Execute source code immediately.
