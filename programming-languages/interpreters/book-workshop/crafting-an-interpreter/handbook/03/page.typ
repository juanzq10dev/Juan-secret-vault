= Classes & prototypes

== Classes
- Based on instances and classes.

=== Instances
- Instances store the state for each object
- Have a reference to instance's class.

=== Classes
- Contain methods and inheritance chain.

#box(
  stroke: stroke(paint: blue, thickness: 3pt),
  fill: (rgb("0000ff20")),
  radius: 10%,
  inset: 4pt,
)[
  *Information:* To call a method on an
  instance, there is always a level of indirection. You look up the instance's class
  and then you find the method there:
]

#figure(caption: "Classes")[
  #image("images/classes.png")
]

== Prototypical inheritance
- Merges, interface and classes.
- There are only objects (no classes).
- Each object contains the state and methods.
- Directly delegate to (inherit) from each other.

#figure(caption: "Classes")[
  #image("images/prototypes.png")
]
