= Módulo 01: Fundamentos del Currículo

#let light-gray = rgb("#d9d9d9")

Estas son unas citas de una charla introductoria de la doctora Mónica:
#block(fill: light-gray, inset: 5pt, width: 100%)[
  #align(center)["El conocimiento no se transmite (la pasión y la ética si pueden transmitirse), SE CONSTRUYE"]
]

#block(fill: light-gray, inset: 5pt, width: 100%)[
  #align(
    center,
  )["Los estudiantes no se dan cuenta que no viven un proceso formativo importante, ES NUESTRO DEBER COMO DOCENTES hacer que los estudiantes se den cuenta que están viviendo un proceso formativo importante."]
]

#block(fill: light-gray, inset: 5pt, width: 100%)[
  #align(
    center,
  )["Si la universidad responde netamente al mercado laboral estamos fregados. La Universidad también debe responder a la ciencia o el conocimiento."]
]

== Currículo

El currículo es el conjunto de acciones desarrolladas dentro de una organización con el objetivo de asegurar un aprendizaje de forma estructurada encaminado a la excelencia. Puede entenderse desde tres perspectivas:

- *Como contenido:* lo que se enseña
- *Como planificación:* cómo se organiza
- *Como relación interactiva:* la dinámica entre docentes y estudiantes

Se materializa a través de clases, experiencias y actividades.

=== Tipos de currículo

#table(
  columns: (auto, 1fr),
  [*Tipo*], [*Descripción*],
  [Oficial], [Contenido formal supervisado por los administradores],
  [Operacional], [La aplicación concreta y práctica del currículo oficial],
  [Oculto], [Lecciones no escritas formalmente que se aprenden en el proceso],
  [Nulo], [Experiencias y valores ausentes que el estudiante no percibe],
  [Extra currículo], [Actividades fuera del horario regular que complementan el oficial],
)

=== Niveles del currículo

- *Macro:* nivel sistémico/nacional (políticas educativas generales)
- *Meso:* nivel institucional (la escuela/universidad)
- *Micro:* nivel de aula (el docente y el estudiante directamente)

=== Componentes del currículo

El currículo se estructura en seis componentes interrelacionados que responden a preguntas clave del proceso educativo:

#table(
  columns: (auto, auto, 1fr),
  [*\#*], [*Componente*], [*Pregunta que responde*],
  [1], [Propósitos educativos], [¿Para qué enseñar?],
  [2], [Contenidos], [¿Qué enseñar?],
  [3], [Secuenciación], [¿Cuándo enseñar?],
  [4], [Metodología], [¿Cómo enseñar?],
  [5], [Recursos didácticos], [¿Con qué enseñar?],
  [6], [Evaluación], [¿Se cumplieron los propósitos?],
)


#page(flipped: true, columns: 1)[
  == Fundamentos filosóficos del currículo

  Cada corriente filosófica plantea una visión distinta sobre el hombre, la realidad y el conocimiento, lo que impacta directamente en cómo se concibe el currículo y la metodología educativa.
  #table(
    columns: (0.7fr, 1fr, 1fr, 1fr),
    [*Corriente*], [*Educación*], [*Currículo*], [*Metodología*],
    [Idealismo Platónico],
    [Llevar al estudiante a la virtud descubriendo la verdad a través de la Filosofía],
    [Verdades eternas e inmutables. \ Centrado en las Humanidades],
    [Transmisión magistral. \ El maestro es modelo a imitar],

    [Realismo Científico],
    [Comprender el orden natural de las cosas a través de las verdades científicas],
    [Centrado en las Ciencias naturales],
    [Magistral apoyada en textos y laboratorios. \ El maestro es experto en su materia],

    [Pragmatismo],
    [Proporcionar experiencias de aprendizaje y promover la cooperación],
    [Experiencias del alumno; materias según las necesidades del entorno],
    [Trabajo en grupo; el maestro guía al estudiante para que halle sus propias soluciones],

    [Existencialismo],
    [Formar al hombre como ser auténtico, exaltando la libertad personal],
    [No hay uno establecido; el alumno elige libremente lo que desea aprender],
    [El estudiante escoge cómo aprender; el maestro crea un ambiente de libertad],

    [Personalismo],
    [Despertar personas y promover el desarrollo personal integral],
    [Actividades planeadas para crear un ambiente propicio para el crecimiento personal],
    [Individual y grupal; el maestro actúa como orientador y guía],

    [Marxismo],
    [Preparar para el trabajo colectivo en función de la comunidad],
    [Ciencia aplicada a la práctica y a la solución de problemas sociales],
    [El estudio y el trabajo tienen igual valor; el maestro concientiza políticamente],

    [Posmodernidad],
    [Priorizar la diversidad, el relativismo y el respeto a las diferencias],
    [Sin estructura fija; flexibilidad curricular y pluralidad de enfoques],
    [No hay una predeterminada; todos los caminos son válidos],
  )

  #pagebreak()
  == Fundamentos psicológicos del currículo

  === Conductismo

  El aprendizaje es un cambio observable de conducta producido por estímulos externos. El ambiente moldea el comportamiento.

  #table(
    columns: (auto, 1fr, auto),
    [*Teoría*], [*Descripción*], [*Referente*],
    [Condicionamiento clásico],
    [Asociación *involuntaria pasiva* entre dos estímulos: uno neutro acaba generando la misma respuesta que el estímulo original],
    [Pavlov],

    [Condicionamiento operante],
    [Comportamiento *voluntario activo* que se fortalece o disminuye según sus consecuencias: refuerzo positivo, negativo o castigo],
    [Skinner],

    [Aprendizaje vicario], [Se aprende observando e imitando a otros, sin necesidad de experiencia directa], [Bandura],
  )

  #figure(caption: "Condicionamiento clásico vs operante")[
    #image("../images/module-01/condicionamiento.png", fit: "contain", height: 60%)
  ]

]


=== Humanismo

- El ser humano es activo, libre y capaz de dirigir su propio desarrollo.
- El fin de la educación es la autorrealización y el desarrollo integral del ser humano.
- El aprendizaje debe ser relevante, vivencial y con sentido personal.
- *Pedagogía centrada en el alumno:* El estudiante es el protagonista.
- *El docente es un facilitador del aprendizaje*, su tarea es despertar en el alumno el deseo de aprender.

*Necesidades de Maslow y educación*
- Un estudiante no puede aprender eficazmente si sus necesidades básicas no están cubiertas.

#figure()[
  #image("../images/module-01/piramide_maslow.png", height: 50%)
]

En base a esta pirámide el docente debe:
- Garantizar un ambiente seguro.
- Fomentar la pertenencia y el trabajo.
- Reconocer logros.
- Estimular autoestima
- entre otras cosas
