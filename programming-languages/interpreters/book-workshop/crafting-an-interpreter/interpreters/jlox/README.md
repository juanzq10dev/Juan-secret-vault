This is a compiler for lox language build from the book  `Crafting Interpreters by Robert Nystrom`

## Compiler usage
1. Ensure `app/build.gradle.kts` has `jlox.Lox` as mainClass
```
application {
    // Define the main class for the application.
    mainClass = "jlox.Lox"
}
```

2. Start the application
```bash
./gradlew build
./gradlew run
```

3. With the application running, you can run parts of the language

```
var a = 10;
```


## AST generator usage
1. Change `app/build.gradle.kts` mainClass to `tool.GenerateAst`
```
application {
    // Define the main class for the application.
    mainClass = "jlox.Lox"
}
```

2. Start the application
```bash
./gradlew build
./gradlew run --args="./src/main/java/jlox/"
```

This will generate `./src/main/java/jlox/Expr.java` a class with rules of the AST. 