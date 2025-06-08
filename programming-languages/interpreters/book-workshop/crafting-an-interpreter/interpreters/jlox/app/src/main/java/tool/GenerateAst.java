package tool;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

public class GenerateAst {
    /* 
     * This is a class created to automatically build the AST 
     * It is not part of the compiler.
    */

    public static void main(String[] args) throws IOException{
        if (args.length != 1) {
            System.err.println("Usage: generate_ast <output directory>");
            System.exit(64);
        }
        String outputDir = args[0];
        defineAst(outputDir, "Expr", Arrays.asList(
            /*  This comes from AST defined on page 63 of the book
             *      Binary   -> expression operator expression
             *      Grouping -> "(" expression ")"
             *      Unary    -> ( "-" | "!" ) expression 
             * 
             */
            "Binary     : Expr Left, Token operator, Expr right",
            "Grouping   : Expr Expression",
            "Literal    : Object value",
            "Unary      : Token operator, Expr right" 
        ));
    }


    private static void defineAst(
        String outputDir, String baseName, List<String> types
    ) throws IOException {
        String path = outputDir + "/" + baseName +  ".java";
        PrintWriter writer = new PrintWriter(path, "UTF-8");
        String outputDirPackage = "package jlox;";

        writer.println(outputDirPackage);
        writer.println();
        writer.println("import java.util.List;");
        writer.println();
        writer.println("abstract class " + baseName + "{");

        for (String type: types) {
            String className = type.split(":")[0].trim();
            String fields = type.split(":")[1].trim();
            defineType(writer, baseName, className, fields);
        }

        writer.println("}");
        writer.close();
    }


    private static void defineType(
        PrintWriter writer, String baseName, String className, String fieldList 
    ) {
        writer.println(" static class " + className + " extends " + baseName + "{");

        // Constructor
        writer.println("\t" + className + "(" + fieldList + ") {");

        // Store parameters in fields
        String[] fields = fieldList.split(", ");
        for (String field: fields) {
            String name = field.split(" ")[1];
            writer.println("\t\t" + "this." + name + " = " + name + ";");
        }

        writer.println("\t\t" + "}");

        // Fields.
        writer.println();
        for (String field: fields) {
            writer.println("\t\t" + "final " + field + ";");
        }

        writer.println("\t" + "}");
    }
    
}
