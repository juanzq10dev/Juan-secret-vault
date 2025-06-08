package jlox;

import java.util.List;

abstract class Expr{
 static class Binary extends Expr{
	Binary(Expr Left, Token operator, Expr right) {
		this.Left = Left;
		this.operator = operator;
		this.right = right;
		}

		final Expr Left;
		final Token operator;
		final Expr right;
	}
 static class Grouping extends Expr{
	Grouping(Expr Expression) {
		this.Expression = Expression;
		}

		final Expr Expression;
	}
 static class Literal extends Expr{
	Literal(Object value) {
		this.value = value;
		}

		final Object value;
	}
 static class Unary extends Expr{
	Unary(Token operator, Expr right) {
		this.operator = operator;
		this.right = right;
		}

		final Token operator;
		final Expr right;
	}
}
