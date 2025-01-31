use core::panic;

use crate::{
    ast::{Constant, Expression, Identifier},
    lexer::{Keyword, LexicalAnalyzer, Literal, Operator, Token, TokenType},
};

pub type ParseResult<'a> = Result<(Expression, &'a [Token]), ParseError>;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedRemains(Vec<Token>),
}

use Keyword::*;
use Token as T;
use TokenType as TT;
//use TT::{Identifier as Id, Keyword as Kw, Separator as Sep};

fn parse<'a>(tokens: &'a [Token]) -> Result<Expression, ParseError> {
    let (expression, remains) = parse_expression(tokens, 0)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

fn parse_prefix<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    match &tokens {
        &[T(TT::Keyword(Let), ..), T(TT::Identifier(binding), ..), T(TT::Equals, ..), remains @ ..] => {
            parse_binding(binding, remains)
        }
        &[T(TT::Literal(Literal::Integer(x)), ..), remains @ ..] => {
            Ok((Expression::Literal(Constant::Int(*x)), remains))
        }
        &[T(TT::Identifier(id), ..), remains @ ..] => {
            Ok((Expression::Variable(Identifier::new(&id)), remains))
        }
        otherwise => {
            panic!("{otherwise:?}");
        }
    }
}

fn parse_binding<'a>(binding: &str, prefix: &'a [Token]) -> ParseResult<'a> {
    let (bound, remains) = parse_expression(prefix, 0)?;

    match &remains {
        &[T(TT::Keyword(In), ..), remains @ ..] => {
            let (body, remains) = parse_expression(remains, 0)?;

            Ok((
                Expression::Binding {
                    binding: Identifier::new(binding),
                    bound: bound.into(),
                    body: body.into(),
                },
                remains,
            ))
        }
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_expression<'a>(tokens: &'a [Token], precedence: usize) -> ParseResult<'a> {
    let (prefix, remains) = parse_prefix(tokens)?;
    parse_infix(prefix, remains, precedence)
}

fn parse_infix<'a>(lhs: Expression, tokens: &'a [Token], precedence: usize) -> ParseResult<'a> {
    match &tokens {
        &[T(TT::Keyword(Keyword::In), ..), ..] => Ok((lhs, tokens)),
        &[T(TT::End, ..), ..] => Ok((lhs, tokens)),

        &[T(TT::Operator(op), ..), remains @ ..] => {
            let op_precendence = op.precedence();
            if op_precendence > precedence {
                let (rhs, remains) = parse_expression(remains, op_precendence)?;
                let apply_op = make_binary_apply_op(*op, lhs, rhs);
                parse_infix(apply_op, remains, precedence)
            } else {
                Ok((lhs, tokens))
            }
        }

        &[_, ..] => parse_juxtaposition(lhs, tokens, precedence),
        _otherwise => Ok((lhs, tokens)),
    }
}

fn parse_juxtaposition(
    lhs: Expression,
    tokens: &[Token],
    precedence: usize,
) -> Result<(Expression, &[Token]), ParseError> {
    let (rhs, remains) = parse_prefix(tokens)?;
    parse_infix(
        Expression::Apply {
            function: lhs.into(),
            argument: rhs.into(),
        },
        remains,
        precedence,
    )
}

fn make_binary_apply_op(op: Operator, lhs: Expression, rhs: Expression) -> Expression {
    let apply_lhs = Expression::Apply {
        function: Expression::Variable(Identifier::new(&op.function_identifier())).into(),
        argument: lhs.into(),
    };
    Expression::Apply {
        function: apply_lhs.into(),
        argument: rhs.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Expression as E;
    use Identifier as Id;

    fn into_input(source: &str) -> Vec<char> {
        source
            .lines()
            .filter(|s| !s.trim().is_empty())
            .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .collect()
    }

    #[test]
    fn let_in_infix() {
        let mut lexer = LexicalAnalyzer::default();

        let expr = parse(lexer.parse(&into_input("let x = 10 in x + 20"))).unwrap();
        assert_eq!(
            E::Binding {
                binding: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Apply {
                    function: E::Apply {
                        function: E::Variable(Id::new(&Operator::Plus.function_identifier()))
                            .into(),
                        argument: E::Variable(Id::new("x")).into(),
                    }
                    .into(),
                    argument: E::Literal(Constant::Int(20)).into(),
                }
                .into(),
            },
            expr
        );
    }

    #[test]
    fn let_in_juxtaposed() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse(lexer.parse(&into_input("let x = 10 in f x 20"))).unwrap();
        assert_eq!(
            E::Binding {
                binding: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Apply {
                    function: E::Apply {
                        function: E::Variable(Id::new("f")).into(),
                        argument: E::Variable(Id::new("x")).into(),
                    }
                    .into(),
                    argument: E::Literal(Constant::Int(20)).into(),
                }
                .into(),
            },
            expr
        );
    }

    #[test]
    fn let_in_juxtaposed3() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse(lexer.parse(&into_input("let x = 10 in f x 1 2"))).unwrap();
        assert_eq!(
            E::Binding {
                binding: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Apply {
                    function: E::Apply {
                        function: E::Apply {
                            function: E::Variable(Id::new("f")).into(),
                            argument: E::Variable(Id::new("x")).into(),
                        }
                        .into(),
                        argument: E::Literal(Constant::Int(1)).into(),
                    }
                    .into(),
                    argument: E::Literal(Constant::Int(2)).into()
                }
                .into(),
            },
            expr
        );
    }

    #[test]
    fn let_in_with_mixed_artithmetics() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse(lexer.parse(&into_input("let x = f 1 2 in 3 * f 4 + 5"))).unwrap();
        assert_eq!(
            E::Binding {
                binding: Id::new("x"),
                bound: E::Apply {
                    function: E::Apply {
                        function: E::Variable(Id::new("f")).into(),
                        argument: E::Literal(Constant::Int(1)).into()
                    }
                    .into(),
                    argument: E::Literal(Constant::Int(2)).into()
                }
                .into(),
                body: E::Apply {
                    function: E::Apply {
                        function: E::Variable(Id::new("builtin::plus")).into(),
                        argument: E::Apply {
                            function: E::Apply {
                                function: E::Variable(Id::new("builtin::times")).into(),
                                argument: E::Literal(Constant::Int(3)).into()
                            }
                            .into(),
                            argument: E::Apply {
                                function: E::Variable(Id::new("f")).into(),
                                argument: E::Literal(Constant::Int(4)).into()
                            }
                            .into()
                        }
                        .into()
                    }
                    .into(),
                    argument: E::Literal(Constant::Int(5)).into()
                }
                .into()
            },
            expr
        );
    }

    #[test]
    fn nested_let_in() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse(lexer.parse(&into_input("let x = 10 in let y = 20 in x + y"))).unwrap();

        assert_eq!(
            E::Binding {
                binding: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Binding {
                    binding: Id::new("y"),
                    bound: E::Literal(Constant::Int(20)).into(),
                    body: E::Apply {
                        function: E::Apply {
                            function: E::Variable(Id::new("builtin::plus")).into(),
                            argument: E::Variable(Id::new("x")).into()
                        }
                        .into(),
                        argument: E::Variable(Id::new("y")).into()
                    }
                    .into()
                }
                .into()
            },
            expr
        );
    }
}
