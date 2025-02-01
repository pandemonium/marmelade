use core::panic;

use crate::{
    ast::{Declaration, Expression, Identifier, Parameter, ValueDeclarator},
    lexer::{Keyword, Layout, Location, Operator, Token, TokenType},
};

pub type ParseResult<'a, A> = Result<(A, &'a [Token]), ParseError>;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedRemains(Vec<Token>),
}

use Keyword::*;
use Token as T;
use TokenType as TT;

pub fn parse_decl<'a>(input: &'a [Token]) -> ParseResult<'a, Declaration> {
    match input {
        [T(TT::Identifier(id), ..), T(TT::Equals, ..), remains @ ..] => {
            parse_value_decl(id, remains)
        }
        _otherwise => todo!(),
    }
}

fn parse_value_decl<'a>(id: &str, input: &'a [Token]) -> ParseResult<'a, Declaration> {
    match input {
        [T(TT::Keyword(Keyword::Fun), ..), remains @ ..] => {
            let (parameters, remains) = parse_params(remains)?;
            if starts_with(TokenType::Arrow, remains) {
                let (body, remains) = parse_expression(&remains[1..], 0)?;
                Ok((
                    Declaration::Value {
                        binding: Identifier::new(&id),
                        declarator: ValueDeclarator::Function {
                            parameters,
                            return_type_annotation: None,
                            body,
                        },
                    },
                    remains,
                ))
            } else {
                Err(todo!())
            }
        }
        _otherwise => todo!(),
    }
}

fn parse_params<'a>(remains: &'a [Token]) -> ParseResult<'a, Vec<Parameter>> {
    let (params, remains) =
        if let Some(end) = remains.iter().position(|t| t.token_type() == &TT::Arrow) {
            (&remains[..end], &remains[end..])
        } else {
            todo!()
        };

    let make_param = |t: &Token| {
        if let TT::Identifier(id) = t.token_type() {
            Some(Parameter {
                name: Identifier::new(id),
                type_annotation: None,
            })
        } else {
            None
        }
    };

    match params.iter().map(make_param).collect::<Option<_>>() {
        Some(params) => Ok((params, remains)),
        _otherwise => Err(todo!()),
    }
}

pub fn parse_expr_phrase<'a>(tokens: &'a [Token]) -> Result<Expression, ParseError> {
    let (expression, remains) = parse_expression(tokens, 0)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

fn parse_prefix<'a>(tokens: &'a [Token]) -> ParseResult<'a, Expression> {
    match tokens {
        [T(TT::Keyword(Let), position), T(TT::Identifier(binder), ..), T(TT::Equals, ..), remains @ ..] => {
            parse_binding(position.clone(), binder, remains)
        }
        [T(TT::Literal(literal), ..), remains @ ..] => {
            Ok((Expression::Literal(literal.clone().into()), remains))
        }
        [T(TT::Identifier(id), ..), remains @ ..] => {
            Ok((Expression::Variable(Identifier::new(&id)), remains))
        }
        // Newline here
        otherwise => {
            panic!("{otherwise:?}");
        }
    }
}

fn starts_with<'a>(tt: TokenType, prefix: &'a [Token]) -> bool {
    matches!(&prefix, &[t, ..] if t.token_type() == &tt)
}

fn strip_if_starts_with<'a>(tt: TokenType, prefix: &'a [Token]) -> &'a [Token] {
    if matches!(&prefix, &[t, ..] if t.token_type() == &tt) {
        &prefix[1..]
    } else {
        prefix
    }
}

fn strip_first_if<'a>(condition: bool, input: &'a [Token]) -> &'a [Token] {
    if condition {
        &input[1..]
    } else {
        input
    }
}

fn parse_binding<'a>(
    position: Location,
    binder: &str,
    input: &'a [Token],
) -> ParseResult<'a, Expression> {
    let indented = starts_with(TokenType::Layout(Layout::Indent), input);
    let (bound, remains) = parse_expression(strip_first_if(indented, input), 0)?;

    let dedented = starts_with(TokenType::Layout(Layout::Dedent), remains);
    match strip_first_if(indented && dedented, remains) {
        [T(TT::Keyword(In), ..), remains @ ..] => {
            let (body, remains) = parse_expression(
                strip_if_starts_with(TokenType::Layout(Layout::Indent), remains),
                0,
            )?;

            Ok((
                Expression::Binding {
                    binder: Identifier::new(binder),
                    bound: bound.into(),
                    body: body.into(),
                    postition: position,
                },
                remains,
            ))
        }
        // In could be offside here, then what? What can I return or do?
        // Well, that is an error then I guess.
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_expression<'a>(tokens: &'a [Token], precedence: usize) -> ParseResult<'a, Expression> {
    let (prefix, remains) = parse_prefix(tokens)?;
    // An initial Indent could occur here, right?
    parse_infix(prefix, remains, precedence)
}

// Infixes end with End and In
fn parse_infix<'a>(
    lhs: Expression,
    input: &'a [Token],
    precedence: usize,
) -> ParseResult<'a, Expression> {
    let is_done = |t: &Token| {
        matches!(
            t.token_type(),
            TT::Keyword(Keyword::In) | TT::End | TT::Layout(Layout::Dedent)
        )
    };

    // Operators can be prefigured by Layout::Newline
    // Juxtapositions though? I would have to be able to ask the Expression
    // about where it started
    match input {
        [t, ..] if is_done(t) => Ok((lhs, input)),

        [T(TT::Operator(op), ..), remains @ ..] => {
            parse_operator(lhs, input, precedence, op, remains)
        }

        [t, T(TT::Operator(op), ..), remains @ ..]
            if t.token_type() == &TT::Layout(Layout::Newline)
                || t.token_type() == &TT::Layout(Layout::Indent) =>
        {
            parse_operator(lhs, input, precedence, op, remains)
        }

        [t, remains @ ..]
            if t.token_type() == &TT::Layout(Layout::Newline)
                || t.token_type() == &TT::SemiColon =>
        {
            let (and_then, remains) = parse_expression(remains, precedence)?;
            Ok((
                Expression::Sequence {
                    this: lhs.into(),
                    and_then: and_then.into(),
                },
                remains,
            ))
        }

        [t, remains @ ..] if t.token_type() == &TT::Layout(Layout::Indent) => {
            parse_juxtaposition(lhs, remains, precedence)
        }

        [_, ..] => parse_juxtaposition(lhs, input, precedence),

        _otherwise => Ok((lhs, input)),
    }
}

fn parse_operator<'a>(
    lhs: Expression,
    input: &'a [Token],
    context_precedence: usize,
    operator: &Operator,
    remains: &'a [Token],
) -> ParseResult<'a, Expression> {
    let this_precedence = operator.precedence();
    if this_precedence > context_precedence {
        let (rhs, remains) = parse_expression(remains, this_precedence)?;
        parse_infix(
            apply_binary_operator(*operator, lhs, rhs),
            remains,
            context_precedence,
        )
    } else {
        Ok((lhs, input))
    }
}

fn parse_juxtaposition<'a>(
    lhs: Expression,
    tokens: &'a [Token],
    precedence: usize,
) -> ParseResult<'a, Expression> {
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

fn apply_binary_operator(op: Operator, lhs: Expression, rhs: Expression) -> Expression {
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
    use crate::{
        ast::{Constant, ValueDeclarator},
        lexer::LexicalAnalyzer,
    };

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

        let expr = parse_expr_phrase(lexer.tokenize(&into_input("let x = 10 in x + 20"))).unwrap();
        assert_eq!(
            E::Binding {
                postition: Location::default(),
                binder: Id::new("x"),
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
        let expr = parse_expr_phrase(lexer.tokenize(&into_input("let x = 10 in f x 20"))).unwrap();
        assert_eq!(
            E::Binding {
                postition: Location::default(),
                binder: Id::new("x"),
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
        let expr = parse_expr_phrase(lexer.tokenize(&into_input("let x = 10 in f x 1 2"))).unwrap();
        assert_eq!(
            E::Binding {
                postition: Location::default(),
                binder: Id::new("x"),
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
        let expr = parse_expr_phrase(lexer.tokenize(&into_input(
            r#"|let x =
               |    print_endline "Hello, world"; f 1 2
               |in
               |    3 * f
               |           4 + 5"#,
        )))
        .unwrap();
        assert_eq!(
            E::Binding {
                postition: Location::default(),
                binder: Id::new("x"),
                bound: E::Sequence {
                    this: E::Apply {
                        function: E::Variable(Id::new("print_endline")).into(),
                        argument: E::Literal(Constant::Text("Hello, world".to_owned())).into(),
                    }
                    .into(),
                    and_then: E::Apply {
                        function: E::Apply {
                            function: E::Variable(Id::new("f")).into(),
                            argument: E::Literal(Constant::Int(1)).into()
                        }
                        .into(),
                        argument: E::Literal(Constant::Int(2)).into()
                    }
                    .into()
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
        let expr =
            parse_expr_phrase(lexer.tokenize(&into_input("let x = 10 in let y = 20 in x + y")))
                .unwrap();

        assert_eq!(
            E::Binding {
                postition: Location::default(),
                binder: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Binding {
                    postition: Location::new(1, 15),
                    binder: Id::new("y"),
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

    #[test]
    fn function_binding() {
        let mut lexer = LexicalAnalyzer::default();
        let (decl, _) =
            parse_decl(lexer.tokenize(&into_input(r#"|create_window = fun x -> x"#))).unwrap();

        assert_eq!(
            Declaration::Value {
                binding: Identifier::new("create_window"),
                declarator: ValueDeclarator::Function {
                    parameters: vec![Parameter {
                        name: Identifier::new("x"),
                        type_annotation: None
                    }],
                    return_type_annotation: None,
                    body: Expression::Variable(Identifier::new("x")),
                }
            },
            decl
        );
    }
}
