use core::panic;

use crate::{
    ast::{
        CompilationUnit, ConstantDeclarator, ControlFlow, Declaration, Expression,
        FunctionDeclarator, Identifier, ModuleDeclarator, Parameter, ValueDeclarator,
    },
    lexer::{Keyword, Layout, Operator, SourcePosition, Token, TokenType},
};

pub type ParseResult<'a, A> = Result<(A, &'a [Token]), ParseError>;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedRemains(Vec<Token>),
    ExpectedTokenType(TokenType),
    DeclarationOffside(SourcePosition, Vec<Token>),
}

use Keyword::*;
use Token as T;
use TokenType as TT;

pub fn parse_compilation_unit<'a>(input: &'a [Token]) -> Result<CompilationUnit, ParseError> {
    let (declarations, ..) = parse_declarations(input)?;

    Ok(CompilationUnit::Implicit(ModuleDeclarator {
        position: SourcePosition::default(),
        name: Identifier::new("main"),
        declarations,
    }))
}

pub fn parse_declarations<'a>(input: &'a [Token]) -> ParseResult<'a, Vec<Declaration>> {
    let mut declarations = Vec::default();
    let mut input = input;

    loop {
        let (declaration, remains) = parse_declaration(input)?;
        let current_block = *declaration.position();
        declarations.push(declaration);

        input = find_next_in_block(current_block, remains, |token_type| {
            matches!(token_type, TT::Identifier(..) | TT::End)
        })?;

        if matches!(input, [T(TT::Layout(Layout::Dedent) | TT::End, ..), ..]) {
            break Ok((declarations, remains));
        }
    }
}

pub fn find_next_in_block<'a>(
    current_block: SourcePosition,
    input: &'a [Token],
    is_next_start_token_type: fn(&TT) -> bool,
) -> Result<&'a [Token], ParseError> {
    let mut input = input;
    loop {
        match input {
            [T(TT::Layout(Layout::Dedent | Layout::Newline), position), remains @ ..] => {
                if position.is_same_block(&current_block) {
                    break Ok(remains);
                } else {
                    input = remains
                }
            }
            remains @ [t, ..] if is_next_start_token_type(t.token_type()) => {
                // This end-check is... bad. Should I get rid of End?
                if t.location().is_same_block(&current_block) || t.token_type() == &TT::End {
                    break Ok(remains);
                } else {
                    break Err(ParseError::DeclarationOffside(
                        *t.location(),
                        remains.to_vec(),
                    ));
                }
            }
            otherwise => break Err(ParseError::UnexpectedRemains(otherwise.to_vec())),
        }
    }
}

pub fn parse_declaration<'a>(input: &'a [Token]) -> ParseResult<'a, Declaration> {
    match input {
        [T(TT::Identifier(id), pos), T(TT::Equals, ..), remains @ ..] => {
            parse_value_binding(id, pos, remains)
        }
        [t, ..] => Err(ParseError::UnexpectedToken(t.clone())),
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_value_binding<'a>(
    binder: &String,
    position: &SourcePosition,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration> {
    let (declarator, remains) =
        parse_value_declarator(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;

    Ok((
        Declaration::Value {
            binder: Identifier::new(&binder),
            declarator,
            position: position.clone(),
        },
        remains,
    ))
}

// These can have type annotations
// let foo :: Int -> String -> String = fun i s -> s
fn parse_value_declarator<'a>(input: &'a [Token]) -> ParseResult<'a, ValueDeclarator> {
    match input {
        [T(TT::Keyword(Fun), ..), remains @ ..] => {
            let (parameters, remains) = parse_parameter_list(remains)?;
            if starts_with(TokenType::Arrow, remains) {
                let (body, remains) = parse_expression(
                    strip_if_starts_with(TT::Layout(Layout::Indent), &remains[1..]),
                    0,
                )?;
                Ok((
                    ValueDeclarator::Function(FunctionDeclarator {
                        parameters,
                        return_type_annotation: None,
                        body,
                    }),
                    remains,
                ))
            } else {
                Err(ParseError::ExpectedTokenType(TT::Arrow))
            }
        }
        remains @ [..] => {
            let (initializer, remains) =
                parse_expression(strip_if_starts_with(TT::Layout(Layout::Indent), remains), 0)?;
            Ok((
                ValueDeclarator::Constant(ConstantDeclarator {
                    initializer,
                    type_annotation: None,
                }),
                remains,
            ))
        }
    }
}

// Should this function eat the -> ?
// a | pattern
fn parse_parameter_list<'a>(remains: &'a [Token]) -> ParseResult<'a, Vec<Parameter>> {
    let (params, remains) =
        // This pattern is quite common...
        if let Some(end) = remains.iter().position(|t| t.token_type() == &TT::Arrow) {
            (&remains[..end], &remains[end..])
        } else {
            Err(ParseError::ExpectedTokenType(TokenType::Arrow))?
        };

    let parse_parameter = |t: &Token| {
        if let TT::Identifier(id) = t.token_type() {
            Ok(Parameter {
                name: Identifier::new(id),
                type_annotation: None,
            })
        } else {
            Err(ParseError::UnexpectedToken(t.clone()))
        }
    };

    Ok((
        params
            .iter()
            .map(parse_parameter)
            .collect::<Result<_, _>>()?,
        remains,
    ))
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
        [T(TT::Keyword(If), position), remains @ ..] => {
            parse_if_expression(position.clone(), remains)
        }
        [T(TT::Literal(literal), ..), remains @ ..] => {
            Ok((Expression::Literal(literal.clone().into()), remains))
        }
        [T(TT::Identifier(id), ..), remains @ ..] => {
            Ok((Expression::Variable(Identifier::new(&id)), remains))
        }
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_if_expression<'a>(
    _position: SourcePosition,
    remains: &'a [Token],
) -> ParseResult<'a, Expression> {
    let (predicate, remains) = parse_expression(remains, 0)?;

    let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
    let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);

    if matches!(remains, [T(TT::Keyword(Then), ..), ..]) {
        let remains = &remains[1..];
        let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
        let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);

        let (consequent, remains) = parse_expression(remains, 0)?;

        let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
        let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);
        let remains = strip_if_starts_with(TT::Layout(Layout::Dedent), remains);
        if matches!(remains, [T(TT::Keyword(Else), ..), ..]) {
            let remains = &remains[1..];
            let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
            let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);

            let (alternate, remains) = parse_expression(&remains[0..], 0)?;

            Ok((
                Expression::ControlFlow(ControlFlow::If {
                    predicate: predicate.into(),
                    consequent: consequent.into(),
                    alternate: alternate.into(),
                }),
                remains,
            ))
        } else {
            Err(ParseError::ExpectedTokenType(TokenType::Keyword(Else)))
        }
    } else {
        Err(ParseError::ExpectedTokenType(TokenType::Keyword(Then)))
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
    position: SourcePosition,
    binder: &str,
    input: &'a [Token],
) -> ParseResult<'a, Expression> {
    let indented = starts_with(TokenType::Layout(Layout::Indent), input);
    let (bound, remains) = parse_expression(strip_first_if(indented, input), 0)?;

    let dedented = starts_with(TokenType::Layout(Layout::Dedent), remains);
    let newlined = starts_with(TokenType::Layout(Layout::Newline), remains);
    match strip_first_if(indented && dedented || newlined, remains) {
        [T(TT::Keyword(In), ..), remains @ ..] => {
            let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
            let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);

            let (body, remains) = parse_expression(remains, 0)?;

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
        //        println!("is_done: {t:?}");
        matches!(
            t.token_type(),
            TT::Keyword(..) | TT::End | TT::Layout(Layout::Dedent)
        )
    };

    // Operators can be prefigured by Layout::Newline
    // Juxtapositions though? I would have to be able to ask the Expression
    // about where it started
    match input {
        [t, ..] if is_done(t) => Ok((lhs, input)),

        [T(TT::Layout(..), ..), u, ..] if is_done(u) => Ok((lhs, input)),

        // <op> <expr>
        [T(TT::Operator(op), ..), remains @ ..] => {
            parse_operator(lhs, input, precedence, op, remains)
        }

        // ( <Newline> | <Indent> ) <op> <expr>
        // -- a continuation of the infix operator sequence on the next line (possibly indented.)
        [T(TT::Layout(Layout::Newline | Layout::Indent), ..), T(TT::Operator(op), ..), remains @ ..] => {
            parse_operator(lhs, input, precedence, op, remains)
        }

        // ( <Newline> | <;> ) <expr>
        // -- an expression sequence, e.g.: <statement>* <expr>
        [T(TT::Layout(Layout::Newline) | TT::Semicolon, ..), lookahead @ ..] if input.len() > 0 => {
            if !starts_with(TT::End, &input[1..]) && !is_toplevel(lookahead) {
                parse_sequence(lhs, &input[1..])
            } else {
                Ok((lhs, input))
            }
        }

        // <expr>
        //     <expr>
        // -- Function application, argument indented
        [T(TT::Layout(Layout::Indent), ..), remains @ ..] => {
            parse_juxtaposed(lhs, remains, precedence)
        }

        // <expr> <expr>
        // -- Function application
        [T(tt, ..), ..]
            if !matches!(tt, TT::Layout(Layout::Dedent) | TT::End | TT::Keyword(..)) =>
        {
            parse_juxtaposed(lhs, input, precedence)
        }

        _otherwise => Ok((lhs, input)),
    }
}

fn is_toplevel<'a>(prefix: &'a [Token]) -> bool {
    matches!(
        prefix,
        [
            T(TT::Identifier(..), ..),
            T(TT::Equals | TT::TypeAssign, ..),
            ..
        ]
    )
}

fn parse_operator<'a>(
    lhs: Expression,
    input: &'a [Token],
    context_precedence: usize,
    operator: &Operator,
    remains: &'a [Token],
) -> ParseResult<'a, Expression> {
    let operator_precedence = operator.precedence();
    if operator_precedence > context_precedence {
        let (rhs, remains) = parse_expression(remains, operator_precedence)?;

        parse_infix(
            apply_binary_operator(*operator, lhs, rhs),
            remains,
            context_precedence,
        )
    } else {
        Ok((lhs, input))
    }
}

fn parse_juxtaposed<'a>(
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

fn parse_sequence<'a>(lhs: Expression, tokens: &'a [Token]) -> ParseResult<'a, Expression> {
    let (rhs, remains) = parse_expression(tokens, 0)?;

    let sequence = Expression::Sequence {
        this: lhs.into(),
        and_then: rhs.into(),
    };

    if matches!(
        remains,
        [T(TT::Semicolon | TT::Layout(Layout::Newline), ..), ..]
    ) {
        parse_sequence(sequence, &remains[1..])
    } else {
        Ok((sequence, remains))
    }
}

fn apply_binary_operator(op: Operator, lhs: Expression, rhs: Expression) -> Expression {
    let apply_lhs = Expression::Apply {
        function: Expression::Variable(op.id()).into(),
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
                postition: SourcePosition::default(),
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
                postition: SourcePosition::default(),
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
                postition: SourcePosition::default(),
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
                postition: SourcePosition::default(),
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
                        function: E::Variable(Id::new("+")).into(),
                        argument: E::Apply {
                            function: E::Apply {
                                function: E::Variable(Id::new("*")).into(),
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
                postition: SourcePosition::default(),
                binder: Id::new("x"),
                bound: E::Literal(Constant::Int(10)).into(),
                body: E::Binding {
                    postition: SourcePosition::new(1, 15),
                    binder: Id::new("y"),
                    bound: E::Literal(Constant::Int(20)).into(),
                    body: E::Apply {
                        function: E::Apply {
                            function: E::Variable(Id::new("+")).into(),
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
        let (decl, _) = parse_declaration(lexer.tokenize(&into_input(
            r#"|create_window =
               |    fun x ->
               |        1 + x"#,
        )))
        .unwrap();

        assert_eq!(
            Declaration::Value {
                binder: Identifier::new("create_window"),
                declarator: ValueDeclarator::Function(FunctionDeclarator {
                    parameters: vec![Parameter {
                        name: Identifier::new("x"),
                        type_annotation: None
                    }],
                    return_type_annotation: None,
                    body: E::Apply {
                        function: E::Apply {
                            function: E::Variable(Id::new("+")).into(),
                            argument: E::Literal(Constant::Int(1)).into()
                        }
                        .into(),
                        argument: Expression::Variable(Identifier::new("x")).into()
                    },
                }),
                position: SourcePosition::default(),
            },
            decl
        );
    }

    //    #[test]
    //    fn unexpected_token_error() {
    //        let mut lexer = LexicalAnalyzer::default();
    //        let result = parse_expr_phrase(lexer.tokenize(&into_input("let x 10 in x + 20")));
    //        assert!(matches!(
    //            result,
    //            Err(ParseError::ExpectedTokenType(TT::Equals))
    //        ));
    //    }

    #[test]
    fn module_function_decls() {
        let mut lexer = LexicalAnalyzer::default();
        let (decls, _) = parse_declarations(lexer.tokenize(&into_input(
            r#"|create_window =
               |    fun x ->
               |        1 + x
               |
               |print_endline = fun s ->
               |    __print s
               |"#,
        )))
        .unwrap();

        assert_eq!(
            vec![
                Declaration::Value {
                    binder: Identifier::new("create_window"),
                    declarator: ValueDeclarator::Function(FunctionDeclarator {
                        parameters: vec![Parameter {
                            name: Identifier::new("x"),
                            type_annotation: None
                        }],
                        return_type_annotation: None,
                        body: E::Apply {
                            function: E::Apply {
                                function: E::Variable(Id::new("+")).into(),
                                argument: E::Literal(Constant::Int(1)).into()
                            }
                            .into(),
                            argument: E::Variable(Id::new("x")).into()
                        },
                    }),
                    position: SourcePosition::default(),
                },
                Declaration::Value {
                    position: SourcePosition::new(5, 1),
                    binder: Id::new("print_endline"),
                    declarator: ValueDeclarator::Function(FunctionDeclarator {
                        parameters: vec![Parameter {
                            name: Identifier::new("s"),
                            type_annotation: None
                        }],
                        return_type_annotation: None,
                        body: E::Apply {
                            function: E::Variable(Id::new("__print")).into(),
                            argument: E::Variable(Identifier::new("s")).into()
                        }
                    })
                }
            ],
            decls
        );
    }
}
