use core::panic;
use std::fmt;

use crate::{
    ast::{
        Apply, Binding, CompilationUnit, Constructor, ConstructorPattern, ControlFlow, Coproduct,
        Declaration, DeconstructInto, Expression, Identifier, Lambda, MatchClause,
        ModuleDeclarator, Parameter, Pattern, SelfReferential, Sequence, TuplePattern, TypeApply,
        TypeDeclaration, TypeDeclarator, TypeExpression, TypeName, TypeSignature,
        UniversalQuantifier, ValueDeclaration, ValueDeclarator,
    },
    lexer::{Keyword, Layout, Operator, SourceLocation, Token, TokenType},
    typer,
};

pub type ParseResult<'a, A> = Result<(A, &'a [Token]), ParseError>;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected token {0}")]
    UnexpectedToken(Token),

    #[error("Unexpected remains {0:?}")]
    UnexpectedRemains(Vec<Token>),

    #[error("Expected: {0}")]
    ExpectedTokenType(TokenType),

    #[error("Declaration stream {1:?} is offside of {0}")]
    DeclarationOffside(SourceLocation, Vec<Token>),
}

use thiserror::Error;
use Keyword::*;
use Token as T;
use TokenType as TT;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ParsingInfo {
    pub position: SourceLocation,
}

impl ParsingInfo {
    pub fn new(position: SourceLocation) -> Self {
        Self { position }
    }

    pub fn location(&self) -> &SourceLocation {
        &self.position
    }
}

impl fmt::Display for ParsingInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { position } = self;
        write!(f, "{position}")
    }
}

impl typer::Parsed for ParsingInfo {
    fn info(&self) -> &ParsingInfo {
        self
    }
}

static DEFAULT_PARSING_INFO: ParsingInfo = ParsingInfo {
    position: SourceLocation { row: 0, column: 0 },
};

impl typer::Parsed for () {
    fn info(&self) -> &ParsingInfo {
        &DEFAULT_PARSING_INFO
    }
}

pub fn parse_compilation_unit<'a>(
    input: &'a [Token],
) -> Result<CompilationUnit<ParsingInfo>, ParseError> {
    let (declarations, ..) = parse_declarations(input)?;

    Ok(CompilationUnit::Implicit(
        ParsingInfo::new(*input[0].location()),
        ModuleDeclarator {
            name: Identifier::new("main"),
            declarations,
        },
    ))
}

pub fn parse_declarations<'a>(
    input: &'a [Token],
) -> ParseResult<'a, Vec<Declaration<ParsingInfo>>> {
    let mut declarations = Vec::default();
    let mut input = input;

    loop {
        let (declaration, remains) = parse_declaration(input)?;
        let current_block = *declaration.position();
        declarations.push(declaration);

        // This logic is highly dubious
        input = find_next_in_block(current_block, remains, |token_type| {
            matches!(token_type, TT::Identifier(..) | TT::End)
        })?;

        if matches!(input, [T(TT::Layout(Layout::Dedent) | TT::End, ..), ..]) {
            break Ok((declarations, remains));
        }
    }
}

pub fn find_next_in_block<'a>(
    current_block: SourceLocation,
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

pub fn parse_declaration<'a>(input: &'a [Token]) -> ParseResult<'a, Declaration<ParsingInfo>> {
    match input {
        [T(TT::Identifier(id), pos), T(TT::Equals, ..), remains @ ..] => {
            parse_value_binding(id, None, pos, remains)
        }
        [T(TT::Identifier(id), pos), T(TT::TypeAscribe, ..), remains @ ..] => {
            parse_typed_value_binding(id, pos, remains)
        }
        [T(TT::Identifier(id), pos), T(TT::TypeAssign, ..), remains @ ..] => {
            parse_type_binding(id, pos, remains)
        }
        [t, ..] => Err(ParseError::UnexpectedToken(t.clone())),
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_typed_value_binding<'a>(
    id: &String,
    pos: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    let (quantifier, remains) = if starts_with(TT::Keyword(Forall), remains) {
        parse_universal_quantifier(&remains[1..])
            .map(|(quantifier, remains)| (Some(quantifier), remains))?
    } else {
        (None, remains)
    };

    let (body, remains) = parse_type_expression(remains)?;
    let remains = expect(TT::Equals, remains)?;

    parse_value_binding(id, Some(TypeSignature { quantifier, body }), pos, remains)
}

fn parse_type_binding<'a>(
    binder: &String,
    position: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    let (declarator, remains) =
        parse_type_declarator(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;

    Ok((
        Declaration::Type(
            ParsingInfo::new(*position),
            TypeDeclaration {
                binding: Identifier::new(&binder),
                declarator,
            },
        ),
        remains,
    ))
}

fn parse_type_declarator<'a>(remains: &'a [Token]) -> ParseResult<'a, TypeDeclarator<ParsingInfo>> {
    // forall a.
    let postition = remains[0].location();
    // Try different avenues down the line.
    let (coproduct, remains) = parse_coproduct(remains)?;

    Ok((
        TypeDeclarator::Coproduct(ParsingInfo::new(*postition), coproduct),
        remains,
    ))
}

fn parse_universal_quantifier<'a>(remains: &'a [Token]) -> ParseResult<'a, UniversalQuantifier> {
    let end = remains
        .iter()
        .position(|t| t.token_type() == &TT::Period)
        .ok_or_else(|| ParseError::ExpectedTokenType(TT::Period))?;
    let (params, remains) = remains.split_at(end);
    let remains = expect(TT::Period, remains)?;

    let parse_parameter = |t: &Token| {
        if let TT::Identifier(id) = t.token_type() {
            Ok(TypeName::new(id))
        } else {
            Err(ParseError::UnexpectedToken(t.clone()))
        }
    };

    Ok((
        UniversalQuantifier(
            params
                .iter()
                .map(parse_parameter)
                .collect::<Result<_, _>>()?,
        ),
        remains,
    ))
}

fn parse_coproduct<'a>(remains: &'a [Token]) -> ParseResult<'a, Coproduct<ParsingInfo>> {
    let (forall, mut remains) = if starts_with(TT::Keyword(Keyword::Forall), remains) {
        parse_universal_quantifier(&remains[1..])?
    } else {
        (UniversalQuantifier::default(), remains)
    };

    // parse the first constructor to see:
    //   if there are more constructors and
    //   how they are separated. (Newline or Pipe.)
    let mut boofer = vec![];
    let (constructor, remains1) =
        parse_constructor(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;

    remains = remains1;
    boofer.push(constructor);

    // Constructors are either inline, separated by |, or broken down, separated by Newline
    if let [T(separator @ (TT::Pipe | TT::Layout(Layout::Newline)), ..), ..] = remains {
        while matches!(remains, [t, ..] if t.token_type() == separator) {
            let (constructor, remains1) = parse_constructor(&remains[1..])?;
            boofer.push(constructor);

            remains = remains1;
        }
    }

    Ok((
        Coproduct {
            forall,
            constructors: boofer,
        },
        remains,
    ))
}

fn parse_constructor<'a>(remains: &'a [Token]) -> ParseResult<'a, Constructor<ParsingInfo>> {
    if let [T(TT::Identifier(name), _position), remains @ ..] = remains {
        let (signature, remains) = parse_constructor_signature(remains)?;
        Ok((
            Constructor {
                name: Identifier::new(name),
                signature,
            },
            remains,
        ))
    } else {
        Err(ParseError::ExpectedTokenType(TT::Identifier(
            "<constructor2>".to_owned(),
        )))
    }
}

fn parse_constructor_signature<'a>(
    mut remains: &'a [Token],
) -> ParseResult<'a, Vec<TypeExpression<ParsingInfo>>> {
    let mut boofer = vec![];

    while matches!(remains, [T(TT::LeftParen | TT::Identifier(..), ..), ..]) {
        let (term, rem) = parse_type_expression(remains)?;
        boofer.push(term);

        remains = if starts_with(TT::RightParen, rem) {
            &rem[1..]
        } else {
            rem
        };
    }

    Ok((boofer, remains))
}

// Type Expression ::= Type-name [ Type-arg* ]
//   Type-name ::= Identifier
//   Type-arg  ::= '(' Type-arg  ')' | Genuine-type | Type-parameter
//   Genuine-type ::= Uppercase?(Identifier)
//   Type-parameter ::= Lowercase?(Identifier)
fn parse_type_expression<'a>(remains: &'a [Token]) -> ParseResult<'a, TypeExpression<ParsingInfo>> {
    let (prefix, remains) = parse_type_expression_prefix(remains)?;
    parse_type_expression_infix(prefix, remains)
}

fn parse_type_expression_prefix<'a>(
    remains: &'a [Token],
) -> ParseResult<'a, TypeExpression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(id), pos), remains @ ..] => {
            Ok((simple_type_expr_term(ParsingInfo::new(*pos), id), remains))
        }
        [T(TT::LeftParen, ..), ..] => {
            let (term, remains) = parse_type_expression(&remains[1..])?;
            if starts_with(TT::RightParen, remains) {
                Ok((term, remains))
            } else {
                Err(ParseError::ExpectedTokenType(TT::RightParen))
            }
        }
        _otherwise => Err(ParseError::ExpectedTokenType(TT::Identifier(
            "<Constructor>".to_owned(),
        ))),
    }
}

fn simple_type_expr_term(pi: ParsingInfo, id: &str) -> TypeExpression<ParsingInfo> {
    if is_lowercasse(id) {
        TypeExpression::Parameter(pi, TypeName::new(id))
    } else {
        TypeExpression::Constant(pi, TypeName::new(id))
    }
}

fn is_lowercasse(id: &str) -> bool {
    id.chars().all(char::is_lowercase)
}

fn parse_type_expression_infix<'a>(
    lhs: TypeExpression<ParsingInfo>,
    remains: &'a [Token],
) -> ParseResult<'a, TypeExpression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(rhs), pos), remains @ ..] => {
            let pi = ParsingInfo::new(*pos);
            let (rhs, remains) =
                parse_type_expression_infix(simple_type_expr_term(pi, rhs), remains)?;

            Ok((
                TypeExpression::Apply(
                    pi,
                    TypeApply {
                        constructor: lhs.into(),
                        argument: rhs.into(),
                    },
                ),
                remains,
            ))
        }
        _otherwise => Ok((lhs, remains)),
    }
}

fn parse_value_binding<'a>(
    binder: &str,
    type_signature: Option<TypeSignature<ParsingInfo>>,
    position: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    let (declarator, remains) = parse_value_declarator(
        binder,
        strip_if_starts_with(TT::Layout(Layout::Indent), remains),
    )?;

    Ok((
        Declaration::Value(
            ParsingInfo::new(*position),
            ValueDeclaration {
                binder: Identifier::new(&binder),
                type_signature,
                declarator,
            },
        ),
        remains,
    ))
}

fn parse_value_declarator<'a>(
    binder: &str,
    input: &'a [Token],
) -> ParseResult<'a, ValueDeclarator<ParsingInfo>> {
    let (expression, remains) = parse_expression(input, 0)?;

    let expression = if let Expression::Lambda(annotation, Lambda { parameter, body }) = expression
    {
        Expression::SelfReferential(
            annotation,
            SelfReferential {
                name: Identifier::new(binder),
                parameter,
                body,
            },
        )
    } else {
        expression
    };

    Ok((ValueDeclarator { expression }, remains))
}

// Should this function eat the -> ?
// a | pattern
// This loses the annotation. Put it back. Later :)
fn parse_parameter_list<'a>(remains: &'a [Token]) -> ParseResult<'a, Vec<Parameter>> {
    let end = remains
        .iter()
        .position(|t| t.token_type() == &TT::Period)
        .ok_or_else(|| ParseError::ExpectedTokenType(TT::Period))?;
    let (params, remains) = remains.split_at(end);

    fn parse_parameter(t: &Token) -> Result<Parameter, ParseError> {
        if let TT::Identifier(id) = t.token_type() {
            Ok(Parameter {
                name: Identifier::new(id),
                type_annotation: None,
            })
        } else {
            Err(ParseError::UnexpectedToken(t.clone()))
        }
    }

    Ok((
        params
            .iter()
            .map(parse_parameter)
            .collect::<Result<_, _>>()?,
        remains,
    ))
}

pub fn parse_expression_phrase<'a>(
    tokens: &'a [Token],
) -> Result<Expression<ParsingInfo>, ParseError> {
    let (expression, remains) = parse_expression(tokens, 0)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

pub fn parse_declaration_phrase<'a>(
    tokens: &'a [Token],
) -> Result<Declaration<ParsingInfo>, ParseError> {
    let (expression, remains) = parse_declaration(tokens)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

fn parse_prefix<'a>(tokens: &'a [Token]) -> ParseResult<'a, Expression<ParsingInfo>> {
    match tokens {
        [T(TT::Keyword(Let), position), T(TT::Identifier(binder), ..), T(TT::Equals, ..), remains @ ..] => {
            parse_binding(*position, binder, remains)
        }
        [T(TT::Keyword(If), position), remains @ ..] => parse_if_expression(*position, remains),
        [T(TT::Keyword(Deconstruct), position), remains @ ..] => {
            parse_deconstruct_into(*position, remains)
        }
        [T(TT::Keyword(Keyword::Lambda), position), remains @ ..] => {
            parse_lambda(*position, remains)
        }
        [T(TT::Literal(literal), position), remains @ ..] => Ok((
            Expression::Literal(ParsingInfo::new(*position), literal.clone().into()),
            remains,
        )),
        [T(TT::Identifier(id), position), remains @ ..] => Ok((
            Expression::Variable(ParsingInfo::new(*position), Identifier::new(&id)),
            remains,
        )),
        [T(TT::LeftParen, ..), remains @ ..] => {
            let (expr, remains) = parse_expression(remains, 0)?;
            Ok((expr, expect(TT::RightParen, remains)?))
        }
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_lambda<'a>(
    position: SourceLocation,
    tokens: &'a [Token],
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let (parameters, remains) = parse_parameter_list(tokens)?;
    let remains = expect(TT::Period, remains)?;
    let (body, remains) =
        parse_expression(strip_if_starts_with(TT::Layout(Layout::Indent), remains), 0)?;

    Ok((
        parameters.into_iter().rfold(body, |body, parameter| {
            Expression::Lambda(
                ParsingInfo::new(position),
                Lambda {
                    parameter,
                    body: body.into(),
                },
            )
        }),
        remains,
    ))
}

// This function is __TERRIBLE__.
fn parse_if_expression<'a>(
    position: SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Expression<ParsingInfo>> {
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
                Expression::ControlFlow(
                    ParsingInfo::new(position),
                    ControlFlow::If {
                        predicate: predicate.into(),
                        consequent: consequent.into(),
                        alternate: alternate.into(),
                    },
                ),
                remains,
            ))
        } else {
            Err(ParseError::ExpectedTokenType(TokenType::Keyword(Else)))
        }
    } else {
        Err(ParseError::ExpectedTokenType(TokenType::Keyword(Then)))
    }
}

fn parse_deconstruct_into<'a>(
    position: SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let (scrutinee, remains) = parse_expression(remains, 0)?;

    let mut remains = expect(
        TT::Keyword(Into),
        strip_if_starts_with(TT::Layout(Layout::Indent), remains),
    )?;

    let mut boofer = vec![];

    let (match_clause, remains1) =
        parse_match_clause(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;
    remains = strip_if_starts_with(TT::Layout(Layout::Dedent), remains1);
    remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
    boofer.push(match_clause);

    // This syntax won't work because Newline triggers the
    // sequence parse rule so the expression in the consequent
    // continues
    while matches!(remains, [T(TT::Pipe, ..), ..]) {
        let (match_clause, remains1) = parse_match_clause(&remains[1..])?;
        boofer.push(match_clause);
        remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains1);
    }

    Ok((
        Expression::DeconstructInto(
            ParsingInfo::new(position),
            DeconstructInto {
                scrutinee: scrutinee.into(),
                match_clauses: boofer,
            },
        ),
        remains,
    ))
}

fn parse_match_clause<'a>(remains: &'a [Token]) -> ParseResult<'a, MatchClause<ParsingInfo>> {
    let (pattern, remains) = parse_pattern(remains)?;
    let remains = expect(TT::Arrow, remains)?;
    let (consequent, remains) = parse_expression(remains, 0)?;

    Ok((
        MatchClause {
            pattern,
            consequent: consequent.into(),
        },
        remains,
    ))
}

fn parse_pattern<'a>(remains: &'a [Token]) -> ParseResult<'a, Pattern<ParsingInfo>> {
    // (1, 2, 3) -> print_endline "1, 2, 3"` XX tuple (flattened?)
    // This x -> print_endline (show x)      XX Coproduct constructor
    // others -> print_endline (show others) XX catch all
    match remains {
        [T(TT::Identifier(id), position), ..] if is_capital_case(id) => {
            parse_constructor_pattern(remains, position)
        }

        [T(TT::Identifier(id), _position), remains @ ..] => {
            Ok((Pattern::Otherwise(Identifier::new(id)), remains))
        }

        [T(TT::Literal(lit), _position), remains @ ..] => {
            Ok((Pattern::Literally(lit.clone().into()), remains))
        }

        [T(TT::LeftParen, position), remains @ ..] => parse_tuple_pattern(remains, position)
            .map(|(tuple, remains)| (Pattern::Tuple(ParsingInfo::new(*position), tuple), remains)),

        otherwise => panic!("{otherwise:?}"),
    }
}

fn is_capital_case(id: &str) -> bool {
    id.starts_with(char::is_uppercase)
}

fn parse_constructor_pattern<'a>(
    remains: &'a [Token],
    _position: &SourceLocation,
) -> ParseResult<'a, Pattern<ParsingInfo>> {
    if let [T(TT::Identifier(constructor), position), remains @ ..] = remains {
        let (patterns, remains) = parse_pattern_list(remains)?;
        Ok((
            Pattern::Coproduct(
                ParsingInfo::new(*position),
                ConstructorPattern {
                    constructor: Identifier::new(&constructor),
                    argument: TuplePattern { elements: patterns },
                },
            ),
            remains,
        ))
    } else {
        panic!("{remains:?}")
    }
}

fn parse_pattern_list<'a>(mut remains: &'a [Token]) -> ParseResult<'a, Vec<Pattern<ParsingInfo>>> {
    let mut boofer = vec![];

    while !matches!(remains, [T(TT::Arrow, ..), ..]) {
        let (pattern, remains1) = parse_pattern(remains)?;
        boofer.push(pattern);
        remains = remains1;
    }

    Ok((boofer, remains))
}

fn parse_tuple_pattern<'a>(
    mut remains: &'a [Token],
    _position: &SourceLocation,
) -> ParseResult<'a, TuplePattern<ParsingInfo>> {
    let mut boofer = vec![];

    let (pattern, remains1) = parse_pattern(remains)?;
    remains = remains1;
    boofer.push(pattern);

    while matches!(remains, [T(TT::Comma, ..), ..]) {
        let (pattern, remains1) = parse_pattern(&remains[1..])?;
        boofer.push(pattern);
        remains = remains1;
    }

    remains = expect(TT::RightParen, remains)?;

    Ok((TuplePattern { elements: boofer }, remains))
}

fn expect<'a>(token_type: TokenType, remains: &'a [Token]) -> Result<&'a [Token], ParseError> {
    if starts_with(token_type.clone(), remains) {
        Ok(&remains[1..])
    } else {
        Err(ParseError::ExpectedTokenType(token_type))
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
    position: SourceLocation,
    binder: &str,
    input: &'a [Token],
) -> ParseResult<'a, Expression<ParsingInfo>> {
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
                Expression::Binding(
                    ParsingInfo::new(position),
                    Binding {
                        binder: Identifier::new(binder),
                        bound: bound.into(),
                        body: body.into(),
                    },
                ),
                remains,
            ))
        }
        // In could be offside here, then what? What can I return or do?
        // Well, that is an error then I guess.
        otherwise => panic!("{otherwise:?}"),
    }
}

pub fn parse_expression<'a>(
    input: &'a [Token],
    precedence: usize,
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let (prefix, remains) = parse_prefix(input)?;

    parse_infix(prefix, remains, precedence)
}

fn is_expression_terminator(t: &Token) -> bool {
    matches!(
        t.token_type(),
        TT::Keyword(Keyword::In | Keyword::Else | Keyword::Then | Keyword::Into)
            | TT::End
            | TT::Layout(Layout::Dedent)
            | TT::RightParen
            | TT::Pipe,
    )
}

// Infixes end with End and In
fn parse_infix<'a>(
    lhs: Expression<ParsingInfo>,
    input: &'a [Token],
    precedence: usize,
) -> ParseResult<'a, Expression<ParsingInfo>> {
    // Operators can be prefigured by Layout::Newline
    // Juxtapositions though? I would have to be able to ask the Expression
    // about where it started
    match input {
        [t, ..] if is_expression_terminator(t) => Ok((lhs, input)),

        [T(TT::Layout(..), ..), u, ..] if is_expression_terminator(u) => Ok((lhs, input)),

        // ( <Newline> | <;> ) <expr>
        // -- an expression sequence, e.g.: <statement>* <expr>
        [T(TT::Layout(Layout::Newline) | TT::Semicolon, ..), lookahead @ ..] if input.len() > 0 => {
            if !starts_with(TT::End, &input[1..]) && !is_toplevel(lookahead) {
                parse_sequence(lhs, &input[1..])
            } else {
                Ok((lhs, input))
            }
        }

        // <op> <expr>
        [T(op, pos), remains @ ..] if Operator::is_defined(op) => {
            let op = Operator::try_from(op).expect("Failed to decode operator");
            parse_operator(lhs, input, precedence, &op, remains, *pos)
        }

        // ( <Newline> | <Indent> ) <op> <expr>
        // -- a continuation of the infix operator sequence on the next line (possibly indented.)
        [T(TT::Layout(Layout::Newline | Layout::Indent), ..), T(op, pos), remains @ ..]
            if Operator::is_defined(op) =>
        {
            let op = Operator::try_from(op).expect("Failed to decode operator");
            parse_operator(lhs, input, precedence, &op, remains, *pos)
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
            if !matches!(
                tt,
                TT::Layout(Layout::Dedent)
                    | TT::End
                    | TT::Keyword(And | Or | Xor | Else | Into | In)
            ) =>
        {
            parse_juxtaposed(lhs, input, precedence)
        }

        _otherwise => Ok((lhs, input)),
    }
}

// This is an annoying function
fn is_toplevel<'a>(prefix: &'a [Token]) -> bool {
    matches!(
        prefix,
        [
            T(TT::Identifier(..), ..),
            T(TT::Equals | TT::TypeAssign | TT::TypeAscribe, ..),
            ..
        ]
    )
}

fn parse_operator<'a>(
    lhs: Expression<ParsingInfo>,
    input: &'a [Token],
    context_precedence: usize,
    operator: &Operator,
    remains: &'a [Token],
    position: SourceLocation,
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let operator_precedence = operator.precedence();
    if operator_precedence > context_precedence {
        let (rhs, remains) = parse_expression(
            remains,
            if operator.is_right_associative() {
                operator_precedence - 1
            } else {
                operator_precedence
            },
        )?;

        parse_infix(
            apply_binary_operator(*operator, lhs, rhs, position),
            remains,
            context_precedence,
        )
    } else {
        Ok((lhs, input))
    }
}

fn parse_juxtaposed<'a>(
    lhs: Expression<ParsingInfo>,
    tokens: &'a [Token],
    precedence: usize,
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let (rhs, remains) = parse_prefix(tokens)?;

    // parse continuing arguments
    parse_infix(
        Expression::Apply(
            ParsingInfo::new(*lhs.position()),
            Apply {
                function: lhs.into(),
                argument: rhs.into(),
            },
        ),
        remains,
        precedence,
    )
}

fn parse_sequence<'a>(
    lhs: Expression<ParsingInfo>,
    tokens: &'a [Token],
) -> ParseResult<'a, Expression<ParsingInfo>> {
    let (rhs, remains) = parse_expression(tokens, 0)?;

    let sequence = Expression::Sequence(
        ParsingInfo::new(*rhs.position()),
        Sequence {
            this: lhs.into(),
            and_then: rhs.into(),
        },
    );

    if matches!(
        remains,
        [T(TT::Semicolon | TT::Layout(Layout::Newline), ..), u, ..] if !is_expression_terminator(u)
    ) {
        parse_sequence(sequence, &remains[1..])
    } else {
        Ok((sequence, remains))
    }
}

fn apply_binary_operator(
    op: Operator,
    lhs: Expression<ParsingInfo>,
    rhs: Expression<ParsingInfo>,
    position: SourceLocation,
) -> Expression<ParsingInfo> {
    let apply_lhs = Expression::Apply(
        ParsingInfo::new(position),
        Apply {
            function: Expression::Variable(ParsingInfo::new(position), op.id()).into(),
            argument: lhs.into(),
        },
    );
    Expression::Apply(
        ParsingInfo::new(position),
        Apply {
            function: apply_lhs.into(),
            argument: rhs.into(),
        },
    )
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

        let expr =
            parse_expression_phrase(lexer.tokenize(&into_input("let x = 10 in x + 20"))).unwrap();
        assert_eq!(
            E::Binding(
                (),
                Binding {
                    binder: Id::new("x"),
                    bound: E::Literal((), Constant::Int(10)).into(),
                    body: E::Apply(
                        (),
                        Apply {
                            function: E::Apply(
                                (),
                                Apply {
                                    function: E::Variable(
                                        (),
                                        Id::new(&Operator::Plus.function_identifier())
                                    )
                                    .into(),
                                    argument: E::Variable((), Id::new("x")).into(),
                                }
                            )
                            .into(),
                            argument: E::Literal((), Constant::Int(20)).into(),
                        }
                    )
                    .into(),
                }
            ),
            expr.map(|_| ())
        );
    }

    #[test]
    fn let_in_juxtaposed() {
        let mut lexer = LexicalAnalyzer::default();
        let expr =
            parse_expression_phrase(lexer.tokenize(&into_input("let x = 10 in f x 20"))).unwrap();
        assert_eq!(
            E::Binding(
                (),
                Binding {
                    binder: Id::new("x"),
                    bound: E::Literal((), Constant::Int(10)).into(),
                    body: E::Apply(
                        (),
                        Apply {
                            function: E::Apply(
                                (),
                                Apply {
                                    function: E::Variable((), Id::new("f")).into(),
                                    argument: E::Variable((), Id::new("x")).into(),
                                }
                            )
                            .into(),
                            argument: E::Literal((), Constant::Int(20)).into(),
                        }
                    )
                    .into(),
                }
            ),
            expr.map(|_| ())
        );
    }

    #[test]
    fn let_in_juxtaposed3() {
        let mut lexer = LexicalAnalyzer::default();
        let expr =
            parse_expression_phrase(lexer.tokenize(&into_input("let x = 10 in f x 1 2"))).unwrap();
        assert_eq!(
            E::Binding(
                (),
                Binding {
                    binder: Id::new("x"),
                    bound: E::Literal((), Constant::Int(10)).into(),
                    body: E::Apply(
                        (),
                        Apply {
                            function: E::Apply(
                                (),
                                Apply {
                                    function: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Variable((), Id::new("f")).into(),
                                            argument: E::Variable((), Id::new("x")).into(),
                                        }
                                    )
                                    .into(),
                                    argument: E::Literal((), Constant::Int(1)).into(),
                                }
                            )
                            .into(),
                            argument: E::Literal((), Constant::Int(2)).into()
                        }
                    )
                    .into(),
                }
            ),
            expr.map(|_| ())
        );
    }

    #[test]
    fn let_in_with_mixed_artithmetics() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse_expression_phrase(lexer.tokenize(&into_input(
            r#"|let x =
               |    print_endline "Hello, world"; f 1 2
               |in
               |    3 * f
               |           4 + 5"#,
        )))
        .unwrap();
        assert_eq!(
            E::Binding(
                (),
                Binding {
                    binder: Id::new("x"),
                    bound: E::Sequence(
                        (),
                        Sequence {
                            this: E::Apply(
                                (),
                                Apply {
                                    function: E::Variable((), Id::new("print_endline")).into(),
                                    argument: E::Literal(
                                        (),
                                        Constant::Text("Hello, world".to_owned())
                                    )
                                    .into(),
                                }
                            )
                            .into(),
                            and_then: E::Apply(
                                (),
                                Apply {
                                    function: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Variable((), Id::new("f")).into(),
                                            argument: E::Literal((), Constant::Int(1)).into()
                                        }
                                    )
                                    .into(),
                                    argument: E::Literal((), Constant::Int(2)).into()
                                }
                            )
                            .into()
                        }
                    )
                    .into(),
                    body: E::Apply(
                        (),
                        Apply {
                            function: E::Apply(
                                (),
                                Apply {
                                    function: E::Variable((), Id::new("+")).into(),
                                    argument: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Apply(
                                                (),
                                                Apply {
                                                    function: E::Variable((), Id::new("*")).into(),
                                                    argument: E::Literal((), Constant::Int(3))
                                                        .into()
                                                }
                                            )
                                            .into(),
                                            argument: E::Apply(
                                                (),
                                                Apply {
                                                    function: E::Variable((), Id::new("f")).into(),
                                                    argument: E::Literal((), Constant::Int(4))
                                                        .into()
                                                }
                                            )
                                            .into()
                                        }
                                    )
                                    .into()
                                }
                            )
                            .into(),
                            argument: E::Literal((), Constant::Int(5)).into()
                        }
                    )
                    .into()
                }
            ),
            expr.map(|_| ())
        );
    }

    #[test]
    fn nested_let_in() {
        let mut lexer = LexicalAnalyzer::default();
        let expr = parse_expression_phrase(
            lexer.tokenize(&into_input("let x = 10 in let y = 20 in x + y")),
        )
        .unwrap();

        assert_eq!(
            E::Binding(
                (),
                Binding {
                    binder: Id::new("x"),
                    bound: E::Literal((), Constant::Int(10)).into(),
                    body: E::Binding(
                        (),
                        Binding {
                            binder: Id::new("y"),
                            bound: E::Literal((), Constant::Int(20)).into(),
                            body: E::Apply(
                                (),
                                Apply {
                                    function: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Variable((), Id::new("+")).into(),
                                            argument: E::Variable((), Id::new("x")).into()
                                        }
                                    )
                                    .into(),
                                    argument: E::Variable((), Id::new("y")).into()
                                }
                            )
                            .into()
                        }
                    )
                    .into()
                }
            ),
            expr.map(|_| ())
        );
    }

    #[test]
    fn function_binding() {
        let mut lexer = LexicalAnalyzer::default();
        let (decl, _) = parse_declaration(lexer.tokenize(&into_input(
            r#"|create_window =
               |    lambda x.
               |        1 + x"#,
        )))
        .unwrap();

        assert_eq!(
            Declaration::Value(
                (),
                ValueDeclaration {
                    binder: Identifier::new("create_window"),
                    type_signature: None,
                    declarator: ValueDeclarator {
                        expression: E::SelfReferential(
                            (),
                            SelfReferential {
                                name: Identifier::new("create_window"),
                                parameter: Parameter::new(Identifier::new("x")),
                                body: E::Apply(
                                    (),
                                    Apply {
                                        function: E::Apply(
                                            (),
                                            Apply {
                                                function: E::Variable((), Id::new("+")).into(),
                                                argument: E::Literal((), Constant::Int(1)).into()
                                            }
                                        )
                                        .into(),
                                        argument: Expression::Variable((), Identifier::new("x"))
                                            .into()
                                    }
                                )
                                .into()
                            }
                        ),
                    },
                }
            ),
            decl.map(|_| ())
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
    fn if_after_juxtaposed() {
        let mut lexer = LexicalAnalyzer::default();
        let unit = parse_compilation_unit(lexer.tokenize(&into_input(
            r#"|create_window =
               |  lambda x.
               |    print_endline "Hi, mom"
               |    if x = 0 then
               |      print "hi"
               |    else
               |      print "hi"
               |
               |"#,
        )))
        .unwrap();

        println!("{}", unit);
    }

    #[test]
    fn module_function_decls() {
        let mut lexer = LexicalAnalyzer::default();
        let (mut decls, _) = parse_declarations(lexer.tokenize(&into_input(
            r#"|create_window =
               |    lambda x.
               |        1 + x
               |
               |print_endline = lambda s.
               |    __print s
               |"#,
        )))
        .unwrap();

        let decls = decls
            .drain(..)
            .map(|decl| decl.map(|_| ()))
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("create_window"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: E::SelfReferential(
                                (),
                                SelfReferential {
                                    name: Identifier::new("create_window"),
                                    parameter: Parameter::new(Identifier::new("x")),
                                    body: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Apply(
                                                (),
                                                Apply {
                                                    function: E::Variable((), Id::new("+")).into(),
                                                    argument: E::Literal((), Constant::Int(1))
                                                        .into()
                                                }
                                            )
                                            .into(),
                                            argument: E::Variable((), Id::new("x")).into()
                                        }
                                    )
                                    .into()
                                }
                            ),
                        },
                    }
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Id::new("print_endline"),
                        declarator: ValueDeclarator {
                            expression: E::SelfReferential(
                                (),
                                SelfReferential {
                                    name: Identifier::new("print_endline"),
                                    parameter: Parameter::new(Identifier::new("s")),
                                    body: E::Apply(
                                        (),
                                        Apply {
                                            function: E::Variable((), Id::new("__print")).into(),
                                            argument: E::Variable((), Identifier::new("s")).into()
                                        }
                                    )
                                    .into()
                                }
                            )
                        },
                        type_signature: None,
                    }
                )
            ],
            decls
        );
    }
}
