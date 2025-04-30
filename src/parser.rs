use core::panic;
use std::fmt;

use crate::{
    ast::{
        Apply, Arrow, Binding, CompilationUnit, Constant, Constructor, ConstructorPattern,
        ControlFlow, Coproduct, Declaration, DeconstructInto, Expression, Identifier, Lambda,
        MatchClause, ModuleDeclarator, Parameter, Pattern, Product, ProductIndex, Project,
        SelfReferential, Sequence, Struct, StructField, StructPattern, TuplePattern, TypeApply,
        TypeDeclaration, TypeDeclarator, TypeExpression, TypeName, TypeSignature,
        UniversalQuantifiers, ValueDeclaration, ValueDeclarator,
    },
    lexer::{Keyword, Layout, Literal, Operator, SourceLocation, Token, TokenType},
    typer,
};

pub type ParseResult<'a, A> = Result<(A, &'a [Token]), ParseError>;

pub trait ParseResultOps<'a, A> {
    fn map_value<B, F>(self, f: F) -> ParseResult<'a, B>
    where
        Self: Sized,
        F: FnOnce(A) -> B;
}

impl<'a, A> ParseResultOps<'a, A> for ParseResult<'a, A> {
    fn map_value<B, F>(self, f: F) -> ParseResult<'a, B>
    where
        Self: Sized,
        F: FnOnce(A) -> B,
    {
        self.map(|(a, remains)| (f(a), remains))
    }
}

#[derive(Debug)]
pub struct DisplayList<A>(Vec<A>);

impl<A> fmt::Display for DisplayList<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut i = self.0.iter();
        if let Some(element) = i.next() {
            write!(f, "{element}")?;
        }

        for element in i {
            write!(f, ", {element}")?;
        }

        Ok(())
    }
}

impl<A> From<Vec<A>> for DisplayList<A>
where
    A: fmt::Display,
{
    fn from(value: Vec<A>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected token {0}")]
    UnexpectedToken(Token),

    #[error("Unexpected remains {0:?}")]
    UnexpectedRemains(Vec<Token>),

    #[error("Expected: {0}")]
    ExpectedTokenType(TokenType),

    #[error("Expected one of {one_of}; got {received}")]
    Expected {
        one_of: DisplayList<TokenType>,
        received: TokenType,
    },

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

pub fn parse_compilation_unit(input: &[Token]) -> Result<CompilationUnit<ParsingInfo>, ParseError> {
    parse_declarations(input)
        .map_value(|declarations| {
            CompilationUnit::Implicit(
                ParsingInfo::new(*input[0].location()),
                ModuleDeclarator {
                    name: Identifier::new("main"),
                    declarations,
                },
            )
        })
        .map(|(fst, _)| fst)
}

pub fn parse_declarations(input: &[Token]) -> ParseResult<Vec<Declaration<ParsingInfo>>> {
    let mut declarations = Vec::default();

    let (decl, mut remains) = parse_declaration(input)?;
    let current_block = *decl.position();
    declarations.push(decl);

    //    println!("parse_decls(1): {:?}", remains);
    //    println!();

    loop {
        // Re-write as a match
        // remains starts with End because
        if let [T(TT::Layout(Layout::Dedent | Layout::Newline), block), remains1 @ ..] = remains {
            if starts_with(&TT::End, remains1) {
                break Ok((declarations, remains));
            }

            if block.is_same_block(&current_block) {
                let (decl, remains1) = parse_declaration(remains1)?;
                //                println!("parse_decls(2): {:?}", remains1);
                //                println!();

                declarations.push(decl);
                remains = remains1;
            } else if block.is_left_of(current_block.column) {
                break Ok((declarations, remains));
            } else {
                Err(ParseError::DeclarationOffside(
                    current_block,
                    remains1.to_vec(),
                ))?
            }
        } else if let [T(TT::End, ..), remains @ ..] = remains {
            break Ok((declarations, remains));
        } else {
            // parse_expression eats my Dedent somewhere.
            let (decl, remains1) = parse_declaration(remains)?;
            //            println!("parse_decls: {:?}", &remains1[..6]);

            declarations.push(decl);
            remains = remains1;
        }
    }
}

pub fn parse_declaration(input: &[Token]) -> ParseResult<Declaration<ParsingInfo>> {
    match input {
        [T(TT::Identifier(id), pos), T(TT::Equals, ..), remains @ ..] => {
            parse_value_binding(id, None, pos, remains)
        }
        [T(TT::Identifier(id), pos), T(TT::TypeAscribe, ..), remains @ ..] => {
            parse_type_annotated_value_binding(id, pos, remains)
        }
        [T(TT::Identifier(id), pos), T(TT::TypeAssign, ..), remains @ ..] => {
            parse_type_binding(id, pos, remains)
        }
        [t, ..] => Err(ParseError::UnexpectedToken(t.clone())),
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_type_annotated_value_binding<'a>(
    id: &str,
    pos: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    let (quantifier, remains) = if starts_with(&TT::Keyword(Forall), remains) {
        parse_universal_quantifier(&remains[1..])
            .map(|(quantifier, remains)| (Some(quantifier), remains))?
    } else {
        (None, remains)
    };

    let (body, remains) = parse_type_expression(remains)?;
    let remains = expect(&TT::Equals, remains)?;

    parse_value_binding(
        id,
        Some(TypeSignature {
            quantifiers: quantifier,
            body,
        }),
        pos,
        remains,
    )
}

fn parse_type_binding<'a>(
    binder: &str,
    position: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    parse_type_declarator(
        Identifier::new(binder),
        strip_if_starts_with(TT::Layout(Layout::Indent), remains),
    )
    .map_value(|declarator| {
        Declaration::Type(
            ParsingInfo::new(*position),
            TypeDeclaration {
                binding: Identifier::new(binder),
                declarator,
            },
        )
    })
}

fn parse_type_declarator(
    binder: Identifier,
    remains: &[Token],
) -> ParseResult<TypeDeclarator<ParsingInfo>> {
    match remains {
        [T(TT::LeftBrace, position), remains @ ..] => {
            let (struct_declarator, remains) = parse_struct_declarator(remains)?;

            let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);
            if starts_with(&TT::Keyword(Keyword::Where), remains) {
                let remains = expect(&TT::Layout(Layout::Indent), &remains[1..])?;
                parse_associated_module(binder, *position, struct_declarator, remains)
            } else {
                Ok((
                    TypeDeclarator::Struct(ParsingInfo::new(*position), struct_declarator),
                    remains,
                ))
            }
        }

        [T(.., pos), ..] => parse_coproduct(remains)
            .map_value(|coproduct| TypeDeclarator::Coproduct(ParsingInfo::new(*pos), coproduct)),

        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_associated_module<'a>(
    binder: Identifier,
    position: SourceLocation,
    struct_declarator: Struct<ParsingInfo>,
    remains: &'a [Token],
) -> ParseResult<'a, TypeDeclarator<ParsingInfo>> {
    println!("parse_associated_module: {:?}", &remains[..6]);
    parse_declarations(remains).map_value(|declarations| {
        println!("parse_associated_module: {declarations:?}");
        let module = ModuleDeclarator {
            name: binder,
            declarations,
        };
        TypeDeclarator::Struct(
            ParsingInfo::new(position),
            Struct {
                associated_module: Some(module),
                ..struct_declarator
            },
        )
    })
}

fn parse_universal_quantifier(remains: &[Token]) -> ParseResult<UniversalQuantifiers> {
    let end = remains
        .iter()
        .position(|t| t.token_type() == &TT::Period)
        .ok_or(ParseError::ExpectedTokenType(TT::Period))?;
    let (params, remains) = remains.split_at(end);
    let remains = expect(&TT::Period, remains)?;

    let parse_parameter = |t: &Token| {
        if let TT::Identifier(id) = t.token_type() {
            Ok(TypeName::new(id))
        } else {
            Err(ParseError::UnexpectedToken(t.clone()))
        }
    };

    Ok((
        UniversalQuantifiers(
            params
                .iter()
                .map(parse_parameter)
                .collect::<Result<_, _>>()?,
        ),
        remains,
    ))
}

fn parse_struct_declarator(remains: &[Token]) -> ParseResult<Struct<ParsingInfo>> {
    let (forall, remains) = if starts_with(&TT::Keyword(Keyword::Forall), remains) {
        // Move this to parse_type_declarator
        parse_universal_quantifier(&remains[1..])?
    } else {
        (UniversalQuantifiers::default(), remains)
    };

    let mut fields = vec![];

    // Strips a potential Indent first here. This is the C++ brace case
    let (field, mut remains) =
        parse_struct_field_declaration(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;
    fields.push(field);

    // Strips another potential Indent here. This is the F# brace case, e.g.:
    // { Foo : X
    //   Bar : Y
    // }
    let (field, remains1) =
        parse_struct_field_declaration(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;
    fields.push(field);
    remains = remains1;

    // field : type [ (; field : type)* ]
    //   or
    // field : type ([
    // field : type
    // ])*
    if let [T(separator @ (TT::Semicolon | TT::Layout(Layout::Newline)), ..), ..] = remains {
        while matches!(remains, [T(t, ..), T(lookahead, ..), ..] if t == separator && lookahead != &TT::RightBrace)
        {
            let (field, remains1) = parse_struct_field_declaration(&remains[1..])?;
            fields.push(field);
            remains = remains1;
        }
    }

    Ok((
        Struct {
            forall,
            fields,
            associated_module: None,
        },
        expect(
            &TT::RightBrace,
            strip_if_starts_with(TT::Layout(Layout::Dedent), remains),
        )?,
    ))
}

fn parse_struct_field_declaration(remains: &[Token]) -> ParseResult<StructField<ParsingInfo>> {
    if let [T(TT::Identifier(field_name), _field_name_pos), T(TT::Colon, ..), remains @ ..] =
        remains
    {
        parse_type_expression(remains).map_value(|type_annotation| StructField {
            name: Identifier::new(field_name),
            type_annotation,
        })
    } else {
        Err(ParseError::ExpectedTokenType(TT::Identifier(
            "field name".to_owned(),
        )))
    }
}

fn parse_coproduct(remains: &[Token]) -> ParseResult<Coproduct<ParsingInfo>> {
    let (forall, mut remains) = if starts_with(&TT::Keyword(Keyword::Forall), remains) {
        // Move this to parse_type_declarator
        parse_universal_quantifier(&remains[1..])?
    } else {
        (UniversalQuantifiers::default(), remains)
    };

    // parse the first constructor to see:
    //   if there are more constructors and
    //   how they are separated. (Newline or Pipe.)
    let mut constructors = vec![];
    let (constructor, remains1) =
        parse_constructor(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;

    remains = remains1;
    constructors.push(constructor);

    // Constructors are either inline, separated by |, or broken down, separated by Newline
    if let [T(separator @ (TT::Pipe | TT::Layout(Layout::Newline)), ..), ..] = remains {
        while matches!(remains, [t, ..] if t.token_type() == separator) {
            let (constructor, remains1) = parse_constructor(&remains[1..])?;
            constructors.push(constructor);

            remains = remains1;
        }
    }

    Ok((
        Coproduct {
            forall,
            constructors,
            associated_module: None,
        },
        remains,
    ))
}

fn parse_constructor(remains: &[Token]) -> ParseResult<Constructor<ParsingInfo>> {
    if let [T(TT::Identifier(name), _), remains @ ..] = remains {
        parse_constructor_signature(remains).map_value(|signature| Constructor {
            name: Identifier::new(name),
            signature,
        })
    } else {
        Err(ParseError::ExpectedTokenType(TT::Identifier(
            "<constructor2>".to_owned(),
        )))
    }
}

fn parse_constructor_signature(
    mut remains: &[Token],
) -> ParseResult<Vec<TypeExpression<ParsingInfo>>> {
    let mut boofer = vec![];

    while matches!(remains, [T(TT::LeftParen | TT::Identifier(..), ..), ..]) {
        let (term, rem) = parse_type_expression(remains)?;
        boofer.push(term);

        remains = if starts_with(&TT::RightParen, rem) {
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
fn parse_type_expression(remains: &[Token]) -> ParseResult<TypeExpression<ParsingInfo>> {
    let (prefix, remains) = parse_type_expression_prefix(remains)?;
    parse_type_expression_infix(prefix, remains)
}

fn parse_type_expression_prefix(remains: &[Token]) -> ParseResult<TypeExpression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(id), pos), remains @ ..] => Ok((
            parse_simple_type_expression_term(ParsingInfo::new(*pos), id),
            remains,
        )),

        [T(TT::LeftParen, ..), ..] => parse_type_expression(&remains[1..]),

        _otherwise => Err(ParseError::ExpectedTokenType(TT::Identifier(
            "<Constructor>".to_owned(),
        ))),
    }
}

fn parse_simple_type_expression_term(pi: ParsingInfo, id: &str) -> TypeExpression<ParsingInfo> {
    if is_lowercasse(id) {
        TypeExpression::Parameter(pi, Identifier::new(id))
    } else {
        TypeExpression::Constructor(pi, Identifier::new(id))
    }
}

fn is_lowercasse(id: &str) -> bool {
    id.chars().all(char::is_lowercase)
}

fn parse_type_expression_infix(
    lhs: TypeExpression<ParsingInfo>,
    remains: &[Token],
) -> ParseResult<TypeExpression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(rhs), pos), remains @ ..] => {
            let pi = ParsingInfo::new(*pos);
            parse_type_expression_infix(
                TypeExpression::Apply(
                    pi,
                    TypeApply {
                        constructor: lhs.into(),
                        argument: parse_simple_type_expression_term(pi, rhs).into(),
                    },
                ),
                remains,
            )
        }

        [T(TT::Arrow, pos), remains @ ..] => {
            let pi = ParsingInfo::new(*pos);
            parse_type_expression(remains).map_value(|rhs| {
                TypeExpression::Arrow(
                    pi,
                    Arrow {
                        domain: lhs.into(),
                        codomain: rhs.into(),
                    },
                )
            })
        }

        // lookahead(1) to see if there is an arrow coming
        // RightParen handled in infix so that we can return a null infix
        [T(TT::RightParen, ..), T(TT::Arrow, ..), ..] => Ok((lhs, &remains[1..])),

        _otherwise => Ok((lhs, remains)),
    }
}

fn parse_value_binding<'a>(
    binder: &str,
    type_signature: Option<TypeSignature<ParsingInfo>>,
    position: &SourceLocation,
    remains: &'a [Token],
) -> ParseResult<'a, Declaration<ParsingInfo>> {
    parse_value_declarator(
        binder,
        strip_if_starts_with(TT::Layout(Layout::Indent), remains),
    )
    .map_value(|declarator| {
        Declaration::Value(
            ParsingInfo::new(*position),
            ValueDeclaration {
                binder: Identifier::new(binder),
                type_signature,
                declarator,
            },
        )
    })
}

fn parse_value_declarator<'a>(
    binder: &str,
    input: &'a [Token],
) -> ParseResult<'a, ValueDeclarator<ParsingInfo>> {
    parse_expression(input, 0).map_value(|expression| {
        let expression =
            if let Expression::Lambda(annotation, Lambda { parameter, body }) = expression {
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

        ValueDeclarator { expression }
    })
}

// Should this function eat the -> ?
// a | pattern
// This loses the annotation. Put it back. Later :)
fn parse_parameter_list(remains: &[Token]) -> ParseResult<Vec<Parameter>> {
    let end = remains
        .iter()
        .position(|t| t.token_type() == &TT::Period)
        .ok_or(ParseError::ExpectedTokenType(TT::Period))?;
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

pub fn parse_expression_phrase(tokens: &[Token]) -> Result<Expression<ParsingInfo>, ParseError> {
    let (expression, remains) = parse_expression(tokens, 0)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

pub fn parse_declaration_phrase(tokens: &[Token]) -> Result<Declaration<ParsingInfo>, ParseError> {
    let (expression, remains) = parse_declaration(tokens)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

fn parse_expression_prefix(tokens: &[Token]) -> ParseResult<Expression<ParsingInfo>> {
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
            Expression::Variable(ParsingInfo::new(*position), Identifier::new(id)),
            remains,
        )),
        [T(TT::LeftParen, ..), remains @ ..] => {
            let (expr, remains) = parse_expression(remains, 0)?;
            Ok((expr, expect(&TT::RightParen, remains)?))
        }
        [T(TT::LeftBrace, pos), remains @ ..] => {
            let (struct_literal, remains) = parse_struct_literal(*pos, remains)?;
            Ok((struct_literal, expect(&TT::RightBrace, remains)?))
        }
        otherwise => panic!("{otherwise:?}"),
    }
}

fn parse_struct_literal(
    pos: SourceLocation,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    let mut fields = vec![];

    let (field, remains) =
        parse_struct_field_initializer(strip_if_starts_with(TT::Layout(Layout::Indent), remains))?;
    fields.push(field);

    let mut remains = remains;

    if let [T(separator @ (TT::Semicolon | TT::Layout(Layout::Newline)), ..), ..] = remains {
        while matches!(remains, [T(t, ..), T(lookahead, ..), ..] if t == separator && lookahead != &TT::RightBrace)
        {
            let (field, remains1) = parse_struct_field_initializer(&remains[1..])?;
            fields.push(field);
            remains = remains1;

            if starts_with(&TT::RightBrace, remains1) {
                break;
            }
        }
    }

    Ok((
        Expression::Product(ParsingInfo::new(pos), Product::Struct(fields)),
        remains,
    ))
}

fn parse_struct_field_initializer(
    remains: &[Token],
) -> ParseResult<(Identifier, Expression<ParsingInfo>)> {
    if let [T(TT::Identifier(field_name), _field_name_pos), T(TT::Colon, ..), remains @ ..] =
        remains
    {
        let identifier = Identifier::new(field_name);
        let (initializer, remains) = parse_expression(remains, 0)?;

        Ok(((identifier, initializer), remains))
    } else {
        Err(ParseError::ExpectedTokenType(TT::Identifier(
            "field name".to_owned(),
        )))
    }
}

fn parse_lambda(
    position: SourceLocation,
    tokens: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    let (parameters, remains) = parse_parameter_list(tokens)?;
    let remains = expect(&TT::Period, remains)?;

    parse_expression(strip_if_starts_with(TT::Layout(Layout::Indent), remains), 0).map_value(
        |body| {
            parameters.into_iter().rfold(body, |body, parameter| {
                Expression::Lambda(
                    ParsingInfo::new(position),
                    Lambda {
                        parameter,
                        body: body.into(),
                    },
                )
            })
        },
    )
}

// This function is __TERRIBLE__.
fn parse_if_expression(
    position: SourceLocation,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
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

            parse_expression(&remains[0..], 0).map_value(|alternate| {
                Expression::ControlFlow(
                    ParsingInfo::new(position),
                    ControlFlow::If {
                        predicate: predicate.into(),
                        consequent: consequent.into(),
                        alternate: alternate.into(),
                    },
                )
            })
        } else {
            Err(ParseError::ExpectedTokenType(TokenType::Keyword(Else)))
        }
    } else {
        Err(ParseError::ExpectedTokenType(TokenType::Keyword(Then)))
    }
}

fn parse_deconstruct_into(
    position: SourceLocation,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    let (scrutinee, remains) = parse_expression(remains, 0)?;

    let mut remains = expect(
        &TT::Keyword(Into),
        strip_if_starts_with(TT::Layout(Layout::Indent), remains),
    )?;

    let mut match_clauses = vec![];

    let (match_clause, remains1) = parse_match_clause(strip_if_starts_with(
        TT::Layout(Layout::Indent),
        strip_if_starts_with(TT::Layout(Layout::Newline), remains),
    ))?;
    remains = strip_if_starts_with(TT::Layout(Layout::Dedent), remains1);
    remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
    match_clauses.push(match_clause);

    while matches!(remains, [T(TT::Pipe, ..), ..]) {
        let (match_clause, remains1) = parse_match_clause(&remains[1..])?;
        match_clauses.push(match_clause);
        remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains1);
        remains = strip_if_starts_with(TT::Layout(Layout::Dedent), remains);
    }

    Ok((
        Expression::DeconstructInto(
            ParsingInfo::new(position),
            DeconstructInto {
                scrutinee: scrutinee.into(),
                match_clauses,
            },
        ),
        remains,
    ))
}

fn parse_match_clause(remains: &[Token]) -> ParseResult<MatchClause<ParsingInfo>> {
    let (pattern, remains) = parse_pattern(remains)?;
    let remains = expect(&TT::Arrow, remains)?;

    parse_expression(strip_if_starts_with(TT::Layout(Layout::Indent), remains), 0).map_value(
        |consequent| MatchClause {
            pattern,
            consequent: consequent.into(),
        },
    )
}

fn parse_pattern(remains: &[Token]) -> ParseResult<Pattern<ParsingInfo>> {
    // (1, 2, 3) -> print_endline "1, 2, 3"` XX tuple (flattened?)
    // This x -> print_endline (show x)      XX Coproduct constructor
    // others -> print_endline (show others) XX catch all
    match remains {
        [T(TT::Identifier(id), position), ..] if is_capital_case(id) => {
            parse_constructor_pattern(remains, position)
        }

        [T(TT::Identifier(id), position), remains @ ..] => Ok((
            Pattern::Otherwise(ParsingInfo::new(*position), Identifier::new(id)),
            remains,
        )),

        [T(TT::Literal(lit), position), remains @ ..] => Ok((
            Pattern::Literally(ParsingInfo::new(*position), lit.clone().into()),
            remains,
        )),

        [T(TT::LeftParen, position), remains @ ..] => parse_tuple_pattern(remains, position)
            .map_value(|tuple| Pattern::Tuple(ParsingInfo::new(*position), tuple)),

        [T(TT::LeftBrace, position), remains @ ..] => parse_struct_pattern(remains, position)
            .map_value(|struct_pattern| {
                Pattern::Struct(ParsingInfo::new(*position), struct_pattern)
            }),

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
        parse_pattern_list(remains).map_value(|patterns| {
            Pattern::Coproduct(
                ParsingInfo::new(*position),
                ConstructorPattern {
                    constructor: Identifier::new(constructor),
                    argument: TuplePattern { elements: patterns },
                },
            )
        })
    } else {
        panic!("{remains:?}")
    }
}

fn parse_pattern_list(mut remains: &[Token]) -> ParseResult<Vec<Pattern<ParsingInfo>>> {
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
    let mut elements = vec![];

    let (pattern, remains1) = parse_pattern(remains)?;
    remains = remains1;
    elements.push(pattern);

    while matches!(remains, [T(TT::Comma, ..), ..]) {
        let (pattern, remains1) = parse_pattern(&remains[1..])?;
        elements.push(pattern);
        remains = remains1;
    }

    remains = expect(&TT::RightParen, remains)?;

    Ok((TuplePattern { elements }, remains))
}

fn parse_struct_pattern<'a>(
    mut remains: &'a [Token],
    _position: &SourceLocation,
) -> ParseResult<'a, StructPattern<ParsingInfo>> {
    let mut fields = vec![];

    let (field, remains1) = parse_struct_pattern_field(remains)?;
    remains = remains1;
    fields.push(field);

    while matches!(remains, [T(TT::Semicolon, ..), ..]) {
        remains = expect(&TT::Semicolon, remains)?;

        let (field, remains1) = parse_struct_pattern_field(remains)?;
        fields.push(field);
        remains = remains1;
    }

    remains = expect(&TT::RightBrace, remains)?;

    Ok((StructPattern { fields }, remains))
}

fn parse_struct_pattern_field(
    remains: &[Token],
) -> ParseResult<(Identifier, Pattern<ParsingInfo>)> {
    if let [T(TT::Identifier(id), _pos), remains @ ..] = remains {
        let identifier = Identifier::new(id);
        match remains {
            [T(TT::Colon, ..), remains @ ..] => {
                let (pattern, remains) = parse_pattern(remains)?;
                Ok(((identifier, pattern), remains))
            }

            [T(TT::Semicolon | TT::RightBrace, position), ..] => Ok((
                (
                    identifier.clone(),
                    Pattern::Otherwise(ParsingInfo::new(*position), identifier),
                ),
                remains,
            )),

            otherwise => Err(ParseError::Expected {
                one_of: vec![TT::Colon, TT::Semicolon].into(),
                received: otherwise[0].token_type().clone(),
            }),
        }
    } else {
        // This pattern is sad
        Err(ParseError::ExpectedTokenType(TT::Identifier(
            "struct field".to_owned(),
        )))
    }
}

fn expect<'a>(token_type: &TokenType, remains: &'a [Token]) -> Result<&'a [Token], ParseError> {
    if starts_with(token_type, remains) {
        Ok(&remains[1..])
    } else {
        Err(ParseError::ExpectedTokenType(token_type.clone()))
    }
}

fn starts_with(tt: &TokenType, prefix: &[Token]) -> bool {
    matches!(&prefix, &[t, ..] if t.token_type() == tt)
}

fn strip_if_starts_with(tt: TokenType, prefix: &[Token]) -> &[Token] {
    if matches!(&prefix, &[t, ..] if t.token_type() == &tt) {
        &prefix[1..]
    } else {
        prefix
    }
}

fn strip_first_if(condition: bool, input: &[Token]) -> &[Token] {
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
    let indented = starts_with(&TokenType::Layout(Layout::Indent), input);
    let (bound, remains) = parse_expression(strip_first_if(indented, input), 0)?;

    let dedented = starts_with(&TokenType::Layout(Layout::Dedent), remains);
    let newlined = starts_with(&TokenType::Layout(Layout::Newline), remains);
    match strip_first_if(indented && dedented || newlined, remains) {
        [T(TT::Keyword(In), ..), remains @ ..] => {
            let remains = strip_if_starts_with(TT::Layout(Layout::Indent), remains);
            let remains = strip_if_starts_with(TT::Layout(Layout::Newline), remains);

            parse_expression(remains, 0).map_value(|body| {
                Expression::Binding(
                    ParsingInfo::new(position),
                    Binding {
                        binder: Identifier::new(binder),
                        bound: bound.into(),
                        body: body.into(),
                    },
                )
            })
        }
        // In could be offside here, then what? What can I return or do?
        // Well, that is an error then I guess.
        otherwise => panic!("{otherwise:?}"),
    }
}

pub fn parse_expression(
    input: &[Token],
    precedence: usize,
) -> ParseResult<Expression<ParsingInfo>> {
    let (prefix, remains) = parse_expression_prefix(input)?;

    parse_expression_infix(prefix, remains, precedence)
}

fn is_expression_terminator(t: &Token) -> bool {
    matches!(
        t.token_type(),
        TT::Keyword(Keyword::In | Keyword::Else | Keyword::Then | Keyword::Into)
            | TT::End
            | TT::Layout(Layout::Dedent)
            | TT::RightParen
            | TT::RightBrace
            | TT::Pipe,
    )
}

// Infixes end with End and In
fn parse_expression_infix(
    lhs: Expression<ParsingInfo>,
    input: &[Token],
    precedence: usize,
) -> ParseResult<Expression<ParsingInfo>> {
    // Operators can be prefigured by Layout::Newline
    // Juxtapositions though? I would have to be able to ask the Expression
    // about where it started
    match input {
        [t, ..] if is_expression_terminator(t) => Ok((lhs, input)),

        [T(TT::Layout(..), ..), lookahead, ..] if is_expression_terminator(lookahead) => {
            Ok((lhs, input))
        }

        // ( <Newline> | <;> ) <expr>
        // -- an expression sequence, e.g.: <statement>* <expr>
        [T(TT::Layout(Layout::Newline) | TT::Semicolon, ..), lookahead @ ..]
            if !input.is_empty() =>
        {
            if !starts_with(&TT::End, &input[1..]) && is_sequence_prefix(lookahead) {
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
        [T(token, ..), lookahead, ..]
            if is_expression_prefix(token)
                && !matches!(lookahead, T(TT::TypeAscribe | TT::Equals, ..)) =>
        {
            parse_juxtaposed(lhs, input, precedence)
        }

        _otherwise => Ok((lhs, input)),
    }
}

fn is_expression_prefix(tt: &TokenType) -> bool {
    !matches!(
        tt,
        TT::Layout(Layout::Dedent) | TT::End | TT::Keyword(And | Or | Xor | Else | Into | In)
    )
}

// This is an annoying function
fn is_sequence_prefix(prefix: &[Token]) -> bool {
    !matches!(
        prefix,
        [
            T(TT::Identifier(..), ..),
            T(
                TT::Equals | TT::TypeAssign | TT::TypeAscribe | TT::Colon,
                ..
            ),
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
        let (lhs, remains) = if operator == &Operator::Select {
            parse_select_operator(lhs, remains)?
        } else {
            parse_operator_default(lhs, operator, remains, position, operator_precedence)?
        };

        parse_expression_infix(lhs, remains, context_precedence)
    } else {
        Ok((lhs, input))
    }
}

//enum Selector<A> {
//    Name(Identifier),
//    Member(ModuleName, Identifier),
//    Project(Project<A>),
//}

fn parse_select_operator(
    lhs: Expression<ParsingInfo>,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    // It is not this simple. Structs are most often going
    // to be accessed through a variable. That is not a stable
    // identifier. The parser cannot know what that is.
    //
    // Then again: Modules are structs in both the TypingContext and
    //             the interpreter Environment.
    //
    // A.b.c is: reduce(project(reduce(project(reduce(Var(A)), b), c)))
    //
    // Hmm, so they should all be Projections? But rename Projection into Select with Selector

    match lhs {
        Expression::Variable(parsing_info, identifier) => {
            parse_identifier_path(parsing_info, identifier, remains)
        }
        lhs => parse_projection(lhs, remains),
    }
}

fn parse_identifier_path(
    annotation: ParsingInfo,
    lhs: Identifier,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(name), _pos), remains @ ..] => Ok((
            Expression::Variable(annotation, Identifier::Select(lhs.into(), name.to_owned())),
            remains,
        )),
        _otherwise => Err(ParseError::Expected {
            // I need a way to talk about a token's actual type
            one_of: vec![
                TT::Identifier("field-name".to_owned()),
                TT::Literal(Literal::Integer(0)),
            ]
            .into(),
            received: remains[0].token_type().clone(),
        })?,
    }
}

fn parse_projection(
    lhs: Expression<ParsingInfo>,
    remains: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
    match remains {
        [T(TT::Identifier(name), pos), remains @ ..] => {
            Ok(((ProductIndex::Struct(Identifier::new(name)), pos), remains))
        }
        [T(TT::Literal(Literal::Integer(index)), pos), remains @ ..] => {
            Ok(((ProductIndex::Tuple(*index as usize), pos), remains))
        }
        _otherwise => Err(ParseError::Expected {
            // I need a way to talk about a token's actual type
            one_of: vec![
                TT::Identifier("field-name".to_owned()),
                TT::Literal(Literal::Integer(0)),
            ]
            .into(),
            received: remains[0].token_type().clone(),
        })?,
    }
    .map_value(|(index, pos)| {
        Expression::Project(
            ParsingInfo::new(*pos),
            Project {
                base: lhs.into(),
                index,
            },
        )
    })
}

fn parse_operator_default<'a>(
    lhs: Expression<ParsingInfo>,
    operator: &Operator,
    remains: &'a [Token],
    position: SourceLocation,
    operator_precedence: usize,
) -> ParseResult<'a, Expression<ParsingInfo>> {
    parse_expression(
        remains,
        if operator.is_right_associative() {
            operator_precedence - 1
        } else {
            operator_precedence
        },
    )
    .map_value(|rhs| apply_infix(position, lhs, *operator, rhs))
}

fn parse_juxtaposed(
    lhs: Expression<ParsingInfo>,
    tokens: &[Token],
    precedence: usize,
) -> ParseResult<Expression<ParsingInfo>> {
    // I wonder if this shouldn't be just parse_expession
    let (rhs, remains) = parse_expression_prefix(tokens)?;

    // parse continuing arguments
    parse_expression_infix(
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

fn parse_sequence(
    lhs: Expression<ParsingInfo>,
    tokens: &[Token],
) -> ParseResult<Expression<ParsingInfo>> {
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

fn apply_infix(
    position: SourceLocation,
    lhs: Expression<ParsingInfo>,
    op: Operator,
    rhs: Expression<ParsingInfo>,
) -> Expression<ParsingInfo> {
    let apply_lhs = Expression::Apply(
        ParsingInfo::new(position),
        Apply {
            function: Expression::Variable(ParsingInfo::new(position), op.as_identifier()).into(),
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
                                    function: E::Variable((), Id::new(&Operator::Plus.name()))
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
            // This semicolon interacts weirdly with
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

    #[test]
    fn type_expression_with_functions() {}
}
