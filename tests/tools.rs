use marmelade::{
    ast::{Apply, Binding, Constant, ControlFlow, Declaration, Expression, Identifier, Sequence},
    context::CompileState,
    interpreter::Value,
    lexer::LexicalAnalyzer,
    parser, stdlib,
};

use Expression as E;

pub fn decl_fixture(source: &str, rhs: Declaration<()>) {
    let mut lexer = LexicalAnalyzer::default();
    let lhs = parser::parse_declaration_phrase(lexer.tokenize(&into_unicode_text(source)))
        .unwrap()
        .map(|_| ());

    assert_eq!(lhs, rhs)
}

pub fn expr_fixture(source: &str, rhs: E<()>) {
    let mut lexer = LexicalAnalyzer::default();
    let lhs = parser::parse_expression_phrase(lexer.tokenize(&into_unicode_text(source)))
        .unwrap()
        .erase_annotation();

    assert_eq!(lhs, rhs)
}

pub fn eval_fixture<A>(source: &str, rhs: A)
where
    A: Into<Value>,
{
    let source_text = into_unicode_text(source);
    let mut compilation = CompileState::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let return_value = compilation.typecheck_and_interpret();

    assert_eq!(
        return_value.unwrap().try_into_base_type().unwrap(),
        rhs.into().try_into_base_type().unwrap()
    )
}
pub fn int(i: i64) -> E<()> {
    E::Literal((), Constant::Int(i))
}

pub fn text(s: &str) -> E<()> {
    E::Literal((), Constant::Text(s.to_owned()))
}

pub fn let_in(binder: &str, bound: E<()>, body: E<()>) -> E<()> {
    E::Binding(
        (),
        Binding {
            binder: Identifier::new(binder),
            bound: bound.into(),
            body: body.into(),
        },
    )
}

pub fn if_else(predicate: E<()>, consequent: E<()>, alternate: E<()>) -> E<()> {
    E::ControlFlow(
        (),
        ControlFlow::If {
            predicate: predicate.into(),
            consequent: consequent.into(),
            alternate: alternate.into(),
        },
    )
}

pub fn seq(this: E<()>, and_then: E<()>) -> E<()> {
    E::Sequence(
        (),
        Sequence {
            this: this.into(),
            and_then: and_then.into(),
        },
    )
}

pub fn apply(f: E<()>, x: E<()>) -> E<()> {
    E::Apply(
        (),
        Apply {
            function: f.into(),
            argument: x.into(),
        },
    )
}

pub fn var(id: &str) -> E<()> {
    E::Variable((), Identifier::new(id))
}

pub fn into_unicode_text(source: &str) -> Vec<char> {
    source
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .collect()
}
