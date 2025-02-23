use marmelade::{
    ast::{Apply, ControlFlow, Expression, Identifier, Sequence},
    lexer::LexicalAnalyzer,
    parser,
};

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

use Expression as E;

fn fixture(source: &str, assert: fn(E<()>)) {
    let mut lexer = LexicalAnalyzer::default();
    let expr = parser::parse_expression_phrase(lexer.tokenize(&into_input(source)))
        .unwrap()
        .erase_annotation();
    assert(expr)
}

#[test]
fn app_sequence() {
    fixture(
        r#"|foo x
           |bar y
           |baz q
           "#,
        |expr| {
            assert_eq!(
                expr,
                seq(
                    apply(var("foo"), var("x")),
                    seq(apply(var("bar"), var("y")), apply(var("baz"), var("q"))),
                )
            )
        },
    );
}

#[test]
fn if_app() {
    fixture(
        r#"|foo x
           |if quux bar
           |  then baz q; lol
           |  else frobnicate
           "#,
        |expr| {
            assert_eq!(
                expr,
                seq(
                    apply(var("foo"), var("x")),
                    if_else(
                        apply(var("quux"), var("bar")),
                        seq(apply(var("baz"), var("q")), var("lol")),
                        var("frobnicate")
                    ),
                )
            )
        },
    );
}

fn if_else(predicate: E<()>, consequent: E<()>, alternate: E<()>) -> E<()> {
    E::ControlFlow(
        (),
        ControlFlow::If {
            predicate: predicate.into(),
            consequent: consequent.into(),
            alternate: alternate.into(),
        },
    )
}

fn seq(this: E<()>, and_then: E<()>) -> E<()> {
    E::Sequence(
        (),
        Sequence {
            this: this.into(),
            and_then: and_then.into(),
        },
    )
}

fn apply(f: E<()>, x: E<()>) -> E<()> {
    E::Apply(
        (),
        Apply {
            function: f.into(),
            argument: x.into(),
        },
    )
}

fn var(id: &str) -> E<()> {
    E::Variable((), Identifier::new(id))
}
