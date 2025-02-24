use marmelade::{
    ast::{Apply, Binding, Constant, ControlFlow, Expression, Identifier, Sequence},
    context::CompileState,
    interpreter::{Base, Value},
    lexer::LexicalAnalyzer,
    parser, stdlib,
};

fn into_unicode_text(source: &str) -> Vec<char> {
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

#[test]
fn app_sequence() {
    eq_fixture(
        r#"|foo x
           |bar y
           |baz q
           "#,
        seq(
            apply(var("foo"), var("x")),
            seq(apply(var("bar"), var("y")), apply(var("baz"), var("q"))),
        ),
    );
}

#[test]
fn if_block_sequencing() {
    eq_fixture(
        r#"|foo x
           |if quux bar
           |  then baz q; lol
           |  else frobnicate
           "#,
        seq(
            apply(var("foo"), var("x")),
            if_else(
                apply(var("quux"), var("bar")),
                seq(apply(var("baz"), var("q")), var("lol")),
                var("frobnicate"),
            ),
        ),
    );
}

#[test]
fn let_in_tests() {
    eq_fixture(
        r#"|let x = foo bar in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    eq_fixture(
        r#"|let x = foo bar in
           |quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    eq_fixture(
        r#"|let x = foo bar
           |in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    eq_fixture(
        r#"|let x =
           |  foo bar
           |in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    eq_fixture(
        r#"|let x =
           |  foo bar
           |in if quux
           |     then 1
           |     else "hi, mom"
           "#,
        let_in(
            "x",
            apply(var("foo"), var("bar")),
            if_else(var("quux"), int(1), text("hi, mom")),
        ),
    );

    eq_fixture(
        r#"|let x =
           |  foo bar; frobnicate the
           |in if quux
           |     then 1
           |     else "hi, mom"
           "#,
        let_in(
            "x",
            seq(
                apply(var("foo"), var("bar")),
                apply(var("frobnicate"), var("the")),
            ),
            if_else(var("quux"), int(1), text("hi, mom")),
        ),
    );

    eq_fixture(
        r#"|let x =
           |  foo bar
           |  frobnicate the
           |in if quux
           |     then
           |       1
           |       2
           |     else "hi, mom";3
           "#,
        let_in(
            "x",
            seq(
                apply(var("foo"), var("bar")),
                apply(var("frobnicate"), var("the")),
            ),
            if_else(
                var("quux"),
                seq(int(1), int(2)),
                seq(text("hi, mom"), int(3)),
            ),
        ),
    );
}

#[test]
fn precedence() {
    eq_fixture(
        r#"|a + b * c
           "#,
        apply(
            apply(var("+"), var("a")),
            apply(apply(var("*"), var("b")), var("c")),
        ),
    );

    eq_fixture(
        r#"|(a + b * c)
           "#,
        apply(
            apply(var("+"), var("a")),
            apply(apply(var("*"), var("b")), var("c")),
        ),
    );

    eq_fixture(
        r#"|a * b + c
           "#,
        apply(
            apply(var("+"), apply(apply(var("*"), var("a")), var("b"))),
            var("c"),
        ),
    );

    eq_fixture(
        r#"|(a * b) + c
           "#,
        apply(
            apply(var("+"), apply(apply(var("*"), var("a")), var("b"))),
            var("c"),
        ),
    );

    eq_fixture(
        r#"|a / (b + c)
           "#,
        apply(
            apply(var("/"), var("a")),
            apply(apply(var("+"), var("b")), var("c")),
        ),
    );

    eval_fixture(r#"|main = 1 + 2 * 3"#, 7);
    eval_fixture(r#"|main = 2 * 3 + 1"#, 7);
    eval_fixture(r#"|main = (1 + 2) * 3"#, 9);
    eval_fixture(r#"|main = (1 + 2) * 3 - 9"#, 0);
    eval_fixture(r#"|main = (1 + 2) * (3 - 9)"#, -18);
    eval_fixture(r#"|main = (3 - 9) / (1 + 2)"#, -2);
}

fn eq_fixture(source: &str, rhs: E<()>) {
    let mut lexer = LexicalAnalyzer::default();
    let lhs = parser::parse_expression_phrase(lexer.tokenize(&into_unicode_text(source)))
        .unwrap()
        .erase_annotation();
    assert_eq!(lhs, rhs)
}

fn eval_fixture<A>(source: &str, rhs: A)
where
    A: Into<Value>,
{
    let mut lexer = LexicalAnalyzer::default();
    let source_text = into_unicode_text(source);
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = CompileState::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let return_value = compilation.typecheck_and_interpret(program);

    assert_eq!(
        return_value.unwrap().try_into_base_type().unwrap(),
        rhs.into().try_into_base_type().unwrap()
    )
}

fn int(i: i64) -> E<()> {
    E::Literal((), Constant::Int(i))
}

fn text(s: &str) -> E<()> {
    E::Literal((), Constant::Text(s.to_owned()))
}

fn let_in(binder: &str, bound: E<()>, body: E<()>) -> E<()> {
    E::Binding(
        (),
        Binding {
            binder: Identifier::new(binder),
            bound: bound.into(),
            body: body.into(),
        },
    )
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
