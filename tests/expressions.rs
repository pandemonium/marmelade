mod tools;

use tools::*;

#[test]
fn app_sequence() {
    expr_fixture(
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
    expr_fixture(
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
    expr_fixture(
        r#"|let x = foo bar in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    expr_fixture(
        r#"|let x = foo bar in
           |quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    expr_fixture(
        r#"|let x = foo bar
           |in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    expr_fixture(
        r#"|let x =
           |  foo bar
           |in quux
           "#,
        let_in("x", apply(var("foo"), var("bar")), var("quux")),
    );

    expr_fixture(
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

    expr_fixture(
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

    expr_fixture(
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
    expr_fixture(
        r#"|a + b * c
           "#,
        apply(
            apply(var("+"), var("a")),
            apply(apply(var("*"), var("b")), var("c")),
        ),
    );

    expr_fixture(
        r#"|(a + b * c)
           "#,
        apply(
            apply(var("+"), var("a")),
            apply(apply(var("*"), var("b")), var("c")),
        ),
    );

    expr_fixture(
        r#"|a * b + c
           "#,
        apply(
            apply(var("+"), apply(apply(var("*"), var("a")), var("b"))),
            var("c"),
        ),
    );

    expr_fixture(
        r#"|(a * b) + c
           "#,
        apply(
            apply(var("+"), apply(apply(var("*"), var("a")), var("b"))),
            var("c"),
        ),
    );

    expr_fixture(
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
    eval_fixture(r#"|main = ((3 - 9) / (1 + 2))"#, -2);

    eval_fixture(r#"|main = 1 = 2"#, false);
    eval_fixture(r#"|main = 1 = 1"#, true);
    eval_fixture(r#"|main = 1 >= 2"#, false);
    eval_fixture(r#"|main = 1 <= 2"#, true);

    eval_fixture(r#"|main = 1 <= 2 or 1 >= 2"#, true);
    eval_fixture(r#"|main = 1 <= 2 xor 1 >= 2"#, true);
    eval_fixture(r#"|main = 1 >= 2 or 1 <= 2"#, true);
    eval_fixture(r#"|main = 1 >= 2 xor 1 <= 2"#, true);
    eval_fixture(r#"|main = 1 <= 2 and 1 >= 2"#, false);
    eval_fixture(r#"|main = 1 <= 2 and 1 >= 0"#, true);
    eval_fixture(r#"|main = 1 <= 2 xor 1 >= 0"#, false);
}

#[test]
fn tuples() {
    //    eval_fixture(r#"|main = 1,2"#, (1, 2));
    //    eval_fixture(r#"|main = (1,2,3,4,5,6) = (1,2,3,4,5,6)"#, true);
    eval_fixture(
        r#"|main = (1,2) = (1,2,"3")
        "#,
        true,
    );
    //    expr_fixture(r#"|(1,2) = (1,2)"#, cmp);
}
