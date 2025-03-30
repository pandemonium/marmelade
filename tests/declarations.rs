use marmelade::{
    ast::{
        Arrow, Constant, ConstructorPattern, Declaration, DeconstructInto, Expression, Identifier,
        MatchClause, Pattern, Product, TuplePattern, TypeApply, TypeDeclaration, TypeExpression,
        TypeName, TypeSignature, UniversallyQuantified, ValueDeclaration, ValueDeclarator,
    },
    interpreter::{Base, Value},
    parser::ParsingInfo,
};
use tools::*;

mod tools;

#[test]
fn pattern_match_basic() {
    expr_fixture(
        r#"|deconstruct 1 into a_number -> 2
       "#,
        Expression::DeconstructInto(
            ParsingInfo::default(),
            DeconstructInto {
                scrutinee: int(1).into(),
                match_clauses: vec![MatchClause {
                    pattern: Pattern::Otherwise(ident("a_number")),
                    consequent: int(2).into(),
                }],
            },
        ),
    );
}

//#[test]
fn pattern_match_basic2() {
    expr_fixture(
        r#"|deconstruct 1, 2 into
           |  (a, b)    -> b
           || otherwise -> 3
           || (x, y)    -> x
          "#,
        Expression::DeconstructInto(
            ParsingInfo::default(),
            DeconstructInto {
                scrutinee: Expression::Product(
                    ParsingInfo::default(),
                    Product::Tuple(vec![int(1), int(2)]),
                )
                .into(),
                match_clauses: vec![
                    MatchClause {
                        pattern: Pattern::Tuple(
                            ParsingInfo::default(),
                            TuplePattern {
                                elements: vec![
                                    Pattern::Otherwise(ident("a")),
                                    Pattern::Otherwise(ident("b")),
                                ],
                            },
                        ),
                        consequent: var("b").into(),
                    },
                    MatchClause {
                        pattern: Pattern::Otherwise(ident("otherwise")),
                        consequent: int(3).into(),
                    },
                    MatchClause {
                        pattern: Pattern::Tuple(
                            ParsingInfo::default(),
                            TuplePattern {
                                elements: vec![
                                    Pattern::Otherwise(ident("x")),
                                    Pattern::Otherwise(ident("y")),
                                ],
                            },
                        ),
                        consequent: var("x").into(),
                    },
                ],
            },
        ),
    );
}

#[test]
fn pattern_match_eval_basic_piped_down() {
    eval_fixture(
        r#"|main =
           |  deconstruct 1, 2
           |    into (a, b) -> b
           |       | otherwise -> 3
           |       | (x, z)    -> z
          "#,
        Value::Base(Base::Int(2)),
    );
}

#[test]
fn pattern_match_eval_basic_inline() {
    eval_fixture(
        r#"|main =
           |  deconstruct 1, 2
           |    into (a, b) -> b | otherwise -> 3 | (x, z) -> z
          "#,
        Value::Base(Base::Int(2)),
    );
}

#[test]
fn pattern_match_tuple_match() {
    expr_fixture(
        r#"|deconstruct x into (1,2) -> 2
       "#,
        Expression::DeconstructInto(
            ParsingInfo::default(),
            DeconstructInto {
                scrutinee: var("x").into(),
                match_clauses: vec![MatchClause {
                    pattern: Pattern::Tuple(
                        ParsingInfo::default(),
                        TuplePattern {
                            elements: vec![
                                Pattern::Literally(Constant::Int(1)),
                                Pattern::Literally(Constant::Int(2)),
                            ],
                        },
                    ),
                    consequent: int(2).into(),
                }],
            },
        ),
    );
}
#[test]
fn pattern_match_constructor() {
    expr_fixture(
        r#"|deconstruct "x" into This (x, y) -> x
       "#,
        Expression::DeconstructInto(
            ParsingInfo::default(),
            DeconstructInto {
                scrutinee: text("x").into(),
                match_clauses: vec![MatchClause {
                    pattern: Pattern::Coproduct(
                        ParsingInfo::default(),
                        ConstructorPattern {
                            constructor: ident("This"),
                            argument: TuplePattern {
                                elements: vec![Pattern::Tuple(
                                    ParsingInfo::default(),
                                    TuplePattern {
                                        elements: vec![
                                            Pattern::Otherwise(ident("x")),
                                            Pattern::Otherwise(ident("y")),
                                        ],
                                    },
                                )],
                            },
                        },
                    ),
                    consequent: var("x").into(),
                }],
            },
        ),
    );
}

#[test]
fn coproduct_perhaps() {
    decl_fixture(
        r#"|Perhaps ::= forall a. This a | Nope
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("Perhaps"),
                declarator: coproduct(
                    UniversallyQuantified::default().add(TypeName::new("a")),
                    vec![
                        constructor("This", vec![typar("a")]),
                        constructor("Nope", vec![]),
                    ],
                ),
            },
        ),
    );
}

#[test]
fn coproduct_list() {
    decl_fixture(
        r#"|List ::= forall a. Cons a (List a) | Nil
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("List"),
                declarator: coproduct(
                    UniversallyQuantified::default().add(TypeName::new("a")),
                    vec![
                        constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
                        constructor("Nil", vec![]),
                    ],
                ),
            },
        ),
    );
}

#[test]
fn eval_coproduct_list() {
    eval_fixture(
        r#"|List ::= forall a. Cons a (List a) | Nil
           |main = Cons 1 Nil
           "#,
        Value::Coproduct {
            name: TypeName::new("List"),
            constructor: Identifier::new("Cons"),
            value: Value::Tuple(vec![
                Value::Base(Base::Int(1)),
                Value::Coproduct {
                    name: TypeName::new("List"),
                    constructor: Identifier::new("Nil"),
                    value: Value::Tuple(vec![]).into(),
                },
            ])
            .into(),
        },
    );
}

#[test]
fn eval_coproduct_eval() {
    eval_fixture(
        r#"|Eval ::= forall a e. Return a | Fault e
           |fail = Fault "hej"
           |main = Return 1
           "#,
        Value::Coproduct {
            name: TypeName::new("Eval"),
            constructor: Identifier::new("Return"),
            value: Value::Tuple(vec![Value::Base(Base::Int(1))]).into(),
        },
    );
}

#[test]
fn coproduct_binary_tree() {
    decl_fixture(
        r#"|BinaryTree ::= forall a. Branch (BinaryTree a) a (BinaryTree a) | Leaf a
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("BinaryTree"),
                declarator: coproduct(
                    UniversallyQuantified::default().add(TypeName::new("a")),
                    vec![
                        constructor(
                            "Branch",
                            vec![
                                tyapp(tyref("BinaryTree"), typar("a")),
                                typar("a"),
                                tyapp(tyref("BinaryTree"), typar("a")),
                            ],
                        ),
                        constructor("Leaf", vec![typar("a")]),
                    ],
                ),
            },
        ),
    );
}

#[test]
fn constant_type_expressions() {
    let pi = ParsingInfo::default();
    let rhs = Declaration::Value(
        pi,
        ValueDeclaration {
            binder: ident("length"),
            type_signature: Some(TypeSignature {
                quantifier: None,
                body: TypeExpression::Constructor(pi, Identifier::new("Int")),
            }),
            declarator: ValueDeclarator {
                expression: int(1).into(),
            },
        },
    );

    decl_fixture(r#"|length :: Int = 1"#, rhs);
}

#[test]
fn type_apply_type_expressions() {
    let pi = ParsingInfo::default();
    let rhs = Declaration::Value(
        pi,
        ValueDeclaration {
            binder: ident("length"),
            type_signature: Some(TypeSignature {
                quantifier: None,
                body: TypeExpression::Apply(
                    pi,
                    TypeApply {
                        constructor: TypeExpression::Constructor(pi, Identifier::new("List"))
                            .into(),
                        argument: TypeExpression::Constructor(pi, Identifier::new("Int")).into(),
                    },
                ),
            }),
            declarator: ValueDeclarator {
                expression: int(1).into(),
            },
        },
    );

    decl_fixture(r#"|length :: List Int = 1"#, rhs);
}

#[test]
fn type_arrow_expressions() {
    let pi = ParsingInfo::default();
    let rhs = Declaration::Value(
        pi,
        ValueDeclaration {
            binder: ident("length"),
            type_signature: Some(TypeSignature {
                quantifier: None,
                body: TypeExpression::Arrow(
                    pi,
                    Arrow {
                        domain: TypeExpression::Constructor(pi, Identifier::new("Int")).into(),
                        codomain: TypeExpression::Constructor(pi, Identifier::new("Text")).into(),
                    },
                ),
            }),
            declarator: ValueDeclarator {
                expression: int(1).into(),
            },
        },
    );

    decl_fixture(r#"|length :: Int -> Text = 1"#, rhs);
}

#[test]
fn complex_type_arrow_expressions() {
    let pi = ParsingInfo::default();
    let rhs = Declaration::Value(
        pi,
        ValueDeclaration {
            binder: ident("length"),
            type_signature: Some(TypeSignature {
                quantifier: None,
                body: TypeExpression::Arrow(
                    pi,
                    Arrow {
                        domain: TypeExpression::Apply(
                            pi,
                            TypeApply {
                                constructor: TypeExpression::Constructor(
                                    pi,
                                    Identifier::new("List"),
                                )
                                .into(),
                                argument: TypeExpression::Constructor(pi, Identifier::new("Int"))
                                    .into(),
                            },
                        )
                        .into(),
                        codomain: TypeExpression::Apply(
                            pi,
                            TypeApply {
                                constructor: TypeExpression::Constructor(
                                    pi,
                                    Identifier::new("Option"),
                                )
                                .into(),
                                argument: TypeExpression::Parameter(pi, Identifier::new("a"))
                                    .into(),
                            },
                        )
                        .into(),
                    },
                ),
            }),
            declarator: ValueDeclarator {
                expression: int(1).into(),
            },
        },
    );

    decl_fixture(r#"|length :: List Int -> Option a = 1"#, rhs);
}

#[test]
fn coproduct_eval() {
    let rhs = Declaration::Type(
        ParsingInfo::default(),
        TypeDeclaration {
            binding: ident("Eval"),
            declarator: coproduct(
                UniversallyQuantified::default()
                    .add(TypeName::new("a"))
                    .add(TypeName::new("e")),
                vec![
                    constructor("Return", vec![typar("a")]),
                    constructor("Fault", vec![typar("e")]),
                ],
            ),
        },
    );

    //    decl_fixture(
    //        r#"|Eval ::= forall a e. Return a | Fault e
    //           "#,
    //        rhs.clone(),
    //    );

    decl_fixture(
        r#"|Eval ::= forall a e.
           |  Return a
           |  Fault e
           "#,
        rhs,
    );
}

// Introduce type annotations for top-level types
// and do checks from there?
//
// "Cons" must find its type constructor.
// I could sort of cheat by looking up the function Cons
// and look at its return value, and infer that type.
//
// I could also work some more on modules and do the Module trick.
//
// Would it be acceptible if:
//   List ::= forall a. Cons a (List a) | Nil
// would create:
//   List.t as the type and
//   List.Cons and List.Nil as the functions?
//
// I also have to be able to annotate parameters with types and
// have that do something.
//
// Have to print the AST to see which of the Nil-positions that
// is getting applied to something. Did I ever do that ()-trick
// parameter for nullary constructors?
//#[test]
fn listy_mclistface() {
    eval_fixture(
        r#"|List ::= forall a. Cons a (List a) | Nil
           |
           |length = lambda base xs.
           |  deconstruct xs into
           |    Cons x xs -> 1 + length base xs
           |  | Nil       -> base
           |
           |map = lambda f xs.
           |  deconstruct xs into
           |    Cons x ys -> Cons (f x) (map f ys)
           |  | Nil       -> Nil
           |
           |fold_left = lambda z f xs.
           |  deconstruct xs into
           |    Cons x ys -> fold_left (f z ys) f ys
           |  | Nil       -> Nil
           |
           |length2 = fold_left 0 (lambda acc x. 1 + acc)
           |
           |output = lambda y. 1 + y
           |
           |main =
           |  let xs = Cons 1 (Cons 2 (Cons 3 (Cons 4 Nil)))
           |  in map show xs
           "#,
        Value::Base(Base::Int(4)),
    );
}

#[test]
fn multiple_arguments() {
    eval_fixture(
        r#"|List ::= forall a. Cons a (List a) | Nil
           |
           |map = lambda f a b c.
           |  print_endline (f a)
           |  print_endline (f b)
           |  print_endline (f c)
           |  Cons a (Cons b (Cons c Nil))
           |
           |main = let xs = Cons 1 (map show 2 3 4) in print_endline (show xs)"#,
        Value::Base(Base::Unit),
    );
}
