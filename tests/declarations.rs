use std::marker::PhantomData;

use marmelade::{
    ast::{
        Constant, ConstructorPattern, Declaration, DeconstructInto, Expression, Forall, Identifier,
        MatchClause, Pattern, TuplePattern, TypeDeclaration, TypeName,
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

#[test]
fn pattern_match_eval_basic() {
    eval_fixture(
        r#"|main = deconstruct (1, 2) into (a, b) -> b
          "#,
        Value::Base(Base::Int(1)),
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
                        TuplePattern {
                            elements: vec![
                                Pattern::Literally(Constant::Int(1)),
                                Pattern::Literally(Constant::Int(2)),
                            ],
                        },
                        PhantomData::default(),
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
                        ConstructorPattern {
                            constructor: ident("This"),
                            argument: TuplePattern {
                                elements: vec![Pattern::Tuple(
                                    TuplePattern {
                                        elements: vec![
                                            Pattern::Otherwise(ident("x")),
                                            Pattern::Otherwise(ident("y")),
                                        ],
                                    },
                                    PhantomData::default(),
                                )],
                            },
                        },
                        PhantomData::default(),
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
                    Forall::default().add(TypeName::new("a")),
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
                    Forall::default().add(TypeName::new("a")),
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
                    Forall::default().add(TypeName::new("a")),
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
fn coproduct_eval() {
    let rhs = Declaration::Type(
        ParsingInfo::default(),
        TypeDeclaration {
            binding: ident("Eval"),
            declarator: coproduct(
                Forall::default()
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
