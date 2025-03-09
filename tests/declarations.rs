use marmelade::{
    ast::{Declaration, Identifier, TypeDeclaration, TypeName},
    interpreter::{Base, Value},
    parser::ParsingInfo,
};
use tools::*;

mod tools;

#[test]
fn coproduct_perhaps() {
    decl_fixture(
        r#"|Perhaps ::= This a | Nope
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("Perhaps"),
                declarator: coproduct(vec![
                    constructor("This", vec![typar("a")]),
                    constructor("Nope", vec![]),
                ]),
            },
        ),
    );
}

#[test]
fn coproduct_list() {
    decl_fixture(
        r#"|List ::= Cons a (List a) | Nil
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("List"),
                declarator: coproduct(vec![
                    constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
                    constructor("Nil", vec![]),
                ]),
            },
        ),
    );
}

#[test]
fn eval_coproduct_list() {
    eval_fixture(
        r#"|List ::= Cons a (List a) | Nil
           |main = Cons 1 (Cons 2 Nil)
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
        r#"|Eval ::= Return a | Fault e
           |fail = Fault "hej"
           |main = Return 1
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
fn coproduct_binary_tree() {
    decl_fixture(
        r#"|BinaryTree ::= Branch (BinaryTree a) a (BinaryTree a) | Leaf a
           "#,
        Declaration::Type(
            ParsingInfo::default(),
            TypeDeclaration {
                binding: ident("BinaryTree"),
                declarator: coproduct(vec![
                    constructor(
                        "Branch",
                        vec![
                            tyapp(tyref("BinaryTree"), typar("a")),
                            typar("a"),
                            tyapp(tyref("BinaryTree"), typar("a")),
                        ],
                    ),
                    constructor("Leaf", vec![typar("a")]),
                ]),
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
            declarator: coproduct(vec![
                constructor("Return", vec![typar("a")]),
                constructor("Fault", vec![typar("e")]),
            ]),
        },
    );

    decl_fixture(
        r#"|Eval ::= Return a | Fault e
           "#,
        rhs.clone(),
    );

    decl_fixture(
        r#"|Eval ::=
           |  Return a
           |  Fault e
           "#,
        rhs,
    );
}
