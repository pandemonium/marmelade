use marmelade::{
    ast::{Declaration, Identifier, TypeDeclaration, TypeName},
    interpreter::{Base, Value},
};
use tools::*;

mod tools;

#[test]
fn coproduct_perhaps() {
    decl_fixture(
        r#"|Perhaps ::= This a | Nope
           "#,
        Declaration::Type(
            (),
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
            (),
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
    // try_initialize blows up because it cannot type main. No such
    // function Cons.
    eval_fixture(
        r#"|List ::= Cons a (List a) | Nil
           |main = Cons 1 Nil
           "#,
        Value::Coproduct {
            name: TypeName::new("List"),
            constructor: Identifier::new("Cons"),
            value: Value::Base(Base::Int(1)).into(),
        },
    );
}

#[test]
fn coproduct_binary_tree() {
    decl_fixture(
        r#"|BinaryTree ::= Branch (BinaryTree a) a (BinaryTree a) | Leaf a
           "#,
        Declaration::Type(
            (),
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
        (),
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
