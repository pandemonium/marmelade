use marmelade::ast::{Declaration, TypeDeclaration};
use tools::*;

mod tools;

fn coproduct() {
    decl_fixture(
        r#"|Perhaps ::= This a | Nope
           "#,
        Declaration::Type(
            (),
            TypeDeclaration {
                binding: todo!(),
                declarator: todo!(),
            },
        ),
    );
}
