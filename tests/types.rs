use marmelade::{
    ast::TypeName,
    types::{CoproductType, ProductType, Type, TypeParameter},
};
use tools::*;

mod tools;

#[test]
fn list_type() {
    let rhs = coproduct(vec![
        constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
        constructor("Nil", vec![]),
    ]);

    let ty = Type::Parameter(TypeParameter::new_for_test(0));
    let lhs = Type::Coproduct(CoproductType::new(vec![
        (
            "Cons".to_owned(),
            Type::Product(ProductType::Tuple(vec![
                ty.clone(),
                Type::Apply(Type::Named(TypeName::new("List")).into(), ty.into()),
            ])),
        ),
        ("Nil".to_owned(), Type::Product(ProductType::Tuple(vec![]))), // hmm
    ]));

    assert_eq!(lhs, rhs.synthesize_type());
}
