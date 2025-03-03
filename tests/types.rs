use std::{collections::HashMap, marker::PhantomData};

use marmelade::{
    ast::{TypeApply, TypeDeclarator, TypeExpression, TypeName, ValueDeclarator},
    parser::ParsingInfo,
    typer::{Binding, CoproductType, ProductType, Type, TypeParameter, TypingContext},
};
use tools::*;

mod tools;

#[test]
fn list_type() {
    let rhs = coproduct(vec![
        constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
        constructor("Nil", vec![]),
    ]);

    let tp = TypeParameter::new_for_test(0);
    let ty = Type::Parameter(tp);
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
    let lhs = Type::Forall(tp, lhs.into());
    assert_eq!(lhs, rhs.synthesize_type());
}

#[test]
fn type_expansions() {
    let mut ctx = TypingContext::default();

    let list_declaration = coproduct(vec![
        constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
        constructor("Nil", vec![]),
    ]);

    let list_type = list_declaration.synthesize_type();
    ctx.bind(Binding::TypeTerm("List".to_owned()), list_type);

    if let TypeDeclarator::Coproduct(_, coproduct) = &list_declaration {
        for constructor in coproduct
            .make_implementation_module(ParsingInfo::default(), TypeName::new("List"))
            .constructors
        {
            println!("{}: {:?}", constructor.binder, constructor.declarator);
            let expr = match constructor.declarator {
                ValueDeclarator::Constant(decl) => decl.initializer,
                ValueDeclarator::Function(decl) => {
                    decl.into_lambda_tree(constructor.binder.clone())
                }
            };
            println!("{} impl: {expr:?}", constructor.binder);
            //            let typing = ctx.infer_type(&expr).unwrap();
            //
            //            println!("{}", typing.inferred_type);
        }
    }

    let abbreviation = TypeExpression::<()>::Apply(
        TypeApply {
            constructor: TypeExpression::Constant(TypeName::new("List")).into(),
            argument: TypeExpression::Parameter(TypeName::new("a")).into(),
        },
        PhantomData::default(),
    )
    .synthesize_type(&mut HashMap::default());

    assert!(abbreviation
        .expand(&ctx)
        .unwrap()
        .unify(
            &list_declaration
                .synthesize_type()
                .instantiate()
                .expand(&ctx)
                .unwrap(),
            &(),
            &ctx
        )
        .is_ok());
}
