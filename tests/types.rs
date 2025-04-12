use std::collections::HashMap;

use marmelade::{
    ast::{
        self, Apply, Expression, Identifier, Lambda, Parameter, Product, TypeApply, TypeDeclarator,
        TypeExpression, TypeName, UniversalQuantifiers,
    },
    context::Linkage,
    parser::ParsingInfo,
    stdlib,
    typer::{
        Binding, CoproductType, ProductType, Type, TypeInference, TypeParameter, TypeScheme,
        TypingContext,
    },
};
use tools::*;

mod tools;

#[test]
fn list_type() {
    let rhs = coproduct(
        UniversalQuantifiers::default().add(TypeName::new("a")),
        vec![
            constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
            constructor("Nil", vec![]),
        ],
    );

    let tp = TypeParameter::new_for_test(1);
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

    let lhs = TypeScheme {
        quantifiers: vec![tp],
        body: lhs,
    };

    assert_eq!(lhs, rhs.synthesize_type(&TypingContext::default()).unwrap());
}

#[test]
fn tuple() {
    let mut ctx = Linkage::default();
    stdlib::import(&mut ctx).unwrap();

    let lhs = Expression::Product(
        ParsingInfo::default(),
        Product::Tuple(vec![int(1), text("2"), float(3.0)]),
    );

    let rhs = Expression::Product(
        ParsingInfo::default(),
        Product::Tuple(vec![int(1), text("2"), float(3.0)]),
    );

    let cmp = Expression::Apply(
        ParsingInfo::default(),
        Apply {
            function: Expression::Apply(
                ParsingInfo::default(),
                Apply {
                    function: var("=").into(),
                    argument: lhs.into(),
                },
            )
            .into(),
            argument: rhs.into(),
        },
    );

    let TypeInference {
        substitutions,
        inferred_type,
    } = ctx.typing_context.infer_type(&cmp).unwrap();

    println!("Inferred: {substitutions:?}, {inferred_type}");
}

#[test]
fn type_expansions() {
    let mut ctx = TypingContext::default();

    let list_declaration = coproduct(
        UniversalQuantifiers::default().add(TypeName::new("a")),
        vec![
            constructor("Cons", vec![typar("a"), tyapp(tyref("List"), typar("a"))]),
            constructor("Nil", vec![]),
        ],
    );

    let list_type = list_declaration.synthesize_type(&ctx).unwrap();
    ctx.bind(Binding::TypeTerm("List".to_owned()), list_type);

    if let TypeDeclarator::Coproduct(_, coproduct) = &list_declaration {
        for constructor in coproduct
            .make_implementation_module(&ParsingInfo::default(), TypeName::new("List"), &ctx)
            .unwrap()
            .constructors
        {
            println!("{}: {:?}", constructor.binder, constructor.declarator);
            let expr = constructor.declarator.expression;
            println!("{} impl: {expr:?}", constructor.binder);
            //            let typing = ctx.infer_type(&expr).unwrap();
            //
            //            println!("{}", typing.inferred_type);
        }
    }

    let _abbreviation = TypeExpression::<ParsingInfo>::Apply(
        ParsingInfo::default(),
        TypeApply {
            constructor: TypeExpression::Constructor(
                ParsingInfo::default(),
                Identifier::new("List"),
            )
            .into(),
            argument: TypeExpression::Parameter(ParsingInfo::default(), Identifier::new("a"))
                .into(),
        },
    )
    .synthesize_type(
        &mut HashMap::from([(TypeName::new("a"), TypeParameter::fresh())]),
        &ctx,
    )
    .unwrap();

    //    println!(
    //        "-------> {:?}",
    //        abbreviation.expand(&ctx).unwrap().unify(
    //            &list_declaration.synthesize_type(),
    //            //                .instantiate()
    //            //                .expand(&ctx)
    //            //.unwrap(),
    //            &(),
    //            &ctx
    //        )
    //    );
}

#[test]
fn inferencing() {
    let ctx = TypingContext::default();

    let expr4 = Expression::Lambda(
        ParsingInfo::default(),
        Lambda {
            parameter: Parameter::new(ident("x")),
            body: Expression::Binding(
                ParsingInfo::default(),
                ast::Binding {
                    binder: ident("y"),
                    bound: var("x").into(),
                    body: var("y").into(),
                },
            )
            .into(),
        },
    );

    let expr5 = Expression::Apply(
        ParsingInfo::default(),
        Apply {
            function: expr4.clone().into(),
            argument: text("s").into(),
        },
    );

    println!("4. {}", ctx.infer_type(&expr4).unwrap());
    println!("5. {}", ctx.infer_type(&expr5).unwrap());
}
