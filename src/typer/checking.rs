use super::{
    unification::{unify, Substitutions},
    BaseType, ProductType, Type, Typing, TypingContext, UntypedExpression,
};
use crate::{
    ast::{
        Apply, Binding, Constant, ControlFlow, Expression, Identifier, Lambda, Parameter, Product,
        SelfReferential, Sequence, Variable,
    },
    parser::ParsingInfo,
    typer::{TupleType, TypeError, TypeScheme},
};

pub fn check(
    expression: &UntypedExpression,
    expected_type: &Type,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let annotation = *expression.annotation();

    match (expected_type, expression) {
        (
            expected,
            Expression::Variable(_, Variable::Identifier(id))
            | Expression::InvokeBridge(_, Variable::Identifier(id)),
        ) => {
            let actual_type = ctx
                .lookup(&id.clone().into())
                .ok_or_else(|| TypeError::UndefinedSymbol(id.clone()))?
                .instantiate(ctx)?;
            expected.unify(&actual_type, annotation)
        }
        (expected, Expression::Literal(pi, constant)) => {
            expected.unify(&constant.synthesize_type()?.inferred_type, *pi)
        }
        (
            expected @ Type::Arrow(ref parameter_type, ref body_type),
            Expression::SelfReferential(
                _,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ),
        ) => {
            let mut ctx = ctx.clone();
            ctx.bind(
                name.clone().into(),
                TypeScheme::from_constant(expected.clone()),
                //                TypeScheme::from_constant(Type::Arrow(
                //                    Type::fresh().into(),
                //                    expected.clone().into(),
                //                )),
            );

            ctx.bind(
                parameter.name.clone().into(),
                TypeScheme::from_constant(*parameter_type.clone()),
            );
            check(body, body_type, &ctx)
        }
        (expected, Expression::Lambda(pi, Lambda { parameter, body }))
            if parameter.type_annotation.is_some() =>
        {
            check_lambda(
                *pi,
                expected,
                &parameter.name,
                parameter.type_annotation.as_ref().expect("type annotation"),
                body,
                ctx,
            )
        }
        (expected, Expression::Apply(pi, Apply { function, argument })) => {
            check_apply(*pi, expected, function, argument, ctx)
        }
        (Type::Product(expected_type), Expression::Product(annotation, product)) => {
            check_product(*annotation, expected_type, product, ctx)
        }

        (expected, Expression::Binding(annotation, binding)) => {
            check_binding(*annotation, expected, binding, ctx)
        }
        (expected, Expression::Sequence(_, sequence)) => check_sequence(expected, sequence, ctx),
        (expected, Expression::ControlFlow(pi, control_flow)) => {
            check_control_flow(*pi, expected, control_flow, ctx)
        }
        //        (expected, Expression::DeconstructInto(_, deconstruct)) => {
        //            todo!()
        //        }
        //
        // This defaults to simple unifications, but what about TypeConstructor vs expanded type?
        //
        //        (expected, Expression::)
        (expected, expression) => {
            let pi = expression.annotation();
            let expression = ctx.infer_type(expression)?;
            unify(
                &expected.clone().apply(&expression.substitutions),
                &expression.inferred_type,
                *pi,
            )
        }
    }
}

fn check_sequence(
    expected: &Type,
    sequence: &Sequence<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let unification = check(&sequence.this, &Type::Constant(BaseType::Unit), ctx)?;
    check(
        &sequence.and_then,
        expected,
        &ctx.apply_substitutions(&unification),
    )
}

fn check_product(
    pi: ParsingInfo,
    expected_type: &ProductType,
    product: &Product<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let product = product.clone();
    match (expected_type.clone(), product) {
        (ProductType::Tuple(types), Product::Tuple(expressions)) => {
            let TupleType(types) = types.unspine();

            let mut s = Substitutions::default();
            for (ty, expr) in types.iter().zip(expressions.into_iter()) {
                s = s.compose(check(&expr, ty, &ctx.apply_substitutions(&s))?);
            }
            Ok(s)
        }

        (ProductType::Struct(mut lhs), Product::Struct(mut rhs)) if lhs.len() == rhs.len() => {
            lhs.sort_by(|(p, _), (q, _)| p.cmp(q));
            rhs.sort_by(|(p, _), (q, _)| p.cmp(q));

            let mut s = Substitutions::default();
            for ((lhs_label, expected), (rhs_label, expr)) in lhs.into_iter().zip(rhs.into_iter()) {
                if lhs_label == rhs_label {
                    s = s.compose(check(&expr, &expected, &ctx.apply_substitutions(&s))?);
                } else {
                    Err(TypeError::UndefinedField {
                        position: *pi.location(),
                        label: rhs_label,
                    })?;
                }
            }

            Ok(s)
        }

        (expected_type, product) => Err(TypeError::ExpectedType {
            expected_type: Type::Product(expected_type),
            literal: Expression::Product(pi, product),
        }
        .into()),
    }
}

fn check_binding(
    _pi: ParsingInfo,
    expected_type: &Type,
    binding: &Binding<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let Binding {
        binder,
        bound,
        body,
    } = binding;
    let bound = ctx.infer_type(bound)?;

    let mut ctx = ctx.clone();
    ctx.bind(binder.clone().into(), bound.inferred_type.generalize(&ctx));

    let unification = check(body, expected_type, &ctx)?;

    Ok(unification.compose(bound.substitutions))
}

fn check_lambda(
    pi: ParsingInfo,
    expected_type: &Type,
    parameter_name: &Identifier,
    parameter_type: &Type,
    body: &UntypedExpression,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let mut ctx = ctx.clone();
    ctx.bind(
        parameter_name.clone().into(),
        TypeScheme::from_constant(parameter_type.clone()),
    );
    let body = ctx.infer_type(body)?;

    let unification = expected_type.unify(
        &Type::Arrow(parameter_type.clone().into(), body.inferred_type.into()),
        pi,
    )?;

    Ok(body.substitutions.compose(unification))
}

fn check_apply(
    _pi: ParsingInfo,
    expected: &Type,
    function: &UntypedExpression,
    argument: &UntypedExpression,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let argument = ctx.infer_type(argument)?;
    let unification = check(
        function,
        &Type::Arrow(argument.inferred_type.into(), expected.clone().into()),
        ctx,
    )?;
    Ok(unification.compose(argument.substitutions))
}

fn check_control_flow(
    _pi: ParsingInfo,
    expected: &Type,
    control_flow: &ControlFlow<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    match control_flow {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            let s1 = check(predicate, &Type::Constant(BaseType::Bool), ctx)?;
            let s2 = check(consequent, expected, ctx)?;
            // Should I apply substitutions?
            let s3 = check(alternate, expected, ctx)?;
            Ok(s1.compose(s2.compose(s3)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_constants() {
        let typing_context = TypingContext::default();
        let parsing_info = ParsingInfo::default();
        check(
            &Expression::Literal(parsing_info, Constant::Int(1)),
            &Type::Constant(BaseType::Int),
            &typing_context,
        )
        .unwrap();
        check(
            &Expression::Literal(parsing_info, Constant::Bool(true)),
            &Type::Constant(BaseType::Bool),
            &typing_context,
        )
        .unwrap();
    }

    #[test]
    fn check_lambdas() {
        let ctx = TypingContext::default();
        let pi = ParsingInfo::default();
        let f = Expression::Lambda(
            pi,
            Lambda {
                parameter: Parameter::new_with_type_annotation(
                    Identifier::new("x"),
                    Type::Constant(BaseType::Int),
                )
                .into(),
                body: Expression::Variable(pi, Variable::Identifier(Identifier::new("x"))).into(),
            },
        );

        check(
            &f,
            &Type::Arrow(
                Type::Constant(BaseType::Int).into(),
                Type::Constant(BaseType::Int).into(),
            ),
            &ctx,
        )
        .unwrap();
    }

    #[test]
    fn check_applies() {
        let ctx = TypingContext::default();
        let pi = ParsingInfo::default();

        let f = Expression::Lambda(
            pi.clone(),
            Lambda {
                parameter: Parameter::new_with_type_annotation(
                    Identifier::new("x"),
                    Type::Constant(BaseType::Int),
                )
                .into(),
                body: Expression::Literal(pi, Constant::Text("Hi, mom".to_owned())).into(),
            },
        );

        let e = Expression::Apply(
            pi,
            Apply {
                function: f.into(),
                argument: Expression::Literal(pi, Constant::Int(2)).into(),
            },
        );

        check(&e, &Type::Constant(BaseType::Text), &ctx).unwrap();
    }
}
