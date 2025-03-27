use super::{
    unification::Substitutions, BaseType, Parsed, ProductType, Type, Typing, TypingContext,
};
use crate::{
    ast::{Apply, Binding, ControlFlow, Expression, Identifier, Lambda, Parameter, Product},
    parser::ParsingInfo,
    typer::{TypeError, TypeScheme},
};

type UntypedExpression = Expression<ParsingInfo>;

pub fn check(
    expression: UntypedExpression,
    expected_type: Type,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let annotation = *expression.annotation();
    match (expected_type, expression) {
        (expected, Expression::Variable(_, id) | Expression::InvokeBridge(_, id)) => {
            let actual_type = ctx
                .lookup(&id.clone().into())
                .ok_or_else(|| TypeError::UndefinedSymbol(id.clone()))?
                .instantiate(ctx)?;
            expected.unify(&actual_type, &annotation)
        }
        (expected, Expression::Literal(pi, constant)) => {
            expected.unify(&constant.synthesize_type()?.inferred_type, &pi)
        }
        (expected, Expression::SelfReferential(_, self_referential)) => todo!(),
        (
            expected,
            Expression::Lambda(
                pi,
                Lambda {
                    parameter:
                        Parameter {
                            name,
                            type_annotation: Some(parameter_type),
                        },
                    body,
                },
            ),
        ) => check_lambda(&pi, expected, name, parameter_type, body, ctx),
        (expected, Expression::Apply(pi, Apply { function, argument })) => {
            check_apply(&pi, expected, *function, *argument, ctx)
        }
        (expected, Expression::Inject(_, inject)) => todo!(),
        (Type::Product(expected_type), Expression::Product(annotation, product)) => {
            check_product(&annotation, expected_type, product, ctx)
        }
        (expected, Expression::Project(_, project)) => todo!(),
        (expected, Expression::Binding(annotation, binding)) => {
            check_binding(&annotation, expected, binding, ctx)
        }
        (expected, Expression::Sequence(_, sequence)) => todo!(),
        (expected, Expression::ControlFlow(pi, control_flow)) => {
            check_control_flow(&pi, expected, control_flow, ctx)
        }
        (expected, Expression::DeconstructInto(_, deconstruct_into)) => todo!(),
        _otherwise => todo!(),
    }
}

fn check_product(
    pi: &ParsingInfo,
    expected_type: ProductType,
    product: Product<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    match (expected_type, product) {
        (ProductType::Tuple(types), Product::Tuple(expressions))
            if types.len() == expressions.len() =>
        {
            let mut s = Substitutions::default();
            for (ty, expr) in types.into_iter().zip(expressions.into_iter()) {
                s = s.compose(check(expr, ty, &ctx.apply_substitutions(&s))?);
            }
            Ok(s)
        }

        (ProductType::Struct(mut lhs), Product::Struct(mut rhs)) if lhs.len() == rhs.len() => {
            lhs.sort_by(|(p, _), (q, _)| p.cmp(&q));
            rhs.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut s = Substitutions::default();
            for ((lhs_label, expected), (rhs_label, expr)) in lhs.into_iter().zip(rhs.into_iter()) {
                if lhs_label == rhs_label {
                    s = s.compose(check(expr, expected, &ctx.apply_substitutions(&s))?);
                } else {
                    Err(TypeError::UndefinedField {
                        position: *pi.info().location(),
                        label: rhs_label,
                    })?
                }
            }

            Ok(s)
        }

        (expected_type, product) => Err(TypeError::ExpectedType {
            expected_type: Type::Product(expected_type),
            literal: Expression::Product(pi.clone(), product),
        }),
    }
}

fn check_binding(
    _pi: &ParsingInfo,
    expected_type: Type,
    binding: Binding<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let Binding {
        binder,
        bound,
        body,
    } = binding;
    let bound = ctx.infer_type(&bound)?;

    let mut ctx = ctx.clone();
    ctx.bind(binder.into(), bound.inferred_type.generalize(&ctx));

    let unification = check(*body, expected_type, &ctx)?;

    Ok(unification.compose(bound.substitutions))
}

fn check_lambda(
    pi: &ParsingInfo,
    expected_type: Type,
    parameter_name: Identifier,
    parameter_type: Type,
    body: Box<UntypedExpression>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let mut ctx = ctx.clone();
    ctx.bind(
        parameter_name.clone().into(),
        TypeScheme::from_constant(parameter_type.clone()),
    );
    let body = ctx.infer_type(&body)?;

    let unification = expected_type.unify(
        &Type::Arrow(parameter_type.clone().into(), body.inferred_type.into()),
        pi,
    )?;

    Ok(body.substitutions.compose(unification))
}

fn check_apply(
    _pi: &ParsingInfo,
    expected: Type,
    function: UntypedExpression,
    argument: UntypedExpression,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    let argument = ctx.infer_type(&argument)?;
    let unification = check(
        function,
        Type::Arrow(argument.inferred_type.into(), expected.clone().into()),
        ctx,
    )?;
    Ok(unification.compose(argument.substitutions))
}

fn check_control_flow(
    _pi: &ParsingInfo,
    expected: Type,
    control_flow: ControlFlow<ParsingInfo>,
    ctx: &TypingContext,
) -> Typing<Substitutions> {
    match control_flow {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            let s1 = check(*predicate, Type::Constant(BaseType::Bool), ctx)?;
            let s2 = check(*consequent, expected.clone(), ctx)?;
            // Should I apply substitutions?
            let s3 = check(*alternate, expected, ctx)?;
            Ok(s1.compose(s2.compose(s3)))
        }
    }
}
