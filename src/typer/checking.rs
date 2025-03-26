use std::fmt;

use super::{unification::Substitutions, BaseType, Parsed, Type, Typing, TypingContext};
use crate::{
    ast::{Apply, Binding, ControlFlow, Expression, Identifier, Lambda, Parameter},
    typer::{TypeError, TypeScheme},
};

pub fn check<A>(
    annotation: &A,
    expr: Expression<A>,
    expected_type: &Type,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed + fmt::Display + Clone,
{
    match (expected_type, expr) {
        (expected, Expression::Variable(_, id) | Expression::InvokeBridge(_, id)) => {
            let actual_type = ctx
                .lookup(&id.clone().into())
                .ok_or_else(|| TypeError::UndefinedSymbol(id))?
                .instantiate(ctx)?;
            expected.unify(&actual_type, annotation)
        }
        (expected, Expression::Literal(_, constant)) => {
            expected.unify(&constant.synthesize_type()?.inferred_type, annotation)
        }
        (expected, Expression::SelfReferential(_, self_referential)) => todo!(),
        (
            expected,
            Expression::Lambda(
                _,
                Lambda {
                    parameter:
                        Parameter {
                            name,
                            type_annotation: Some(parameter_type),
                        },
                    body,
                },
            ),
        ) => check_lambda(annotation, expected, name, parameter_type, body, ctx),
        (expected, Expression::Apply(_, Apply { function, argument })) => {
            check_apply(annotation, expected, function, argument, ctx)
        }
        (expected, Expression::Inject(_, inject)) => todo!(),
        (expected, Expression::Product(_, product)) => todo!(),
        (expected, Expression::Project(_, project)) => todo!(),
        (expected, Expression::Binding(annotation, binding)) => {
            check_binding(&annotation, expected, binding, ctx)
        }
        (expected, Expression::Sequence(_, sequence)) => todo!(),
        (expected, Expression::ControlFlow(_, control_flow)) => {
            check_control_flow(annotation, expected, control_flow, ctx)
        }
        (expected, Expression::DeconstructInto(_, deconstruct_into)) => todo!(),
        _otherwise => todo!(),
    }
}

fn check_binding<A>(
    annotation: &A,
    expected_type: &Type,
    binding: Binding<A>,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed + fmt::Display + Clone,
{
    let Binding {
        binder,
        bound,
        body,
    } = binding;
    let bound = ctx.infer_type(&bound)?;

    let mut ctx = ctx.clone();
    ctx.bind(binder.into(), bound.inferred_type.generalize(&ctx));

    let unification = check(annotation, *body, expected_type, &ctx)?;

    Ok(unification.compose(bound.substitutions))
}

fn check_lambda<A>(
    annotation: &A,
    expected_type: &Type,
    parameter_name: Identifier,
    parameter_type: Type,
    body: Box<Expression<A>>,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed + fmt::Display + Clone,
{
    let mut ctx = ctx.clone();
    ctx.bind(
        parameter_name.into(),
        TypeScheme::from_constant(parameter_type.clone()),
    );
    let body = ctx.infer_type(&body)?;

    let unification = expected_type.unify(
        &Type::Arrow(parameter_type.into(), body.inferred_type.into()),
        annotation,
    )?;

    Ok(body.substitutions.compose(unification))
}

fn check_apply<A>(
    annotation: &A,
    expected: &Type,
    function: Box<Expression<A>>,
    argument: Box<Expression<A>>,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed + fmt::Display + Clone,
{
    let argument = ctx.infer_type(&argument)?;
    let unification = check(
        annotation,
        *function,
        &Type::Arrow(argument.inferred_type.into(), expected.clone().into()),
        ctx,
    )?;
    Ok(unification.compose(argument.substitutions))
}

fn check_control_flow<A>(
    annotation: &A,
    expected: &Type,
    control_flow: ControlFlow<A>,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed + fmt::Display + Clone,
{
    match control_flow {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            let s1 = check(annotation, *predicate, &Type::Constant(BaseType::Bool), ctx)?;
            let s2 = check(annotation, *consequent, expected, ctx)?;
            // Should I apply substitutions?
            let s3 = check(annotation, *alternate, expected, ctx)?;
            Ok(s1.compose(s2.compose(s3)))
        }
    }
}
