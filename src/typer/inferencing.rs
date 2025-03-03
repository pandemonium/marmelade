use std::{collections::HashMap, fmt};

use crate::{
    ast,
    parser::ParsingInfo,
    typer::{Binding, Type, TypeInference},
};

use super::{
    unification::Substitutions, BaseType, Parsed, ProductType, TypeError, Typing, TypingContext,
};

pub fn generalize_type(ty: Type, ctx: &TypingContext) -> Type {
    let context_variables = ctx.free_variables();
    let type_variables = ty.free_variables();
    let non_quantified = type_variables.difference(&context_variables);

    non_quantified.fold(ty, |body, ty_var| Type::Forall(*ty_var, Box::new(body)))
}

pub fn infer_type<A>(expr: &ast::Expression<A>, ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    match expr {
        ast::Expression::Variable(_, binding) | ast::Expression::CallBridge(_, binding) => {
            // Ref-variants of Binding too?
            if let Some(ty) = ctx.lookup(&binding.clone().into()) {
                Ok(TypeInference {
                    substitutions: Substitutions::default(),
                    inferred_type: ty.instantiate(),
                })
            } else {
                Err(TypeError::UndefinedSymbol(binding.clone()))
            }
        }
        ast::Expression::Literal(_, constant) => synthesize_type_of_constant(constant, ctx),
        ast::Expression::SelfReferential(
            annotation,
            ast::SelfReferential {
                name,
                parameter,
                body,
            },
        ) => {
            println!("synthesize_type: Self Referential {name}");
            let mut ctx = ctx.clone();
            ctx.bind(name.clone().into(), Type::fresh());
            infer_lambda(parameter, body, &annotation.info(), &ctx)
        }
        ast::Expression::Lambda(annotation, ast::Lambda { parameter, body }) => {
            infer_lambda(parameter, body, &annotation.info(), ctx)
        }
        ast::Expression::Apply(annotation, ast::Apply { function, argument }) => {
            infer_application(function, argument, annotation, ctx)
        }
        ast::Expression::Binding(
            _,
            ast::Binding {
                binder,
                bound,
                body,
                ..
            },
        ) => infer_binding(binder, bound, body, ctx),
        ast::Expression::Inject(
            annotation,
            ast::Inject {
                name,
                constructor,
                argument,
            },
        ) => infer_coproduct(name, constructor, argument, annotation, ctx),
        ast::Expression::Product(_, product) => infer_product(product, ctx),
        ast::Expression::Project(_, ast::Project { base, index }) => {
            infer_projection(base, index, ctx)
        }
        ast::Expression::Sequence(_, ast::Sequence { this, and_then }) => {
            infer_type(this, ctx)?;
            infer_type(and_then, ctx)
        }
        ast::Expression::ControlFlow(annotation, control) => {
            infer_control_flow(control, &annotation, ctx)
        }
    }
}

// revisit this function
fn infer_lambda<A>(
    ast::Parameter {
        name,
        type_annotation,
    }: &ast::Parameter<A>,
    body: &ast::Expression<A>,
    info: &ParsingInfo,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let expected_param_type = if let Some(param_type) = type_annotation {
        // Why doesn't it have to look something up?
        // It does.

        let has_unit = ctx
            .lookup(&Binding::TypeTerm("builtin::Unit".to_owned()))
            .is_some();
        println!("infer_lambda: has unit {}", has_unit);

        param_type
            .clone()
            .synthesize_type(&mut HashMap::default())
            .instantiate()
            .expand(ctx)?
    } else {
        Type::fresh()
    };

    let mut ctx = ctx.clone();
    ctx.bind(name.clone().into(), expected_param_type.clone());

    println!("infer_lambda: XXXX {body}");

    let body = infer_type(body, &ctx)?;

    println!("infer_lambda: YYYY {}", body.inferred_type);

    let inferred_param_type = expected_param_type.clone().apply(&body.substitutions);

    let annotation_unification = expected_param_type.unify(&inferred_param_type, info, &ctx)?;

    let function_type = generalize_type(
        Type::Arrow(inferred_param_type.into(), body.inferred_type.into()),
        &ctx,
    );

    println!("infer_lambda: function type {function_type}");

    Ok(TypeInference {
        substitutions: body.substitutions.compose(annotation_unification),
        inferred_type: function_type,
    })
}

fn infer_struct<A>(
    elements: &HashMap<ast::Identifier, ast::Expression<A>>,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let mut substitutions = Substitutions::default();
    let mut types = Vec::with_capacity(elements.len());

    for (label, initializer) in elements {
        let initializer = ctx.infer_type(initializer)?;

        substitutions = substitutions.compose(initializer.substitutions);
        types.push((label.clone(), initializer.inferred_type));
    }

    Ok(TypeInference {
        substitutions,
        inferred_type: Type::Product(ProductType::Struct(types.drain(..).collect())),
    })
}

fn infer_tuple<A>(elements: &[ast::Expression<A>], ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let mut substitutions = Substitutions::default();
    let mut types = Vec::with_capacity(elements.len());

    for element in elements {
        let element = ctx.infer_type(element)?;

        substitutions = substitutions.compose(element.substitutions);
        types.push(element.inferred_type);
    }

    Ok(TypeInference {
        substitutions,
        // todo: don't I have to substitute my element types?
        inferred_type: Type::Product(ProductType::Tuple(types)),
    })
}

fn synthesize_trivial(ty: BaseType) -> Typing {
    Ok(TypeInference {
        substitutions: Substitutions::default(),
        inferred_type: Type::Constant(ty),
    })
}

fn infer_coproduct<A>(
    name: &ast::TypeName,
    constructor: &ast::Identifier,
    argument: &ast::Expression<A>,
    annotation: &A,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    if let ref constructed_type @ Type::Coproduct(ref coproduct) = ctx
        .lookup(&name.clone().into())
        .ok_or_else(|| TypeError::UndefinedType(name.clone()))
        .map(|ty| ty.instantiate())?
    {
        let argument = infer_type(argument, ctx)?;
        if let Some(rhs) = coproduct.find_constructor(constructor) {
            let lhs = &argument.inferred_type;

            let substitutions = argument
                .substitutions
                .compose(lhs.unify(rhs, annotation, &ctx)?);
            let inferred_type = constructed_type.clone().apply(&substitutions);
            Ok(TypeInference {
                substitutions,
                inferred_type,
            })
        } else {
            Err(TypeError::UndefinedCoproductConstructor {
                coproduct: name.to_owned(),
                constructor: constructor.to_owned(),
            })
        }
    } else {
        Err(TypeError::UndefinedType(name.clone()))
    }
}

fn synthesize_type_of_constant(c: &ast::Constant, _ctx: &TypingContext) -> Typing {
    match c {
        ast::Constant::Int(..) => synthesize_trivial(BaseType::Int),
        ast::Constant::Float(..) => synthesize_trivial(BaseType::Float),
        ast::Constant::Text(..) => synthesize_trivial(BaseType::Text),
        ast::Constant::Bool(..) => synthesize_trivial(BaseType::Bool),
        ast::Constant::Unit => synthesize_trivial(BaseType::Unit),
    }
}

fn infer_product<A>(product: &ast::Product<A>, ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    match product {
        ast::Product::Tuple(elements) => infer_tuple(elements, ctx),
        ast::Product::Struct { bindings } => infer_struct(bindings, ctx),
    }
}

fn infer_projection<A>(
    base: &ast::Expression<A>,
    index: &ast::ProductIndex,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let base = infer_type(base, ctx)?;

    match (&base.inferred_type, index) {
        (Type::Product(ProductType::Tuple(elements)), ast::ProductIndex::Tuple(index))
            if *index < elements.len() =>
        {
            Ok(TypeInference {
                substitutions: base.substitutions,
                inferred_type: elements[*index].clone(),
            })
        }
        (Type::Product(ProductType::Struct(elements)), ast::ProductIndex::Struct(id)) => {
            if let Some((_, inferred_type)) = elements.iter().find(|(field, _)| field == id) {
                Ok(TypeInference {
                    substitutions: base.substitutions,
                    inferred_type: inferred_type.clone(),
                })
            } else {
                Err(TypeError::BadProjection {
                    base: base.inferred_type,
                    index: index.clone(),
                })
            }
        }
        _otherwise => Err(TypeError::BadProjection {
            base: base.inferred_type,
            index: index.clone(),
        }),
    }
}

fn infer_binding<A>(
    binding: &ast::Identifier,
    bound: &ast::Expression<A>,
    body: &ast::Expression<A>,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let bound = infer_type(bound, ctx)?;
    let bound_type = generalize_type(bound.inferred_type, ctx);

    let mut ctx = ctx.clone();
    ctx.bind(binding.clone().into(), bound_type);

    let TypeInference {
        substitutions,
        inferred_type,
    } = infer_type(body, &ctx.apply_substitutions(&bound.substitutions))?;

    Ok(TypeInference {
        substitutions: bound.substitutions.compose(substitutions),
        inferred_type,
    })
}

fn infer_if_expression<A>(
    predicate: &ast::Expression<A>,
    consequent: &ast::Expression<A>,
    alternate: &ast::Expression<A>,
    annotation: &A,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let predicate_type = infer_type(predicate, ctx)?;
    let predicate = predicate_type
        .inferred_type
        .unify(&Type::Constant(BaseType::Bool), annotation, &ctx)
        .inspect_err(|e| println!("infer_if_expression: predicate unify error: {e}"))?;

    let ctx = ctx.apply_substitutions(&predicate_type.substitutions.clone());
    let consequent = infer_type(consequent, &ctx)?;
    let alternate = infer_type(alternate, &ctx)?;

    let branch = consequent
        .inferred_type
        .clone() //wtf
        .unify(&alternate.inferred_type, annotation, &ctx)
        .inspect_err(|e| println!("infer_if_expression: branch unify error: {e}"))?;

    let substitutions = predicate
        .compose(predicate_type.substitutions)
        .compose(consequent.substitutions)
        .compose(alternate.substitutions)
        .compose(branch);

    let inferred_type = consequent.inferred_type.apply(&substitutions);

    Ok(TypeInference {
        substitutions,
        inferred_type,
    })
}

fn infer_control_flow<A>(
    control: &ast::ControlFlow<A>,
    annotation: &A,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    match control {
        ast::ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => infer_if_expression(predicate, consequent, alternate, annotation, ctx),
    }
}

fn infer_application<A>(
    function: &ast::Expression<A>,
    argument: &ast::Expression<A>,
    annotation: &A,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let function = infer_type(function, ctx)?;
    let argument = infer_type(argument, &ctx.apply_substitutions(&function.substitutions))?;

    let return_type = Type::fresh();

    let unified_substitutions = function
        .inferred_type
        .instantiate()
        .apply(&function.substitutions)
        .unify(
            &Type::Arrow(
                argument.inferred_type.apply(&function.substitutions).into(),
                return_type.clone().into(),
            ),
            annotation,
            &ctx,
        )?;

    println!("infer_application: {unified_substitutions:?}");

    let return_type = return_type.apply(&unified_substitutions);
    Ok(TypeInference {
        substitutions: function
            .substitutions
            .compose(argument.substitutions)
            .compose(unified_substitutions),
        inferred_type: return_type,
    })
}
