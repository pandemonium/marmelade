use std::{collections::HashMap, fmt};

use crate::{
    ast,
    parser::ParsingInfo,
    typer::{
        unification::Substitutions, BaseType, Parsed, ProductType, Type, TypeError, TypeInference,
        TypeScheme, Typing, TypingContext,
    },
};

impl ast::Constant {
    fn synthesize_type(&self) -> Typing {
        match self {
            ast::Constant::Int(..) => synthesize_constant(BaseType::Int),
            ast::Constant::Float(..) => synthesize_constant(BaseType::Float),
            ast::Constant::Text(..) => synthesize_constant(BaseType::Text),
            ast::Constant::Bool(..) => synthesize_constant(BaseType::Bool),
            ast::Constant::Unit => synthesize_constant(BaseType::Unit),
        }
    }
}

fn synthesize_constant(ty: BaseType) -> Typing {
    Ok(TypeInference {
        substitutions: Substitutions::default(),
        inferred_type: Type::Constant(ty),
    })
}

pub fn infer_type<A>(expr: &ast::Expression<A>, ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    match expr {
        ast::Expression::Variable(_, binding) | ast::Expression::InvokeBridge(_, binding) => {
            if let Some(scheme) = ctx.lookup(&binding.clone().into()) {
                let inferred = scheme.instantiate(ctx)?;
                Ok(TypeInference::trivially(inferred))
            } else {
                Err(TypeError::UndefinedSymbol(binding.clone()))
            }
        }
        ast::Expression::Literal(_, constant) => constant.synthesize_type(),
        ast::Expression::SelfReferential(
            annotation,
            ast::SelfReferential {
                name,
                parameter,
                body,
            },
        ) => {
            let mut ctx = ctx.clone();
            ctx.bind(
                name.clone().into(),
                TypeScheme {
                    quantifiers: vec![],
                    body: Type::fresh(),
                },
            );
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
        ast::Expression::DeconstructInto(_annotation, deconstruct) => {
            println!("infer_type: ----> TEMPORARY infer_type FOR deconstruct_into. ALWAYS Int");
            //infer_type(
            //    &deconstruct.match_clauses.iter().next().unwrap().consequent,
            //    ctx, // THIS IS TEMPORARY, etc
            //)
            Ok(TypeInference {
                substitutions: Substitutions::default(),
                inferred_type: Type::Constant(BaseType::Int),
            })
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
    let expected_param_type = Type::fresh();

    let mut ctx = ctx.clone();
    ctx.bind(
        name.clone().into(),
        TypeScheme {
            quantifiers: vec![],
            body: expected_param_type.clone(),
        },
    );

    let body = infer_type(body, &ctx)?;

    let inferred_param_type = expected_param_type.clone().apply(&body.substitutions);

    let annotation_unification = expected_param_type.unify(&inferred_param_type, info)?;

    let function_type = Type::Arrow(
        inferred_param_type.apply(&annotation_unification).into(),
        body.inferred_type.into(),
    );

    Ok(TypeInference::new(
        body.substitutions.compose(annotation_unification),
        function_type,
    ))
}

fn infer_struct<A>(
    elements: &[(ast::Identifier, ast::Expression<A>)],
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

    Ok(TypeInference::new(
        substitutions,
        Type::Product(ProductType::Struct(types.drain(..).collect())),
    ))
}

fn infer_tuple<A>(elements: &[ast::Expression<A>], ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let mut substitutions = Substitutions::default();
    let mut types = Vec::with_capacity(elements.len());

    for element in elements.iter().rev() {
        let element = ctx.infer_type(element)?;

        substitutions = substitutions.compose(element.substitutions);
        types.push(element.inferred_type);
    }

    let mut types = types
        .drain(..)
        .map(|t| t.apply(&substitutions))
        .collect::<Vec<_>>();
    types.reverse();

    // todo: don't I have to substitute my element types?
    Ok(TypeInference::new(
        substitutions,
        Type::Product(ProductType::Tuple(types)),
    ))
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
    let type_constructor = ctx
        .lookup(&name.clone().into())
        .ok_or_else(|| TypeError::UndefinedType(name.clone()))?
        .instantiate(ctx)?;

    if let Type::Coproduct(ref coproduct) = type_constructor.clone().expand_type(ctx)? {
        let argument = infer_type(argument, ctx)?;

        if let Some(lhs) = coproduct.find_constructor(constructor) {
            let rhs = &argument.inferred_type;
            let substitutions = argument.substitutions.compose(lhs.unify(rhs, annotation)?);
            let inferred_type = type_constructor.apply(&substitutions);

            Ok(TypeInference::new(substitutions, inferred_type))
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

fn infer_product<A>(product: &ast::Product<A>, ctx: &TypingContext) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    match product {
        ast::Product::Tuple(elements) => infer_tuple(elements, ctx),
        ast::Product::Struct(bindings) => infer_struct(bindings, ctx),
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
            Ok(TypeInference::new(
                base.substitutions,
                elements[*index].clone(),
            ))
        }
        (Type::Product(ProductType::Struct(elements)), ast::ProductIndex::Struct(id)) => {
            if let Some((_, inferred_type)) = elements.iter().find(|(field, _)| field == id) {
                Ok(TypeInference::new(
                    base.substitutions,
                    inferred_type.clone(),
                ))
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
    let bound_type = bound.inferred_type.generalize(ctx);

    let mut ctx = ctx.clone();
    ctx.bind(binding.clone().into(), bound_type);

    let TypeInference {
        substitutions,
        inferred_type,
    } = infer_type(body, &ctx.apply_substitutions(&bound.substitutions))?;

    // Think about a map_substitutions function
    Ok(TypeInference::new(
        bound.substitutions.compose(substitutions),
        inferred_type,
    ))
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
        .unify(&Type::Constant(BaseType::Bool), annotation)
        .inspect_err(|e| println!("infer_if_expression: predicate unify error: {e}"))?;

    let ctx = ctx.apply_substitutions(&predicate_type.substitutions.clone());
    let consequent = infer_type(consequent, &ctx)?;
    let alternate = infer_type(alternate, &ctx)?;

    let branch = consequent
        .inferred_type
        .clone() //wtf
        .unify(&alternate.inferred_type, annotation)
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
    let function = infer_type(function, &ctx)?;
    let argument = infer_type(argument, &ctx.apply_substitutions(&function.substitutions))?;
    let return_type = Type::fresh();
    let unified_substitutions = function
        .inferred_type
        .apply(&function.substitutions)
        .unify(
            &Type::Arrow(
                argument.inferred_type.apply(&function.substitutions).into(),
                return_type.clone().into(),
            ),
            annotation,
        )?;

    let return_type = return_type.apply(&unified_substitutions);
    Ok(TypeInference {
        substitutions: function
            .substitutions
            .compose(argument.substitutions)
            .compose(unified_substitutions),
        inferred_type: return_type,
    })
}
