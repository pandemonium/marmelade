use std::{fmt, marker::PhantomData};

use crate::{
    ast::{
        self, Constructor, ConstructorPattern, Expression, Identifier, MatchClause, Pattern,
        TuplePattern,
    },
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
        ast::Expression::DeconstructInto(
            annotation,
            ast::DeconstructInto {
                scrutinee,
                match_clauses,
            },
        ) => infer_deconstruct_into(annotation, scrutinee, match_clauses, ctx),
    }
}

fn infer_deconstruct_into<A>(
    annotation: &A,
    scrutinee: &Expression<A>,
    match_clauses: &[MatchClause<A>],
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    let mut ctx = ctx.clone();
    let scrutinee = infer_type(scrutinee, &ctx)?;
    let mut consequents = vec![];

    for MatchClause {
        pattern,
        consequent,
    } in match_clauses
    {
        let pattern_type = pattern.synthesize_type(&ctx)?;

        println!(
            "infer_deconstruct_into: pattern {}, scrutinee {}",
            pattern_type.inferred_type, scrutinee.inferred_type
        );

        let unification = pattern_type
            .inferred_type
            .unify(&scrutinee.inferred_type, annotation)?;

        // Yeah?
        // ctx = ctx.apply_substitutions(&unification);

        let bindings = pattern.check_type(
            annotation,
            &scrutinee.inferred_type.clone().apply(&unification),
            &ctx,
        )?;

        for (binding, scrutinee) in bindings {
            ctx.bind(binding.into(), scrutinee.generalize(&ctx));
        }

        consequents.push(infer_type(consequent, &ctx)?);
    }

    let TypeInference {
        mut substitutions,
        mut inferred_type,
    } = consequents.remove(0);

    for consequent in consequents {
        let consequent_ty = consequent.inferred_type.apply(&substitutions);
        substitutions =
            substitutions.compose(inferred_type.unify(&consequent_ty, annotation)?.clone());
        inferred_type = inferred_type.apply(&substitutions);
    }

    Ok(TypeInference {
        substitutions,
        inferred_type,
    })
}

#[derive(Debug, Default)]
struct Match {
    bindings: Vec<(Identifier, Type)>,
}

impl Match {
    fn add(&mut self, id: Identifier, ty: Type) {
        self.bindings.push((id, ty));
    }
}

// For some Pattern types, a type can actually be inferred
// independently of the scrutinee. 1, for instance. Or Some.
// I
impl<A> Pattern<A>
where
    A: fmt::Display + Clone + Parsed,
{
    fn synthesize_type(&self, ctx: &TypingContext) -> Typing {
        match self {
            Self::Coproduct(pattern, _) => {
                let coproduct_type = Constructor::<A>::constructed_type(&pattern.constructor, ctx)
                    .ok_or_else(|| TypeError::UndefinedSymbol(pattern.constructor.clone()))?
                    .instantiate(ctx)?
                    .expand_type(ctx)?;

                println!("synthesize_type: {}", coproduct_type);

                Ok(TypeInference::trivially(coproduct_type))
            }
            Self::Tuple(TuplePattern { elements }, _) => {
                let elements = elements
                    .iter()
                    .map(|pattern| pattern.synthesize_type(ctx))
                    .collect::<Typing<Vec<_>>>()?;

                let (substitutions, element_types) = elements.into_iter().fold(
                    (Substitutions::default(), vec![]),
                    |(subs, mut el_tys),
                     TypeInference {
                         substitutions,
                         inferred_type,
                     }| {
                        el_tys.push(inferred_type);
                        (subs.compose(substitutions), el_tys)
                    },
                );

                Ok(TypeInference {
                    substitutions,
                    inferred_type: Type::Product(ProductType::Tuple(element_types)),
                })
            }
            Self::Literally(pattern) => pattern.synthesize_type(),
            Self::Otherwise(_) => Ok(TypeInference::fresh()),
        }
    }

    fn check_type(
        &self,
        annotation: &A,
        scrutinee: &Type,
        ctx: &TypingContext,
    ) -> Typing<Vec<(Identifier, Type)>> {
        println!(
            "check_type : {} {}",
            matches!(self, Pattern::Tuple(..)),
            scrutinee
        );

        match (self, scrutinee) {
            (
                Self::Coproduct(
                    ConstructorPattern {
                        constructor,
                        argument,
                    },
                    _,
                ),
                Type::Coproduct(coproduct),
            ) => {
                if let Some(constructor) = coproduct.constructor_signature(constructor) {
                    Self::Tuple(argument.clone(), PhantomData::default()).check_type(
                        annotation,
                        constructor,
                        ctx,
                    )
                } else {
                    Err(TypeError::PatternMatchImpossible {
                        pattern: self.clone().map(|annotation| annotation.info().clone()),
                        scrutinee: scrutinee.clone(),
                    })
                }
            }

            (
                // This does not hit because Cons takes two arguments,
                // these are in the first element of the argument tuple. As a tuple.
                // Wtf, patrik.
                Self::Tuple(TuplePattern { elements }, _),
                Type::Product(ProductType::Tuple(tuple)),
            ) if elements.len() == tuple.len() => {
                let mut matched = vec![];
                for (pattern, scrutinee) in elements.iter().zip(tuple.iter()) {
                    matched.extend(pattern.check_type(annotation, scrutinee, ctx)?);
                }
                Ok(matched)
            }

            (Self::Literally(pattern), scrutinee) => {
                let pattern_type = pattern.synthesize_type()?.inferred_type;
                if scrutinee == &pattern_type {
                    Ok(vec![])
                } else {
                    Err(TypeError::UnifyImpossible {
                        lhs: pattern_type,
                        rhs: scrutinee.clone(),
                        position: annotation.info().location().clone(),
                    })
                }
            }

            (Self::Otherwise(pattern), scrutinee) => Ok(vec![(pattern.clone(), scrutinee.clone())]),

            // This ought to  be all bindings in the pattern, but with
            // fresh types
            (pattern, scrutinee) => Err(TypeError::PatternMatchImpossible {
                pattern: pattern.clone().map(|annotation| annotation.info().clone()),
                scrutinee: scrutinee.clone(),
            }),
        }
    }
}

// revisit this function
fn infer_lambda<A>(
    ast::Parameter { name, .. }: &ast::Parameter<A>,
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

        if let Some(lhs) = coproduct.constructor_signature(constructor) {
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
