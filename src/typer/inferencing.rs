use std::fmt;

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
                    body: Type::Arrow(Type::fresh().into(), Type::fresh().into()),
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
    let scrutinee = infer_type(scrutinee, &ctx)?;
    let mut subs = scrutinee.substitutions.clone();

    println!("infer_deconstruct_into: scrutinee {scrutinee}");
    let mut ctx = ctx.apply_substitutions(&scrutinee.substitutions);

    let mut consequents = vec![];

    for MatchClause {
        pattern,
        consequent,
    } in match_clauses
    {
        let pattern_type = pattern.synthesize_type(&ctx)?;

        let substitutions = scrutinee
            .substitutions
            .compose(pattern_type.substitutions.clone());

        let mut unification = scrutinee.inferred_type.unify(
            &pattern_type
                .inferred_type
                .clone()
                .apply(&scrutinee.substitutions),
            annotation,
        )?;

        println!(
            "XXX {substitutions} / {unification} {}",
            pattern_type.substitutions
        );

        subs = subs.compose(unification.clone());

        //        pattern_type.inferred_type = pattern_type.inferred_type.apply(&unification);

        // TODO: move the annotation into the Pattern
        let Match {
            bindings,
            substitutions,
        } = pattern.deconstruct(
            annotation,
            &scrutinee.inferred_type.clone().apply(&unification),
            &ctx,
        )?;

        unification = unification.compose(substitutions);

        println!("infer_deconstruct_into: uni {unification}");

        for (binding, scrutinee) in bindings {
            //            let scheme = scrutinee.clone().apply(&unification).generalize(&ctx);
            let scheme = TypeScheme::from_constant(scrutinee.clone().apply(&unification));
            println!("infer_deconstruct_into: binding {binding} to {scheme} scrutinee {scrutinee}");
            ctx.bind(binding.into(), TypeScheme::from_constant(scrutinee));
        }

        let value = infer_type(consequent, &ctx)?;

        println!("XXX value {value}");

        consequents.push(value);
    }

    let TypeInference {
        mut substitutions,
        mut inferred_type,
    } = consequents.remove(0);

    for consequent in consequents {
        let consequent_ty = consequent.inferred_type.apply(&substitutions);

        let substitutions1 = inferred_type.unify(&consequent_ty, annotation)?.clone();

        substitutions = substitutions.compose(substitutions1);
        inferred_type = inferred_type.apply(&substitutions);
    }

    Ok(TypeInference {
        substitutions: subs.compose(substitutions.clone()),
        inferred_type: inferred_type.apply(&subs.compose(substitutions)),
    })
}

#[derive(Debug, Default)]
struct Match {
    bindings: Vec<(Identifier, Type)>,
    substitutions: Substitutions,
}

impl Match {
    fn merge_with(&mut self, rhs: Self) {
        self.bindings.extend(rhs.bindings);
        self.substitutions = self.substitutions.compose(rhs.substitutions);
    }

    fn add_binding(&mut self, binding: Identifier, ty: Type) {
        self.bindings.push((binding, ty));
    }

    fn add_substitutions(&mut self, substitutions: Substitutions) {
        self.substitutions = self.substitutions.compose(substitutions)
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
            Self::Coproduct(_, pattern) => {
                let coproduct_type = Constructor::<A>::constructed_type(&pattern.constructor, ctx)
                    .ok_or_else(|| TypeError::UndefinedSymbol(pattern.constructor.clone()))?
                    .instantiate(ctx)?;

                Ok(TypeInference::trivially(coproduct_type))
            }
            Self::Tuple(_, TuplePattern { elements }) => {
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

    fn deconstruct(
        &self,
        annotation: &A,
        scrutinee_in: &Type,
        ctx: &TypingContext,
    ) -> Typing<Match> {
        let scrutinee = scrutinee_in.clone().expand_type(ctx)?;
        match (self, &scrutinee) {
            (
                Self::Coproduct(
                    annotation,
                    ConstructorPattern {
                        constructor,
                        argument,
                    },
                ),
                Type::Coproduct(coproduct),
            ) => {
                if let Some(constructor) = coproduct.constructor_signature(constructor) {
                    Self::Tuple(annotation.clone(), argument.clone()).deconstruct(
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
                Self::Tuple(_, TuplePattern { elements }),
                Type::Product(ProductType::Tuple(tuple)),
            ) if elements.len() == tuple.len() => {
                let mut matched = Match::default();
                for (pattern, scrutinee) in elements.iter().zip(tuple.iter()) {
                    matched.merge_with(pattern.deconstruct(annotation, scrutinee, ctx)?)
                }
                Ok(matched)
            }

            (Self::Literally(pattern), scrutinee) => {
                let pattern = pattern.synthesize_type()?;
                let substutitions = scrutinee.unify(&pattern.inferred_type, annotation)?;
                let mut matched = Match::default();
                matched.add_substitutions(substutitions.compose(pattern.substitutions));

                Ok(matched)
            }

            (Self::Otherwise(pattern), _scrutinee) => {
                let mut matched = Match::default();
                // This has the expanded type here.
                matched.add_binding(pattern.clone(), scrutinee_in.clone());
                Ok(matched)
            }

            // This ought to  be all bindings in the pattern, but with
            // fresh types
            (pattern, scrutinee) => Err(TypeError::PatternMatchImpossible {
                pattern: pattern.clone().map(|annotation| annotation.info().clone()),
                scrutinee: scrutinee.clone(),
            }),
        }
    }
}

fn infer_lambda<A>(
    ast::Parameter {
        name,
        type_annotation,
    }: &ast::Parameter,
    body: &ast::Expression<A>,
    _info: &ParsingInfo,
    ctx: &TypingContext,
) -> Typing
where
    A: fmt::Display + Clone + Parsed,
{
    println!("infer_lambda: type annotation {:?}", type_annotation);
    let domain_type = if let Some(ty) = type_annotation {
        println!("infer_lambda: domain {ty}");
        ty.clone()
    } else {
        Type::fresh()
    };
    let domain = TypeScheme::from_constant(domain_type);

    let mut ctx = ctx.clone();
    ctx.bind(name.clone().into(), domain.clone());

    let codomain = infer_type(body, &ctx)?;
    println!("infer_lambda(1): {codomain}");

    let function_type = Type::Arrow(
        domain
            .instantiate(&ctx)?
            .apply(&codomain.substitutions)
            .into(),
        // whatever body is should have applied those substitutions
        codomain.inferred_type.apply(&codomain.substitutions).into(),
    );

    println!("infer_lambda(2): {function_type}");

    Ok(TypeInference::new(codomain.substitutions, function_type))
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
        .into_iter()
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
    let bound_type = bound
        .inferred_type
        .apply(&bound.substitutions)
        .generalize(ctx);

    let mut ctx = ctx.clone();
    println!("infer_binding: binding {binding} to {bound_type}");
    ctx.bind(binding.clone().into(), bound_type);

    let TypeInference {
        substitutions,
        inferred_type,
    } = infer_type(body, &ctx)?;

    // Think about a map_substitutions function
    Ok(TypeInference::new(
        bound.substitutions.compose(substitutions.clone()),
        inferred_type
            .clone()
            .apply(&bound.substitutions.compose(substitutions)),
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

    println!("infer_application: function {}", function);
    println!("infer_application: argument {}", argument);

    let return_type = Type::fresh();
    let unified_substitutions = function
        .inferred_type
        .apply(
            &function
                .substitutions
                .compose(argument.substitutions.clone()),
        )
        .unify(
            &Type::Arrow(
                argument
                    .inferred_type
                    .apply(
                        &function
                            .substitutions
                            .compose(argument.substitutions.clone()),
                    )
                    .into(),
                return_type.clone().into(),
            ),
            annotation,
        )?;

    let substitutions = function
        .substitutions
        .compose(argument.substitutions)
        .compose(unified_substitutions);
    let return_type = return_type.apply(&substitutions);
    Ok(TypeInference {
        substitutions,
        inferred_type: return_type,
    })
}
