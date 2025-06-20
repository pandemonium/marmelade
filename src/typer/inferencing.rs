use std::{collections::HashMap, mem};

use crate::{
    ast::{
        self, Constructor, ConstructorPattern, DomainExpression, Identifier, MatchClause, Pattern,
        PatternMatrix, StructPattern, TuplePattern, Variable,
    },
    parser::ParsingInfo,
    typer::{
        unification::Substitutions, BaseType, Parsed, ProductType, Type, TypeChecker, TypeError,
        TypeInference, TypeScheme, Typing, TypingContext,
    },
};

use super::{TupleType, UntypedExpression};

impl ast::Constant {
    pub fn synthesize_type(&self) -> Typing {
        Ok(TypeInference {
            substitutions: Substitutions::default(),
            inferred_type: Type::Constant(match self {
                Self::Int(..) => BaseType::Int,
                Self::Float(..) => BaseType::Float,
                Self::Text(..) => BaseType::Text,
                Self::Bool(..) => BaseType::Bool,
                Self::Unit => BaseType::Unit,
            }),
        })
    }
}

impl TypingContext {
    pub fn infer_type(&self, expression: &UntypedExpression) -> Typing {
        match expression {
            UntypedExpression::TypeAscription(
                _,
                ast::TypeAscription {
                    type_signature,
                    underlying,
                },
            ) => self.infer_ascription(type_signature, underlying),
            UntypedExpression::Variable(_, binding)
            | UntypedExpression::InvokeBridge(_, binding) => {
                if let Variable::Identifier(binding) = binding {
                    if let Some(scheme) = self.lookup(&binding.clone().into()) {
                        let inferred = scheme.instantiate(self)?;
                        Ok(TypeInference::trivially(inferred))
                    } else {
                        Err(TypeError::UndefinedSymbol(binding.clone()).into())
                    }
                } else {
                    panic!("Variables must not be a De Bruijn index at this point. {binding}")
                }
            }
            UntypedExpression::Literal(_, constant) => constant.synthesize_type(),
            UntypedExpression::Interpolation(..) => {
                Ok(TypeInference::trivially(Type::Constant(BaseType::Text)))
            }
            UntypedExpression::SelfReferential(
                parsing_info,
                ast::SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                let mut ctx = self.clone();
                ctx.bind(
                    name.clone().into(),
                    TypeScheme::from_constant(Type::Arrow(
                        Type::fresh().into(),
                        Type::fresh().into(),
                    )),
                );
                ctx.infer_lambda(parameter, body, *parsing_info)
            }
            UntypedExpression::Lambda(parsing_info, ast::Lambda { parameter, body }) => {
                self.infer_lambda(parameter, body, *parsing_info)
            }
            UntypedExpression::Apply(parsing_info, ast::Apply { function, argument }) => {
                self.infer_application(function, argument, *parsing_info)
            }
            UntypedExpression::Binding(
                _,
                ast::Binding {
                    binder,
                    bound,
                    body,
                    ..
                },
            ) => self.infer_binding(binder, bound, body),
            UntypedExpression::Inject(
                parsing_info,
                ast::Inject {
                    name,
                    constructor,
                    argument,
                },
            ) => self.infer_coproduct(name, constructor, argument, *parsing_info),
            UntypedExpression::Product(_, product) => self.infer_product(product),
            UntypedExpression::Project(_, ast::Project { base, index }) => {
                self.infer_projection(base, index)
            }
            UntypedExpression::Sequence(_, ast::Sequence { this, and_then }) => {
                self.infer_type(this)?;
                self.infer_type(and_then)
            }
            UntypedExpression::ControlFlow(parsing_info, control) => {
                self.infer_control_flow(control, *parsing_info)
            }
            UntypedExpression::DeconstructInto(
                parsing_info,
                ast::DeconstructInto {
                    scrutinee,
                    match_clauses,
                },
            ) => self.infer_deconstruct_into(*parsing_info, scrutinee, match_clauses),
        }
    }

    fn infer_ascription(
        &self,
        ascribed_type: &ast::TypeSignature<ParsingInfo>,
        underlying: &ast::Expression<ParsingInfo>,
    ) -> Typing {
        let ascribed_type = ascribed_type.synthesize_type(self)?;
        let checker = TypeChecker::new(self.clone());

        // I have a Type Scheme - why doesn't the checker deal with that?
        // ... should I instantiate it?
        let ascribed_type = ascribed_type.instantiate(self)?;
        checker.check(&ascribed_type, underlying)?;

        Ok(TypeInference::trivially(ascribed_type))
    }

    fn infer_lambda(
        &self,
        ast::Parameter {
            name,
            type_annotation,
        }: &ast::Parameter,
        body: &UntypedExpression,
        _info: ParsingInfo,
    ) -> Typing {
        let domain = type_annotation
            .as_ref()
            .map_or_else(Type::fresh, |domain| domain.clone());

        let domain = TypeScheme::from_constant(domain);

        let mut ctx = self.clone();
        ctx.bind(name.clone().into(), domain.clone());

        let codomain = ctx.infer_type(body)?;
        let function_type = Type::Arrow(
            domain
                .instantiate(&ctx)?
                .apply(&codomain.substitutions)
                .into(),
            // whatever body is should have applied those substitutions
            codomain.inferred_type.apply(&codomain.substitutions).into(),
        );

        Ok(TypeInference::new(codomain.substitutions, function_type))
    }

    fn infer_deconstruct_into(
        &self,
        parsing_info: ParsingInfo,
        scrutinee: &UntypedExpression,
        match_clauses: &[MatchClause<ParsingInfo>],
    ) -> Typing {
        let scrutinee_type = self.infer_type(scrutinee)?;
        let mut substitutions = scrutinee_type.substitutions.clone();
        let ctx = self.apply_substitutions(&scrutinee_type.substitutions);

        let consequents = ctx.infer_match_clauses(
            parsing_info,
            match_clauses,
            &scrutinee_type,
            &mut substitutions,
        )?;

        let consequent = ctx.unify_consequents(parsing_info, &consequents)?;
        let substitutions = substitutions.compose(consequent.substitutions);
        let inferred_type = consequent.inferred_type.apply(&substitutions);

        let mut matrix = PatternMatrix::from_scrutinee(
            scrutinee_type
                .inferred_type
                .apply(&substitutions)
                .expand_type(&ctx)?,
            &ctx,
        )?;

        // I think I need to unify scrutinee_type with the pattern type.
        for clause in match_clauses {
            let pattern = DomainExpression::from_pattern(&clause.pattern, &ctx)?;
            if matrix.is_useful(&pattern) {
                matrix.integrate(pattern);
            } else {
                eprintln!(
                    // It is because Expr here is an Apply tree to the TupleSyntax
                    // operator. HTF do I print this?
                    "{}: case {} is not useful for {scrutinee}.",
                    clause.pattern.annotation().location(),
                    clause.pattern
                );
            }
        }

        let residual = matrix.residual(&ctx)?;
        if residual.is_nothing() {
            Ok(TypeInference {
                substitutions,
                inferred_type,
            })
        } else {
            Err(TypeError::IncompleteDeconstruction {
                at: *parsing_info.info().location(),
                scrutinee: scrutinee.clone(),
                residual,
            }
            .into())
        }
    }

    fn infer_match_clauses(
        &self,
        parsing_info: ParsingInfo,
        match_clauses: &[MatchClause<ParsingInfo>],
        scrutinee: &TypeInference,
        substitutions: &mut Substitutions,
    ) -> Typing<Vec<TypeInference>> {
        let mut ctx = self.clone();
        let mut consequents = vec![];

        for clause in match_clauses {
            let pattern = clause.pattern.synthesize_type(&ctx)?;

            *substitutions = substitutions.compose(scrutinee.inferred_type.unify(
                &pattern.inferred_type.apply(&scrutinee.substitutions),
                parsing_info,
            )?);

            let pattern = clause
                .pattern
                .deconstruct(&scrutinee.inferred_type.clone().apply(substitutions), &ctx)?;

            *substitutions = substitutions.compose(pattern.substitutions);

            for (binding, scrutinee) in pattern.bindings {
                ctx.bind(
                    binding.into(),
                    TypeScheme::from_constant(scrutinee.apply(substitutions)),
                );
            }

            consequents.push(ctx.infer_type(&clause.consequent)?);
        }

        Ok(consequents)
    }

    fn unify_consequents(
        &self,
        parsing_info: ParsingInfo,
        consequents: &[TypeInference],
    ) -> Typing {
        let mut substitutions = consequents[0].substitutions.clone();
        let mut inferred_type = consequents[0].inferred_type.clone().expand_type(self)?;

        for consequent in &consequents[1..] {
            let rhs = consequent
                .inferred_type
                .clone()
                .apply(&substitutions)
                .expand_type(self)?;

            substitutions = substitutions.compose(inferred_type.unify(&rhs, parsing_info)?);
            inferred_type = inferred_type.apply(&substitutions);
        }

        Ok(TypeInference {
            substitutions,
            inferred_type,
        })
    }

    fn infer_projection(
        &self,
        base: &ast::Expression<ParsingInfo>,
        index: &ast::ProductIndex,
    ) -> Typing {
        let base = self.infer_type(base)?;
        match (&base.inferred_type, index) {
            // elements is a right-biased 2-tree so #3 is .1.1
            (Type::Product(ProductType::Tuple(tuple)), ix @ ast::ProductIndex::Tuple(index)) => {
                // This match clause is not efficient at all
                let TupleType(elements) = tuple.clone().unspine();

                if *index < elements.len() {
                    Ok(TypeInference::new(
                        base.substitutions,
                        elements[*index].clone(),
                    ))
                } else {
                    Err(TypeError::BadProjection {
                        base: base.inferred_type,
                        index: ix.clone(),
                    }
                    .into())
                }
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
                    }
                    .into())
                }
            }
            _otherwise => Err(TypeError::BadProjection {
                base: base.inferred_type,
                index: index.clone(),
            }
            .into()),
        }
    }

    fn infer_product(&self, product: &ast::Product<ParsingInfo>) -> Typing {
        match product {
            ast::Product::Tuple(elements) => self.infer_tuple(elements),
            ast::Product::Struct(bindings) => self.infer_struct(bindings),
        }
    }

    fn infer_tuple(&self, elements: &[ast::Expression<ParsingInfo>]) -> Typing {
        let mut substitutions = Substitutions::default();
        let mut types = Vec::with_capacity(elements.len());

        for element in elements.iter().rev() {
            let element = self.infer_type(element)?;

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
            Type::Product(ProductType::Tuple(TupleType(types))),
        ))
    }

    fn infer_struct(&self, elements: &[(ast::Identifier, UntypedExpression)]) -> Typing {
        let mut substitutions = Substitutions::default();
        let mut types = Vec::with_capacity(elements.len());

        for (label, initializer) in elements {
            let initializer = self.infer_type(initializer)?;

            substitutions = substitutions.compose(initializer.substitutions);
            types.push((label.clone(), initializer.inferred_type));
        }

        let mut types = types
            .into_iter()
            .map(|(field, t)| (field, t.apply(&substitutions)))
            .collect::<Vec<_>>();

        Ok(TypeInference::new(
            substitutions,
            Type::Product(ProductType::Struct(mem::take(&mut types))),
        ))
    }

    fn infer_coproduct(
        &self,
        name: &ast::TypeName,
        constructor: &ast::Identifier,
        argument: &UntypedExpression,
        annotation: ParsingInfo,
    ) -> Typing {
        let type_constructor = self
            .lookup(&name.clone().into())
            .ok_or_else(|| TypeError::UndefinedType(name.clone()))?
            .instantiate(self)?;

        if let Type::Coproduct(ref coproduct) = type_constructor.clone().expand_type(self)? {
            let argument = self.infer_type(argument)?;

            if let Some(lhs) = coproduct.constructor_signature(constructor) {
                let rhs = &argument.inferred_type;
                let substitutions = argument.substitutions.compose(lhs.unify(rhs, annotation)?);
                let inferred_type = type_constructor.apply(&substitutions);

                Ok(TypeInference::new(substitutions, inferred_type))
            } else {
                Err(TypeError::UndefinedCoproductConstructor {
                    coproduct: name.to_owned(),
                    constructor: constructor.to_owned(),
                }
                .into())
            }
        } else {
            Err(TypeError::UndefinedType(name.clone()).into())
        }
    }

    fn with_binding(&self, binding: super::Binding, scheme: TypeScheme) -> Self {
        let mut ctx = self.clone();
        ctx.bind(binding, scheme);
        ctx
    }

    fn infer_binding(
        &self,
        binding: &ast::Identifier,
        bound: &UntypedExpression,
        body: &UntypedExpression,
    ) -> Typing {
        let bound = self.infer_type(bound)?;
        let bound_type = bound.inferred_type.apply(&bound.substitutions);
        let bound_type = TypeScheme::from_constant(bound_type);

        let TypeInference {
            substitutions,
            inferred_type,
        } = self
            .with_binding(binding.clone().into(), bound_type.clone())
            .infer_type(body)?;

        println!("infer_binding: bound {binding} to {bound_type} and inferred {inferred_type}");

        let substitutions = bound.substitutions.compose(substitutions);
        let inferred_type = inferred_type.apply(&substitutions);
        Ok(TypeInference::new(substitutions, inferred_type))
    }

    fn with_applied_substitutions(&self, substitutions: &Substitutions) -> Self {
        self.apply_substitutions(substitutions)
    }

    fn infer_application(
        &self,
        function: &UntypedExpression,
        argument: &UntypedExpression,
        parsing_info: ParsingInfo,
    ) -> Typing {
        let function = self.infer_type(function)?;
        let argument = self
            .with_applied_substitutions(&function.substitutions)
            .infer_type(argument)?;

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
                parsing_info,
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

    fn infer_control_flow(
        &self,
        control: &ast::ControlFlow<ParsingInfo>,
        parsing_info: ParsingInfo,
    ) -> Typing {
        match control {
            ast::ControlFlow::If {
                predicate,
                consequent,
                alternate,
            } => self.infer_if_expression(predicate, consequent, alternate, parsing_info),
        }
    }

    fn infer_if_expression(
        &self,
        predicate: &UntypedExpression,
        consequent: &UntypedExpression,
        alternate: &UntypedExpression,
        annotation: ParsingInfo,
    ) -> Typing {
        let predicate_type = self.infer_type(predicate)?;
        let predicate = predicate_type
            .inferred_type
            .unify(&Type::Constant(BaseType::Bool), annotation)?;

        let engine = self.with_applied_substitutions(&predicate_type.substitutions);
        let consequent = engine.infer_type(consequent)?;
        let alternate = engine.infer_type(alternate)?;

        let branch = consequent
            .inferred_type
            .unify(&alternate.inferred_type, annotation)?;

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
}

#[derive(Debug, Default)]
struct Deconstructed {
    bindings: Vec<(Identifier, Type)>,
    substitutions: Substitutions,
}

impl Deconstructed {
    fn merge_with(&mut self, rhs: Self) {
        self.bindings.extend(rhs.bindings);
        self.substitutions = self.substitutions.compose(rhs.substitutions);
    }

    fn add_binding(&mut self, binding: Identifier, ty: Type) {
        self.bindings.push((binding, ty));
    }

    fn add_substitutions(&mut self, substitutions: Substitutions) {
        self.substitutions = self.substitutions.compose(substitutions);
    }
}

// For some Pattern types, a type can actually be inferred
// independently of the scrutinee. 1, for instance. Or Some.
// I
impl Pattern<ParsingInfo> {
    pub fn synthesize_type(&self, ctx: &TypingContext) -> Typing {
        match self {
            Self::Coproduct(_, pattern) => {
                let coproduct_type =
                    Constructor::<ParsingInfo>::constructed_type(&pattern.constructor, ctx)
                        .ok_or_else(|| TypeError::UndefinedSymbol(pattern.constructor.clone()))?
                        .instantiate(ctx)?;

                Ok(TypeInference::trivially(coproduct_type))
            }

            Self::Tuple(_, TuplePattern { elements }) => {
                // Can these two be combined?
                let elements = elements
                    .iter()
                    .map(|pattern| pattern.synthesize_type(ctx))
                    .collect::<Typing<Vec<_>>>()?;

                let length = elements.len();
                let (substitutions, element_types) = elements.into_iter().fold(
                    (Substitutions::default(), Vec::with_capacity(length)),
                    |(subs, mut element_types),
                     TypeInference {
                         substitutions,
                         inferred_type,
                     }| {
                        element_types.push(inferred_type);
                        (subs.compose(substitutions), element_types)
                    },
                );

                Ok(TypeInference {
                    substitutions,
                    inferred_type: Type::Product(ProductType::Tuple(TupleType(element_types))),
                })
            }

            Self::Struct(_, StructPattern { fields }) => {
                let fields = fields
                    .iter()
                    .map(|(field, pattern)| {
                        pattern
                            .synthesize_type(ctx)
                            .map(|pattern_type| (field, pattern_type))
                    })
                    .collect::<Typing<Vec<_>>>()?;

                let length = fields.len();
                let (substitutions, types) = fields.into_iter().fold(
                    (Substitutions::default(), Vec::with_capacity(length)),
                    |(substitutions, mut fields), (identifier, inference)| {
                        fields.push((identifier.clone(), inference.inferred_type));
                        (substitutions.compose(inference.substitutions), fields)
                    },
                );

                Ok(TypeInference {
                    substitutions,
                    inferred_type: Type::Product(ProductType::Struct(types)),
                })
            }

            Self::Literally(_, pattern) => pattern.synthesize_type(),

            Self::Otherwise(..) => Ok(TypeInference::fresh()),
        }
    }

    fn deconstruct(&self, scrutinee: &Type, ctx: &TypingContext) -> Typing<Deconstructed> {
        let scrutinee = scrutinee.clone().expand_type(ctx)?;
        match (self, &scrutinee) {
            (
                Self::Coproduct(
                    parsing_info,
                    ConstructorPattern {
                        constructor,
                        argument,
                    },
                ),
                Type::Coproduct(coproduct),
            ) => coproduct.constructor_signature(constructor).map_or_else(
                || {
                    Err(TypeError::PatternMatchImpossible {
                        pattern: self.clone().map(|annotation| *annotation.info()),
                        scrutinee: scrutinee.clone(),
                    }
                    .into())
                },
                |constructor| {
                    Self::Tuple(*parsing_info, argument.clone()).deconstruct(constructor, ctx)
                },
            ),

            (
                pattern @ Self::Tuple(_, TuplePattern { elements }),
                Type::Product(ProductType::Tuple(tuple)),
            ) => {
                let TupleType(tuple) = tuple.clone().unspine();

                if tuple.len() == elements.len() {
                    let mut deconstructed = Deconstructed::default();
                    for (pattern, scrutinee) in elements.iter().zip(tuple.iter()) {
                        deconstructed.merge_with(pattern.deconstruct(scrutinee, ctx)?);
                    }
                    Ok(deconstructed)
                } else {
                    Err(TypeError::PatternMatchImpossible {
                        pattern: pattern.clone(),
                        scrutinee,
                    }
                    .into())
                }
            }

            (
                Self::Struct(parsing_info, StructPattern { fields }),
                Type::Product(ProductType::Struct(struct_type)),
            ) if fields.len() == struct_type.len() => {
                let mut deconstructed = Deconstructed::default();
                let rhs = struct_type.iter().cloned().collect::<HashMap<_, _>>();
                for (field, pattern) in fields {
                    let scrutinee = rhs.get(field).ok_or_else(|| TypeError::UndefinedField {
                        position: *parsing_info.location(),
                        label: field.clone(),
                    })?;
                    deconstructed.merge_with(pattern.deconstruct(scrutinee, ctx)?);
                }
                Ok(deconstructed)
            }

            (Self::Literally(parsing_info, pattern), scrutinee) => {
                let pattern = pattern.synthesize_type()?;
                let substutitions = scrutinee.unify(&pattern.inferred_type, *parsing_info)?;
                let mut deconstructed = Deconstructed::default();
                deconstructed.add_substitutions(substutitions.compose(pattern.substitutions));

                Ok(deconstructed)
            }

            (Self::Otherwise(_parsing_info, pattern), _scrutinee) => {
                let mut deconstructed = Deconstructed::default();
                // This has the expanded type here.
                deconstructed.add_binding(pattern.clone(), scrutinee.clone());
                Ok(deconstructed)
            }

            // This ought to  be all bindings in the pattern, but with
            // fresh types
            (pattern, scrutinee) => Err(TypeError::PatternMatchImpossible {
                pattern: pattern.clone(),
                scrutinee: scrutinee.clone(),
            }
            .into()),
        }
    }
}
