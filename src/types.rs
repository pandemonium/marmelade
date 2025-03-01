use core::panic;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    slice::Iter,
};
use thiserror::Error;

use crate::{
    ast::{self, TypeName},
    lexer::SourcePosition,
    parser::ParsingInfo,
};

/*
    The structure of this file is off.
    The toplevel module ought to be typer.
    What does the types/ typer dichotomy afford us?
    I don't really like the way the associated functions
      are distributed over the types either.
*/
pub trait Parsed {
    fn info(&self) -> &ParsingInfo;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Parameter(TypeParameter),
    Constant(BaseType),
    Product(ProductType),
    Coproduct(CoproductType),
    Arrow(Box<Type>, Box<Type>),
    Forall(TypeParameter, Box<Type>),
    Named(TypeName),
    Apply(Box<Type>, Box<Type>),
}

impl Type {
    fn is_constant(&self) -> bool {
        matches!(self, Type::Constant(..))
    }

    fn resolve_constants(self, ctx: &TypingContext) -> Typing<Self> {
        self.resolve(
            &mut |ty| {
                if let Type::Named(name) = &ty {
                    ctx.lookup(&Binding::TypeTerm(name.as_str().to_owned()))
                        .filter(|t| t.is_constant())
                        .cloned()
                        .unwrap_or_else(|| ty)
                } else {
                    ty
                }
            },
            ctx,
        )
    }

    fn expand(self, ctx: &TypingContext) -> Typing<Self> {
        self.resolve(
            &mut |ty| {
                println!("expand: {ty}");
                if let Type::Named(name) = &ty {
                    ctx.lookup(&Binding::TypeTerm(name.as_str().to_owned()))
                        .cloned()
                        .unwrap_or_else(|| ty)
                } else {
                    ty
                }
            },
            ctx,
        )
    }

    fn resolve<F>(self, resolver: &mut F, ctx: &TypingContext) -> Typing<Self>
    where
        F: FnMut(Self) -> Self,
    {
        fn traverse_type<F>(
            ty: Type,
            resolver: &mut F,
            ctx: &TypingContext,
            seen: &mut HashSet<TypeName>,
        ) -> Typing<Type>
        where
            F: FnMut(Type) -> Type,
        {
            match ty {
                Type::Parameter(..) => Ok(ty),
                Type::Constant(..) => Ok(ty),
                Type::Product(product) => Ok(Type::Product(match product {
                    ProductType::Tuple(mut elements) => ProductType::Tuple(
                        elements
                            .drain(..)
                            .map(|t| traverse_type(t, resolver, ctx, seen))
                            .collect::<Typing<_>>()?,
                    ),
                    ProductType::Struct(mut fields) => ProductType::Struct(
                        fields
                            .drain(..)
                            .map(|(name, t)| {
                                traverse_type(t, resolver, ctx, seen).map(|t| (name, t))
                            })
                            .collect::<Typing<_>>()?,
                    ),
                })),
                Type::Coproduct(CoproductType(mut constructors)) => {
                    Ok(Type::Coproduct(CoproductType(
                        constructors
                            .drain(..)
                            .map(|(name, constructor)| {
                                traverse_type(constructor, resolver, ctx, seen).map(|t| (name, t))
                            })
                            .collect::<Typing<_>>()?,
                    )))
                }
                Type::Arrow(domain, codomain) => Ok(Type::Arrow(
                    traverse_type(*domain, resolver, ctx, seen)?.into(),
                    traverse_type(*codomain, resolver, ctx, seen)?.into(),
                )),
                Type::Forall(parameter, body) => Ok(Type::Forall(
                    parameter,
                    traverse_type(*body, resolver, ctx, seen)?.into(),
                )),
                Type::Named(ref name) => {
                    if seen.contains(&name) {
                        println!("traverse_type: seen {name}");
                        Ok(ty)
                    } else {
                        println!("traverse_type: have not seen {name}");
                        seen.insert(name.to_owned());

                        ctx.lookup(&Binding::TypeTerm(name.as_str().to_owned()))
                            .cloned()
                            .ok_or_else(|| TypeError::UndefinedType(name.clone()))
                            // Unless seen?
                            .and_then(|t| traverse_type(t, resolver, ctx, seen))
                    }
                }
                Type::Apply(type_constructor, argument) => Ok(Type::Apply(
                    traverse_type(*type_constructor, resolver, ctx, seen)?.into(),
                    traverse_type(*argument, resolver, ctx, seen)?.into(),
                )),
            }
        }

        traverse_type(self, resolver, ctx, &mut HashSet::default())
    }

    fn apply(self, subs: &typer::Substitutions) -> Self {
        match self {
            // Recurse into apply here to apply the whole chain of
            // substitutions?
            Self::Parameter(param) => subs
                .lookup(&param)
                .cloned()
                .map(|ty| ty.apply(subs))
                .unwrap_or_else(|| Self::Parameter(param)),
            Self::Arrow(domain, codomain) => {
                Self::Arrow(domain.apply(subs).into(), codomain.apply(subs).into())
            }
            Self::Forall(param, body) => {
                // This can be done without a Clone. Can add a closure method instead:
                // subs.except(param, |subs| Self::Forall, etc.)
                let mut subs = subs.clone();
                subs.remove(&param);
                Self::Forall(param, body.apply(&subs).into())
            }
            Self::Coproduct(coproduct) => Self::Coproduct(coproduct.apply(subs)),
            Self::Product(product) => Self::Product(product.apply(subs)),
            Self::Apply(constructor, argument) => {
                Self::Apply(constructor.apply(subs).into(), argument.apply(subs).into())
            }
            trivial => trivial,
        }
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        match self {
            Self::Parameter(param) => HashSet::from([*param]),
            Self::Arrow(tv0, tv1) => {
                let mut vars = tv0.free_variables();
                vars.extend(tv1.free_variables());
                vars
            }
            Self::Coproduct(c) => c.iter().flat_map(|(_, ty)| ty.free_variables()).collect(),
            Self::Product(x) => match x {
                ProductType::Tuple(elements) => {
                    elements.iter().flat_map(|ty| ty.free_variables()).collect()
                }
                ProductType::Struct(elements) => elements
                    .iter()
                    .flat_map(|(_, ty)| ty.free_variables())
                    .collect(),
            },
            Self::Forall(ty_var, ty) => {
                let mut unbound = ty.free_variables();
                unbound.remove(ty_var);
                unbound
            }
            Self::Apply(tv0, tv1) => {
                let mut vars = tv0.free_variables();
                vars.extend(tv1.free_variables());
                vars
            }
            _trivial => HashSet::default(),
        }
    }

    // Re-write this without the rename
    fn instantiate(&self) -> Self {
        let mut map = HashMap::default();
        fn rename(ty: Type, fresh: &mut HashMap<TypeParameter, Type>) -> Type {
            match ty {
                Type::Forall(ty_var, ty) => {
                    fresh.insert(ty_var, Type::fresh());
                    rename(*ty, fresh)
                }
                Type::Parameter(ty_var) => fresh
                    .get(&ty_var)
                    .cloned()
                    .unwrap_or_else(|| Type::Parameter(ty_var)),
                Type::Arrow(domain, codomain) => Type::Arrow(
                    rename(*domain, fresh).into(),
                    rename(*codomain, fresh).into(),
                ),
                Type::Apply(constructor, argument) => {
                    let constructor = rename(*constructor, fresh);
                    let argument = rename(*argument, fresh);
                    Type::Apply(constructor.into(), argument.into())
                }
                _ => ty,
            }
        }

        rename(self.clone(), &mut map)
    }

    pub fn fresh() -> Type {
        Self::Parameter(TypeParameter::fresh())
    }

    fn unify_tuples<A>(
        lhs: &[Type],
        rhs: &[Type],
        annotation: &A,
    ) -> Result<typer::Substitutions, TypeError>
    where
        A: Parsed,
    {
        if lhs.len() == rhs.len() {
            let mut substitution = typer::Substitutions::default();
            for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                substitution = substitution.compose(lhs.unify(rhs, annotation)?);
            }
            Ok(substitution)
        } else {
            Err(TypeError::BadTupleArity {
                lhs: ProductType::Tuple(lhs.to_vec()),
                rhs: ProductType::Tuple(rhs.to_vec()),
            })
        }
    }

    fn unify_coproducts<A>(
        lhs: &CoproductType,
        rhs: &CoproductType,
        annotation: &A,
    ) -> Result<typer::Substitutions, TypeError>
    where
        A: Parsed,
    {
        if lhs.arity() == rhs.arity() {
            let mut sub = typer::Substitutions::default();

            for ((lhs_constructor, lhs_ty), (rhs_constructor, rhs_ty)) in lhs.iter().zip(rhs.iter())
            {
                if lhs_constructor == rhs_constructor {
                    println!("unify_coproducts: `{lhs_ty}` `{rhs_ty}`");
                    sub = sub.compose(lhs_ty.unify(rhs_ty, annotation)?)
                } else {
                    Err(TypeError::IncompatibleCoproducts {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    })?
                }
            }

            Ok(sub)
        } else {
            Err(TypeError::IncompatibleCoproducts {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            })
        }
    }

    fn unify<A>(&self, rhs: &Type, annotation: &A) -> Result<typer::Substitutions, TypeError>
    where
        A: Parsed,
    {
        let lhs = self;

        match (lhs, rhs) {
            (Self::Constant(t), Self::Constant(u)) if t == u => Ok(typer::Substitutions::default()),
            (Self::Parameter(t), Self::Parameter(u)) if t == u => {
                Ok(typer::Substitutions::default())
            }
            (Self::Parameter(param), ty) | (ty, Self::Parameter(param)) => {
                if ty.free_variables().contains(param) {
                    Err(TypeError::InfiniteType {
                        param: param.clone(),
                        ty: ty.clone(),
                    })
                } else {
                    let mut substitution = typer::Substitutions::default();
                    substitution.add(param.clone(), ty.clone());
                    Ok(substitution)
                }
            }
            (Self::Product(ProductType::Tuple(lhs)), Self::Product(ProductType::Tuple(rhs))) => {
                Self::unify_tuples(lhs, rhs, annotation)
            }
            (Self::Product(ProductType::Struct(lhs)), Self::Product(ProductType::Struct(rhs))) => {
                Self::unify_structs(lhs, rhs, annotation)
            }
            (Self::Coproduct(lhs), Self::Coproduct(rhs)) => {
                Self::unify_coproducts(lhs, rhs, annotation)
            }
            (Self::Arrow(lhs_domain, lhs_codomain), Self::Arrow(rhs_domain, rhs_codomain)) => {
                let domain = lhs_domain.unify(rhs_domain, annotation)?;
                let codomain = lhs_codomain
                    .clone()
                    .apply(&domain)
                    .unify(&rhs_codomain.clone().apply(&domain), annotation)?;

                Ok(domain.compose(codomain))
            }
            (Self::Apply(cons1, arg1), Self::Apply(cons2, arg2)) => {
                let constructors = cons1.unify(cons2, annotation)?;
                let arguments = arg1
                    .clone()
                    .apply(&constructors)
                    .unify(&arg2.clone().apply(&constructors), annotation)?;
                Ok(constructors.compose(arguments))
            }
            // Also: c must not have type parameters
            (Self::Apply(constructor, _), rhs)
                if &**constructor == rhs && constructor.free_variables().is_empty() =>
            {
                Ok(typer::Substitutions::default())
            }
            (lhs, Self::Apply(constructor, _))
                if lhs == &**constructor && constructor.free_variables().is_empty() =>
            {
                Ok(typer::Substitutions::default())
            }
            (Self::Named(lhs), Self::Named(rhs)) if lhs == rhs => {
                Ok(typer::Substitutions::default())
            }
            (lhs, rhs) => {
                //                panic!("`{lhs}` != `{rhs}`");

                Err(TypeError::UnifyImpossible {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    position: { annotation.info().location().clone() },
                })
            }
        }
    }

    // Make a StructProduct (and a TupleProduct) type.
    // todo: this is not ideal. Store a Vec instead of a HashMap
    // This unifies without the nominal part of the type, if it
    // even has one (a name.)
    fn unify_structs<A>(
        lhs: &[(ast::Identifier, Type)],
        rhs: &[(ast::Identifier, Type)],
        annotation: &A,
    ) -> Result<typer::Substitutions, TypeError>
    where
        A: Parsed,
    {
        if lhs.len() == rhs.len() {
            let mut lhs_fields = lhs.iter().collect::<Vec<_>>();
            lhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut rhs_fields = rhs.iter().collect::<Vec<_>>();
            rhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut substitutions = typer::Substitutions::default();
            for ((lhs_id, lhs_ty), (rhs_id, rhs_ty)) in
                lhs_fields.drain(..).zip(rhs_fields.drain(..))
            {
                if lhs_id == rhs_id {
                    substitutions = substitutions.compose(lhs_ty.unify(rhs_ty, annotation)?)
                } else {
                    Err(TypeError::UnifyImpossible {
                        lhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                        rhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                        position: annotation.info().location().clone(),
                    })?
                }
            }

            Ok(substitutions)
        } else {
            Err(TypeError::UnifyImpossible {
                lhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                rhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                position: annotation.info().location().clone(),
            })
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter(ty_var) => write!(f, "{ty_var}"),
            Self::Constant(ty) => write!(f, "{ty}"),
            Self::Coproduct(ty) => write!(f, "{ty}"),
            Self::Product(ty) => write!(f, "{ty}"),
            Self::Arrow(ty0, ty1) => write!(f, "{ty0}->{ty1}"),
            Self::Forall(ty_var, ty) => write!(f, "forall {ty_var}.{ty}"),
            Self::Named(cons) => write!(f, "{cons}"),
            Self::Apply(cons, arg) => write!(f, "{cons}[{arg}]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BaseType {
    Unit,
    Int,
    Bool,
    Float,
    Text,
}

impl BaseType {
    pub fn into_type_name(self) -> TypeName {
        TypeName::new(match self {
            Self::Unit => "builtin::Unit",
            Self::Int => "builtin::Int",
            Self::Bool => "builtin::Bool",
            Self::Float => "builtin::Float",
            Self::Text => "builtin::Text",
        })
    }
}

impl fmt::Display for BaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => write!(f, "Unit"),
            Self::Int => write!(f, "Int"),
            Self::Bool => write!(f, "Bool"),
            Self::Float => write!(f, "Float"),
            Self::Text => write!(f, "Text"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProductType {
    Tuple(Vec<Type>),
    Struct(Vec<(ast::Identifier, Type)>),
}
impl ProductType {
    fn apply(self, subs: &typer::Substitutions) -> ProductType {
        match self {
            ProductType::Tuple(mut elements) => {
                ProductType::Tuple(elements.drain(..).map(|ty| ty.apply(subs)).collect())
            }
            ProductType::Struct(mut elements) => ProductType::Struct(
                elements
                    .drain(..)
                    .map(|(label, ty)| (label, ty.apply(subs)))
                    .collect(),
            ),
        }
    }
}

impl fmt::Display for ProductType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(elements) => {
                write!(f, "(")?;

                write!(
                    f,
                    "{}",
                    elements
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )?;

                write!(f, ")")
            }
            Self::Struct(_elements) => {
                todo!()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoproductType(Vec<(String, Type)>);

impl CoproductType {
    pub fn new(mut signature: Vec<(String, Type)>) -> Self {
        signature.sort_by(|(p, ..), (q, ..)| p.cmp(q));
        Self(signature)
    }

    fn find_constructor(&self, name: &ast::Identifier) -> Option<&Type> {
        let Self(constructors) = self;
        constructors
            .iter()
            .find_map(|(constructor, ty)| (name.as_str() == constructor).then_some(ty))
    }

    fn arity(&self) -> usize {
        let Self(constructors) = self;
        constructors.len()
    }

    fn iter(&self) -> Iter<'_, (String, Type)> {
        let Self(constructors) = self;
        constructors.iter()
    }

    fn apply(self, subs: &typer::Substitutions) -> Self {
        let Self(mut constructors) = self;
        Self(
            constructors
                .drain(..)
                .map(|(constructor, parameter_type)| (constructor, parameter_type.apply(subs)))
                .collect(),
        )
    }
}

impl fmt::Display for CoproductType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(constructors) = self;
        let line = constructors
            .iter()
            .map(|(c, ty)| format!("{c} {ty}"))
            .collect::<Vec<_>>()
            .join(" | ");

        write!(f, "{line}")
    }
}

pub type Typing<A = TypeInference> = Result<A, TypeError>;

#[derive(Debug)]
pub struct TypeInference {
    pub substitutions: typer::Substitutions,
    pub inferred_type: Type,
}

#[derive(Clone, Debug, Error)]
pub enum TypeError {
    #[error("Undefined symbol {0}")]
    UndefinedSymbol(ast::Identifier),

    #[error("{base} has no member {index}")]
    BadProjection {
        base: Type,
        index: ast::ProductIndex,
    },

    #[error("{param} is in {ty} which makes it infinite")]
    InfiniteType { param: TypeParameter, ty: Type },

    #[error("{lhs} and {rhs} are different product types")]
    BadTupleArity { lhs: ProductType, rhs: ProductType },

    #[error("{lhs} and {rhs} are different co-product types")]
    IncompatibleCoproducts {
        lhs: CoproductType,
        rhs: CoproductType,
    },

    #[error("{position}: {lhs} and {rhs} do not unify")]
    UnifyImpossible {
        lhs: Type,
        rhs: Type,
        position: SourcePosition,
    },

    #[error("No such type {0}")]
    UndefinedType(ast::TypeName),

    #[error("{coproduct} does not have a constructor {constructor}")]
    UndefinedCoproductConstructor {
        coproduct: ast::TypeName,
        constructor: ast::Identifier,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Binding {
    TypeTerm(String),
    ValueTerm(String),
}

impl From<ast::Identifier> for Binding {
    fn from(value: ast::Identifier) -> Self {
        Binding::ValueTerm(value.as_str().to_owned())
    }
}

impl From<ast::TypeName> for Binding {
    fn from(value: ast::TypeName) -> Self {
        Binding::TypeTerm(value.as_str().to_owned())
    }
}

// Couldn't this also be a Vec?
// It ought to be hierarchical just like the Environment in the interpreter.
#[derive(Debug, Clone, Default)]
pub struct TypingContext(HashMap<Binding, Type>);

impl TypingContext {
    pub fn bind(&mut self, binding: Binding, ty: Type) {
        self.0.insert(binding, ty.clone());
    }

    fn lookup(&self, binding: &Binding) -> Option<&Type> {
        let Self(map) = self;
        map.get(binding)
    }

    pub fn infer_type<A>(&self, e: &ast::Expression<A>) -> Typing
    where
        A: Clone + Parsed,
    {
        typer::synthesize_type(e, self)
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        let Self(map) = self;
        map.values().flat_map(|ty| ty.free_variables()).collect()
    }

    fn apply_substitutions(&self, subs: &typer::Substitutions) -> Self {
        let Self(map) = self;

        let mut map = map.clone();
        map.values_mut().for_each(|ty| *ty = ty.clone().apply(subs));

        Self(map)
    }
}

impl fmt::Display for TypingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (binding, ty) in &self.0 {
            match binding {
                Binding::TypeTerm(id) => writeln!(f, "{id} ::= {ty}")?,
                Binding::ValueTerm(id) => writeln!(f, "{id} :: {ty}")?,
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeParameter(u32);

mod typer {
    use std::{
        collections::HashMap,
        fmt,
        sync::atomic::{AtomicU32, Ordering},
    };

    use super::{
        BaseType, Parsed, ProductType, Type, TypeError, TypeInference, TypeParameter, Typing,
        TypingContext,
    };
    use crate::{
        ast::{
            self, Apply, Binding, Construct, ControlFlow, Expression, Lambda, Project,
            SelfReferential, Sequence,
        },
        parser::ParsingInfo,
    };

    static FRESH_TYPE_ID: AtomicU32 = AtomicU32::new(0);

    impl TypeParameter {
        pub fn fresh() -> Self {
            Self(FRESH_TYPE_ID.fetch_add(1, Ordering::SeqCst))
        }

        pub fn new_for_test(id: u32) -> Self {
            Self(id)
        }
    }

    impl fmt::Display for TypeParameter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let Self(id) = self;
            write!(f, "T#{id}")
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct Substitutions(HashMap<TypeParameter, Type>);

    impl Substitutions {
        pub fn add(&mut self, param: TypeParameter, ty: Type) {
            let Self(map) = self;
            map.insert(param, ty);
        }

        pub fn lookup(&self, param: &TypeParameter) -> Option<&Type> {
            let Self(map) = self;
            map.get(param)
        }

        pub fn remove(&mut self, param: &TypeParameter) {
            let Self(map) = self;
            map.remove(param);
        }

        pub fn compose(&self, Substitutions(mut rhs): Self) -> Self {
            let mut composed = rhs
                .drain()
                .map(|(param, ty)| (param, ty.apply(self)))
                .collect::<HashMap<_, _>>();

            let Self(lhs) = self;
            composed.extend(lhs.clone());

            Self(composed)
        }
    }

    // revisit this function
    fn infer_lambda<A>(
        ast::Parameter {
            name,
            type_annotation,
        }: &ast::Parameter<A>,
        body: &Expression<A>,
        info: &ParsingInfo,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        let expected_param_type = if let Some(param_type) = type_annotation {
            // Why doesn't it have to look something up?
            param_type.clone().synthesize_type(&mut HashMap::default()) // hmmmm
                                                                        //            ctx.lookup(&param_type.clone().into())
                                                                        //                .cloned()
                                                                        //                .ok_or_else(|| TypeError::UndefinedType(param_type.clone()))?
        } else {
            Type::fresh()
        };

        let mut ctx = ctx.clone();
        ctx.bind(name.clone().into(), expected_param_type.clone());

        let body = synthesize_type(body, &ctx)?;

        let inferred_param_type = expected_param_type.clone().apply(&body.substitutions);

        let annotation_unification = expected_param_type.unify(&inferred_param_type, info)?;

        let function_type = generalize_type(
            Type::Arrow(inferred_param_type.into(), body.inferred_type.into()),
            &ctx,
        );

        Ok(TypeInference {
            substitutions: body.substitutions.compose(annotation_unification),
            inferred_type: function_type,
        })
    }

    fn infer_application<A>(
        function: &ast::Expression<A>,
        argument: &ast::Expression<A>,
        annotation: &A,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        let function = synthesize_type(function, ctx)?;
        let argument = synthesize_type(argument, &ctx)?;

        let return_type = Type::fresh();

        let unified_substitutions = function
            .inferred_type
            .instantiate()
            .expand(ctx)?
            .apply(&function.substitutions)
            .unify(
                &Type::Arrow(
                    argument.inferred_type.apply(&function.substitutions).into(),
                    return_type.clone().into(),
                ),
                annotation,
            )?;

        Ok(TypeInference {
            substitutions: function
                .substitutions
                .compose(argument.substitutions)
                .compose(unified_substitutions),
            inferred_type: return_type.apply(&function.substitutions).expand(ctx)?,
        })
    }

    pub fn synthesize_type<A>(expr: &ast::Expression<A>, ctx: &TypingContext) -> Typing
    where
        A: Clone + Parsed,
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
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                let mut ctx = ctx.clone();
                ctx.bind(name.clone().into(), Type::fresh());
                infer_lambda(parameter, body, &annotation.info(), &ctx)
            }
            ast::Expression::Lambda(annotation, Lambda { parameter, body }) => {
                infer_lambda(parameter, body, &annotation.info(), ctx)
            }
            ast::Expression::Apply(annotation, Apply { function, argument }) => {
                infer_application(function, argument, annotation, ctx)
            }
            ast::Expression::Binding(
                _,
                Binding {
                    binder,
                    bound,
                    body,
                    ..
                },
            ) => infer_binding(binder, bound, body, ctx),
            ast::Expression::Construct(
                annotation,
                Construct {
                    name,
                    constructor,
                    argument,
                },
            ) => infer_coproduct(name, constructor, argument, annotation, ctx),
            ast::Expression::Product(_, product) => infer_product(product, ctx),
            ast::Expression::Project(_, Project { base, index }) => {
                infer_projection(base, index, ctx)
            }
            ast::Expression::Sequence(_, Sequence { this, and_then }) => {
                synthesize_type(this, ctx)?;
                synthesize_type(and_then, ctx)
            }
            ast::Expression::ControlFlow(annotation, control) => {
                infer_control_flow(control, &annotation, ctx)
            }
        }
    }

    fn infer_control_flow<A>(
        control: &ControlFlow<A>,
        annotation: &A,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        match control {
            ControlFlow::If {
                predicate,
                consequent,
                alternate,
            } => infer_if_expression(predicate, consequent, alternate, annotation, ctx),
        }
    }

    fn infer_if_expression<A>(
        predicate: &Expression<A>,
        consequent: &Expression<A>,
        alternate: &Expression<A>,
        annotation: &A,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        let predicate_type = synthesize_type(predicate, ctx)?;
        let predicate = predicate_type
            .inferred_type
            .unify(&Type::Constant(BaseType::Bool), annotation)
            .inspect_err(|e| println!("infer_if_expression: predicate unify error: {e}"))?;

        let ctx = ctx.apply_substitutions(&predicate_type.substitutions.clone());
        let consequent = synthesize_type(consequent, &ctx)?;
        let alternate = synthesize_type(alternate, &ctx)?;

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

    fn infer_binding<A>(
        binding: &ast::Identifier,
        bound: &ast::Expression<A>,
        body: &ast::Expression<A>,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        let bound = synthesize_type(bound, ctx)?;
        let bound_type = generalize_type(bound.inferred_type, ctx);

        let mut ctx = ctx.clone();
        ctx.bind(binding.clone().into(), bound_type);

        let TypeInference {
            substitutions,
            inferred_type,
        } = synthesize_type(body, &ctx.apply_substitutions(&bound.substitutions))?;

        Ok(TypeInference {
            substitutions: bound.substitutions.compose(substitutions),
            inferred_type,
        })
    }

    fn infer_projection<A>(
        base: &Expression<A>,
        index: &ast::ProductIndex,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        let base = synthesize_type(base, ctx)?;

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

    fn synthesize_type_of_constant(c: &ast::Constant, _ctx: &TypingContext) -> Typing {
        match c {
            ast::Constant::Int(..) => synthesize_trivial(BaseType::Int),
            ast::Constant::Float(..) => synthesize_trivial(BaseType::Float),
            ast::Constant::Text(..) => synthesize_trivial(BaseType::Text),
            ast::Constant::Bool(..) => synthesize_trivial(BaseType::Bool),
            ast::Constant::Unit => synthesize_trivial(BaseType::Unit),
        }
    }

    fn infer_coproduct<A>(
        name: &ast::TypeName,
        constructor: &ast::Identifier,
        argument: &Expression<A>,
        annotation: &A,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
    {
        if let ref constructed_type @ Type::Coproduct(ref coproduct) = ctx
            .lookup(&name.clone().into())
            .ok_or_else(|| TypeError::UndefinedType(name.clone()))
            .map(|ty| ty.instantiate())?
        {
            let argument = synthesize_type(argument, ctx)?;
            if let Some(rhs) = coproduct.find_constructor(constructor) {
                let lhs = &argument.inferred_type;

                let substitutions = argument.substitutions.compose(lhs.unify(rhs, annotation)?);
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

    fn infer_product<A>(product: &ast::Product<A>, ctx: &TypingContext) -> Typing
    where
        A: Clone + Parsed,
    {
        match product {
            ast::Product::Tuple(elements) => infer_tuple(elements, ctx),
            ast::Product::Struct { bindings } => infer_struct(bindings, ctx),
        }
    }

    fn infer_struct<A>(
        elements: &HashMap<ast::Identifier, ast::Expression<A>>,
        ctx: &TypingContext,
    ) -> Typing
    where
        A: Clone + Parsed,
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
        A: Clone + Parsed,
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

    fn generalize_type(ty: Type, ctx: &TypingContext) -> Type {
        let context_variables = ctx.free_variables();
        let type_variables = ty.free_variables();
        let non_quantified = type_variables.difference(&context_variables);

        non_quantified.fold(ty, |body, ty_var| Type::Forall(*ty_var, Box::new(body)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{Binding, CoproductType, TypingContext};
    use crate::{
        ast::{self, Apply, Construct, Lambda, Project, TypeExpression, TypeName},
        types::{BaseType, ProductType, Type, TypeParameter},
    };

    fn mk_apply(f: ast::Expression<()>, arg: ast::Expression<()>) -> ast::Expression<()> {
        ast::Expression::Apply(
            (),
            Apply {
                function: f.into(),
                argument: arg.into(),
            },
        )
    }

    fn mk_identity() -> ast::Expression<()> {
        ast::Expression::Lambda(
            (),
            Lambda {
                parameter: ast::Parameter::new(ast::Identifier::new("x")),
                body: ast::Expression::Variable((), ast::Identifier::new("x")).into(),
            },
        )
    }

    #[test]
    fn identity() {
        let id = mk_identity();

        let ctx = TypingContext::default();

        let e = mk_apply(
            id.clone(),
            ast::Expression::Literal((), ast::Constant::Int(10)),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, Type::Constant(BaseType::Int));

        let e = mk_apply(
            id.clone(),
            ast::Expression::Product(
                (),
                ast::Product::Tuple(vec![
                    ast::Expression::Literal((), ast::Constant::Int(10)),
                    ast::Expression::Literal((), ast::Constant::Float(1.0)),
                ]),
            ),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Tuple(vec![
                Type::Constant(BaseType::Int),
                Type::Constant(BaseType::Float)
            ]))
        );
    }

    #[test]
    fn tuples() {
        let ctx = TypingContext::default();
        let e = ast::Expression::Product(
            (),
            ast::Product::Tuple(vec![
                ast::Expression::Literal((), ast::Constant::Int(1)),
                ast::Expression::Literal((), ast::Constant::Float(1.0)),
            ]),
        );

        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Tuple(vec![
                Type::Constant(BaseType::Int),
                Type::Constant(BaseType::Float)
            ]))
        );

        let e = ast::Expression::Project(
            (),
            Project {
                base: e.into(),
                index: ast::ProductIndex::Tuple(0),
            },
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, Type::Constant(BaseType::Int));
    }

    #[test]
    fn applies() {
        let mut ctx = TypingContext::default();
        ctx.bind(
            Binding::TypeTerm("builtin::Int".to_owned()),
            Type::Constant(BaseType::Int),
        );
        ctx.bind(
            Binding::TypeTerm("builtin::Float".to_owned()),
            Type::Constant(BaseType::Float),
        );

        let e = mk_apply(
            mk_identity(),
            ast::Expression::Product(
                (),
                ast::Product::Struct {
                    bindings: HashMap::from([
                        (ast::Identifier::new("id"), mk_identity()),
                        (
                            ast::Identifier::new("x"),
                            mk_constant(ast::Constant::Int(1)),
                        ),
                        (
                            ast::Identifier::new("y"),
                            mk_constant(ast::Constant::Float(1.0)),
                        ),
                    ]),
                },
            ),
        );
        let t = ctx.infer_type(&e).unwrap();
        let expected_type = Type::Product(ProductType::Struct(Vec::from([
            (ast::Identifier::new("id"), mk_identity_type()),
            (ast::Identifier::new("x"), mk_constant_type(BaseType::Int)),
            (ast::Identifier::new("y"), mk_constant_type(BaseType::Float)),
        ])));
        t.inferred_type.unify(&expected_type, &()).unwrap();

        let e = mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0)));
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(t.inferred_type, mk_constant_type(BaseType::Float));

        let abs = ast::Expression::Lambda(
            (),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("x"),
                    type_annotation: Some(TypeExpression::Constant(TypeName::new(
                        "builtin::Float",
                    ))),
                },
                body: ast::Expression::Variable((), ast::Identifier::new("x")).into(),
            },
        );

        let e = mk_apply(abs, mk_constant(ast::Constant::Float(1.0)));

        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, mk_constant_type(BaseType::Float));
    }

    #[test]
    fn coproducts() {
        let mut ctx = TypingContext::default();
        let t = TypeParameter::fresh();
        ctx.bind(
            Binding::TypeTerm("Option".to_owned()),
            Type::Forall(
                t,
                Type::Coproduct(CoproductType::new(vec![
                    ("The".to_owned(), Type::Parameter(t)),
                    ("Nil".to_owned(), Type::Constant(BaseType::Unit)),
                ]))
                .into(),
            ),
        );

        let e = ast::Expression::Construct(
            (),
            Construct {
                name: ast::TypeName::new("Option"),
                constructor: ast::Identifier::new("The"),
                argument: mk_constant(ast::Constant::Float(1.0)).into(),
            },
        );
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(
            t.inferred_type,
            Type::Coproduct(CoproductType::new(vec![
                ("The".to_owned(), Type::Constant(BaseType::Float)),
                ("Nil".to_owned(), Type::Constant(BaseType::Unit)),
            ]))
        );

        let e = ast::Expression::Construct(
            (),
            Construct {
                name: ast::TypeName::new("Option"),
                constructor: ast::Identifier::new("Nil"),
                argument: mk_constant(ast::Constant::Unit).into(),
            },
        );
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(
            t.inferred_type,
            Type::Coproduct(CoproductType::new(vec![
                (
                    "The".to_owned(),
                    Type::Parameter(TypeParameter::new_for_test(0))
                ),
                ("Nil".to_owned(), Type::Constant(BaseType::Unit)),
            ]))
        )
    }

    #[test]
    fn polymorphic() {
        let mut ctx = TypingContext::default();

        let t = TypeParameter::fresh();
        ctx.bind(
            Binding::ValueTerm("id".to_owned()),
            Type::Arrow(Type::Parameter(t).into(), Type::Parameter(t).into()).into(),
            //            Type::Forall(
            //                t,
            //            ),
        );

        let e = mk_apply(
            ast::Expression::Variable((), ast::Identifier::new("id")),
            ast::Expression::Product(
                (),
                ast::Product::Struct {
                    bindings: HashMap::from([
                        (
                            ast::Identifier::new("x"),
                            mk_apply(mk_identity(), mk_constant(ast::Constant::Int(1))),
                        ),
                        (
                            ast::Identifier::new("y"),
                            mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0))),
                        ),
                    ]),
                },
            ),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Struct(Vec::from([
                (ast::Identifier::new("x"), mk_constant_type(BaseType::Int)),
                (ast::Identifier::new("y"), mk_constant_type(BaseType::Float)),
            ])))
        )
    }

    #[test]
    fn rank_n() {
        let mut gamma = TypingContext::default();
        gamma.bind(Binding::TypeTerm("Id".to_owned()), mk_identity_type());

        let apply_to_five_expr = ast::Expression::Lambda(
            (),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("f"), // Parameter `f`
                    type_annotation: Some(TypeExpression::Constant(TypeName::new("Id"))), // Annotate with the type alias
                },
                body: Box::new(ast::Expression::Apply(
                    (),
                    Apply {
                        function: ast::Expression::Variable((), ast::Identifier::new("f")).into(), // Apply `f`
                        argument: ast::Expression::Literal((), ast::Constant::Int(5)).into(), // Argument: 5
                    },
                )),
            },
        );

        let t = gamma.infer_type(&apply_to_five_expr).unwrap();
        println!("t::{t:?}");

        gamma.bind(Binding::ValueTerm("id".to_owned()), mk_identity_type());

        let apply_to_five_to_id = ast::Expression::Apply(
            (),
            Apply {
                function: apply_to_five_expr.clone().into(), // Use `applyToFive`
                argument: ast::Expression::Variable((), ast::Identifier::new("id")).into(), // Apply it to `id`
            },
        );

        let t = gamma.infer_type(&apply_to_five_to_id).unwrap();
        println!("t::{t:?}");
    }

    fn mk_identity_type() -> Type {
        let ty = TypeParameter::new_for_test(1);
        Type::Arrow(Type::Parameter(ty).into(), Type::Parameter(ty).into()).into()
    }

    fn mk_constant_type(ty: BaseType) -> Type {
        Type::Constant(ty)
    }

    fn mk_constant(int: ast::Constant) -> ast::Expression<()> {
        ast::Expression::Literal((), int)
    }
}
