use std::{
    collections::{HashMap, HashSet},
    fmt,
    slice::Iter,
};
use thiserror::Error;
use unification::Substitutions;

use crate::{ast, lexer::SourcePosition, parser::ParsingInfo};

mod inferencing;
mod unification;

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
    Named(ast::TypeName),
    Apply(Box<Type>, Box<Type>),
}

impl Type {
    pub fn description(&self) -> String {
        match self {
            Self::Parameter(tp) => format!("Parameter({tp})"),
            Self::Constant(ty) => format!("Constant({ty})"),
            Self::Product(ty) => format!("Product({ty})"),
            Self::Coproduct(ty) => format!("Coproduct({ty})"),
            Self::Arrow(ty0, ty1) => format!("({} -> {})", ty0.description(), ty1.description()),
            Self::Forall(tv, body) => format!("Forall({tv}, {})", body.description()),
            Self::Named(nm) => format!("Named({nm})"),
            Self::Apply(ty0, ty1) => format!("{}[{}]", ty0.description(), ty1.description()),
        }
    }

    pub fn generalize(self, ctx: &TypingContext) -> Type {
        let context_variables = ctx.free_variables();
        let type_variables = self.free_variables();
        let non_quantified = type_variables.difference(&context_variables);

        non_quantified.fold(self, |body, ty_var| Type::Forall(*ty_var, Box::new(body)))
    }

    pub fn expand(self, ctx: &TypingContext) -> Typing<Self> {
        match self {
            Self::Named(name) => ctx
                .lookup(&Binding::TypeTerm(name.as_str().to_owned()))
                .cloned()
                .ok_or_else(|| TypeError::UndefinedType(name)),
            Self::Apply(constructor, ref argument) => {
                constructor.apply_constructor(*argument.clone(), ctx)
            }
            otherwise => Ok(otherwise),
        }
    }

    fn apply_constructor(self, at: Type, ctx: &TypingContext) -> Typing<Type> {
        match self {
            Self::Forall(param, body) => {
                let mut subs = Substitutions::default();
                subs.add(param, at);
                // Should sub here.
                Ok(body.instantiate())
            }
            Self::Named(..) => self.expand(ctx)?.apply_constructor(at, ctx),

            otherwise => Err(TypeError::WrongKind(otherwise)),
        }
    }

    fn apply(self, subs: &Substitutions) -> Self {
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
    pub fn instantiate(&self) -> Self {
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

    pub fn unify<A>(&self, rhs: &Type, annotation: &A, ctx: &TypingContext) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        unification::unify(self, rhs, annotation, ctx)
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

pub const BASE_TYPES: &[BaseType] = &[
    BaseType::Unit,
    BaseType::Int,
    BaseType::Bool,
    BaseType::Float,
    BaseType::Text,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BaseType {
    Unit,
    Int,
    Bool,
    Float,
    Text,
}

impl BaseType {
    pub fn type_name(&self) -> ast::TypeName {
        ast::TypeName::new(match self {
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
    fn apply(self, subs: &Substitutions) -> ProductType {
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
            Self::Struct(elements) => {
                write!(f, "{{")?;
                write!(
                    f,
                    "{}",
                    elements
                        .iter()
                        .map(|(field, ty)| format!("{field} : {ty}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )?;
                write!(f, "}}")
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

    fn apply(self, subs: &Substitutions) -> Self {
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
    pub substitutions: Substitutions,
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
    #[error("{0} does not take type parameters")]
    WrongKind(Type),
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
        A: fmt::Display + Clone + Parsed,
    {
        inferencing::infer_type(e, self)
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        let Self(map) = self;
        map.values().flat_map(|ty| ty.free_variables()).collect()
    }

    fn apply_substitutions(&self, subs: &Substitutions) -> Self {
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

pub mod internal {
    use std::{
        fmt,
        sync::atomic::{AtomicU32, Ordering},
    };

    use super::TypeParameter;

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
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::{Binding, CoproductType, TypingContext};
    use crate::{
        ast::{self, Apply, Inject, Lambda, Project, TypeExpression, TypeName},
        parser::ParsingInfo,
        typer::{BaseType, ProductType, Type, TypeParameter},
    };

    fn mk_apply<A>(f: ast::Expression<A>, arg: ast::Expression<A>) -> ast::Expression<A>
    where
        A: fmt::Display + Clone,
    {
        ast::Expression::Apply(
            f.annotation().clone(),
            Apply {
                function: f.into(),
                argument: arg.into(),
            },
        )
    }

    fn mk_identity() -> ast::Expression<ParsingInfo> {
        ast::Expression::Lambda(
            ParsingInfo::default(),
            Lambda {
                parameter: ast::Parameter::new(ast::Identifier::new("x")),
                body: ast::Expression::Variable(ParsingInfo::default(), ast::Identifier::new("x"))
                    .into(),
            },
        )
    }

    #[test]
    fn identity() {
        let id = mk_identity();

        let ctx = TypingContext::default();

        let e = mk_apply(
            id.clone(),
            ast::Expression::Literal(id.annotation().clone(), ast::Constant::Int(10)),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, Type::Constant(BaseType::Int));

        let e = mk_apply(
            id.clone(),
            ast::Expression::Product(
                id.annotation().clone(),
                ast::Product::Tuple(vec![
                    ast::Expression::Literal(id.annotation().clone(), ast::Constant::Int(10)),
                    ast::Expression::Literal(id.annotation().clone(), ast::Constant::Float(1.0)),
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
            ParsingInfo::default(),
            ast::Product::Tuple(vec![
                ast::Expression::Literal(ParsingInfo::default(), ast::Constant::Int(1)),
                ast::Expression::Literal(ParsingInfo::default(), ast::Constant::Float(1.0)),
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
            ParsingInfo::default(),
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
        // Re-write the whole tree, substituting all constant types? Is that really good?
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
                ParsingInfo::default(),
                ast::Product::Struct(Vec::from([
                    (ast::Identifier::new("id"), mk_identity()),
                    (
                        ast::Identifier::new("x"),
                        mk_constant(ast::Constant::Int(1)),
                    ),
                    (
                        ast::Identifier::new("y"),
                        mk_constant(ast::Constant::Float(1.0)),
                    ),
                ])),
            ),
        );
        let t = ctx.infer_type(&e).unwrap();
        let expected_type = Type::Product(ProductType::Struct(Vec::from([
            (ast::Identifier::new("id"), mk_identity_type()),
            (ast::Identifier::new("x"), mk_constant_type(BaseType::Int)),
            (ast::Identifier::new("y"), mk_constant_type(BaseType::Float)),
        ])));
        t.inferred_type.unify(&expected_type, &(), &ctx).unwrap();

        let e = mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0)));
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(t.inferred_type, mk_constant_type(BaseType::Float));

        let abs = ast::Expression::Lambda(
            ParsingInfo::default(),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("x"),
                    type_annotation: Some(TypeExpression::Constant(TypeName::new(
                        "builtin::Float",
                    ))),
                },
                body: ast::Expression::Variable(ParsingInfo::default(), ast::Identifier::new("x"))
                    .into(),
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

        let e = ast::Expression::Inject(
            ParsingInfo::default(),
            Inject {
                name: ast::TypeName::new("Option"),
                constructor: ast::Identifier::new("Nil"),
                argument: mk_constant(ast::Constant::Unit).into(),
            },
        );
        let t = ctx.infer_type(&e).unwrap();

        //        assert_eq!(
        //            t.inferred_type,
        //            Type::Coproduct(CoproductType::new(vec![
        //                (
        //                    "The".to_owned(),
        //                    Type::Parameter(TypeParameter::new_for_test(0))
        //                ),
        //                ("Nil".to_owned(), Type::Constant(BaseType::Unit)),
        //            ]))
        //        );

        assert!(matches!(
            t.inferred_type,
            Type::Coproduct(CoproductType(ref variants))
                if variants.len() == 2
                && matches!(variants.as_slice(),
                    [(nil, Type::Constant(BaseType::Unit)), (the, Type::Parameter(..))]
                    if the == "The" && nil == "Nil"
                )
        ));

        let e = ast::Expression::Inject(
            ParsingInfo::default(),
            Inject {
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
            ast::Expression::Variable(ParsingInfo::default(), ast::Identifier::new("id")),
            ast::Expression::Product(
                ParsingInfo::default(),
                ast::Product::Struct(Vec::from([
                    (
                        ast::Identifier::new("x"),
                        mk_apply(mk_identity(), mk_constant(ast::Constant::Int(1))),
                    ),
                    (
                        ast::Identifier::new("y"),
                        mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0))),
                    ),
                ])),
            ),
        );

        // There is an ordering problem in this verification
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
            ParsingInfo::default(),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("f"), // Parameter `f`
                    type_annotation: Some(TypeExpression::Constant(TypeName::new("Id"))), // Annotate with the type alias
                },
                body: Box::new(ast::Expression::Apply(
                    ParsingInfo::default(),
                    Apply {
                        function: ast::Expression::Variable(
                            ParsingInfo::default(),
                            ast::Identifier::new("f"),
                        )
                        .into(), // Apply `f`
                        argument: ast::Expression::Literal(
                            ParsingInfo::default(),
                            ast::Constant::Int(5),
                        )
                        .into(), // Argument: 5
                    },
                )),
            },
        );

        let t = gamma.infer_type(&apply_to_five_expr).unwrap();
        println!("t::{t:?}");

        gamma.bind(Binding::ValueTerm("id".to_owned()), mk_identity_type());

        let apply_to_five_to_id = ast::Expression::Apply(
            ParsingInfo::default(),
            Apply {
                function: apply_to_five_expr.clone().into(), // Use `applyToFive`
                argument: ast::Expression::Variable(
                    ParsingInfo::default(),
                    ast::Identifier::new("id"),
                )
                .into(), // Apply it to `id`
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

    fn mk_constant(int: ast::Constant) -> ast::Expression<ParsingInfo> {
        ast::Expression::Literal(ParsingInfo::default(), int)
    }
}
