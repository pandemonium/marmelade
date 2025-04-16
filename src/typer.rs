use std::{
    collections::{HashMap, HashSet},
    fmt,
    slice::Iter,
    vec::IntoIter,
};
use thiserror::Error;

use crate::{
    ast::{
        self, DomainExpression, Expression, Identifier, Pattern, TypeExpression, TypeName,
        ValueDeclaration,
    },
    lexer::SourceLocation,
    parser::ParsingInfo,
};
use unification::Substitutions;

mod checking;
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

type UntypedExpression = Expression<ParsingInfo>;
type UntypedValueDeclaration = ValueDeclaration<ParsingInfo>;

#[derive(Debug, Default)]
pub struct TypeChecker(TypingContext);

impl TypeChecker {
    pub fn new(ctx: TypingContext) -> Self {
        Self(ctx)
    }

    // I would like to assign a type to every AST node
    pub fn check_declaration(&mut self, declaration: &UntypedValueDeclaration) -> Typing<()> {
        let typing_context = self.typing_context();
        if let Some(signature) = &declaration.type_signature {
            let type_scheme = signature.synthesize_type(typing_context)?;
            let expected_type = type_scheme.clone().instantiate(typing_context)?;
            // Expand the type?
            self.check(
                expected_type.expand_type(&typing_context)?,
                &declaration.declarator.expression,
            )?;
            let id = declaration.clone().binder;
            println!("type_check: `{id}` is `{type_scheme}`");
            self.bind_type(id, type_scheme);
        } else {
            let inference = self
                .typing_context()
                .infer_type(&declaration.declarator.expression)?;
            let type_scheme = inference
                .inferred_type
                .apply(&inference.substitutions)
                .generalize(typing_context);
            let id = declaration.clone().binder;
            println!("type_check: `{id}` is `{type_scheme}`");
            self.bind_type(id, type_scheme);
        }

        Ok(())
    }

    fn bind_type(&mut self, id: Identifier, type_scheme: TypeScheme) {
        let Self(typing_context) = self;
        typing_context.bind(id.into(), type_scheme);
    }

    fn check(&self, expected_type: Type, expression: &UntypedExpression) -> Typing<()> {
        checking::check(expression, expected_type, self.typing_context()).map(|_| ())
    }

    fn typing_context(&self) -> &TypingContext {
        let Self(ctx) = self;
        ctx
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeScheme {
    pub quantifiers: Vec<TypeParameter>,
    pub body: Type,
}

impl TypeScheme {
    pub fn new(quantifiers: &[TypeParameter], body: Type) -> Self {
        Self {
            quantifiers: quantifiers.to_vec(),
            body,
        }
    }

    pub fn from_constant(body: Type) -> Self {
        Self {
            quantifiers: vec![],
            body,
        }
    }

    pub fn is_type_constructor(&self) -> bool {
        !self.quantifiers.is_empty()
    }

    fn into_type_apply_tree(self, name: TypeName) -> TypeScheme {
        let quantifiers = &self.quantifiers;
        let type_apply_tree = quantifiers.iter().fold(Type::Named(name), |tree, param| {
            Type::Apply(tree.into(), Type::Parameter(param.clone()).into())
        });
        Self::new(&self.quantifiers, type_apply_tree)
    }

    pub fn instantiate(self, _ctx: &TypingContext) -> Typing<Type> {
        let mut subs = Substitutions::default();
        for param in self.quantifiers {
            subs.add(param, Type::fresh());
        }
        Ok(self.body.apply(&subs))
    }

    pub fn free_variables(&self) -> &[TypeParameter] {
        &self.quantifiers
    }

    // LOTS and LOTS of cloning of this poor Substitutions
    // Will I ever want to _not_ apply subs to the type scheme inside
    // the typing context? So perhaps &mut self.
    fn apply(self, subs: &Substitutions) -> TypeScheme {
        let Self { quantifiers, body } = self;

        let mut subs = subs.clone();
        for q in &quantifiers {
            subs.remove(&q);
        }

        Self {
            quantifiers,
            body: body.apply(&subs),
        }
    }

    pub fn map_body(self, f: fn(Type) -> Type) -> Self {
        let Self { quantifiers, body } = self;
        Self {
            quantifiers,
            body: f(body),
        }
    }
}

impl fmt::Display for TypeScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { quantifiers, body } = self;
        write!(
            f,
            "forall {}. {}",
            quantifiers
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            body
        )
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
pub enum Type {
    Parameter(TypeParameter),
    Constant(BaseType),
    Product(ProductType),
    Coproduct(CoproductType),
    Arrow(Box<Type>, Box<Type>),
    Named(ast::TypeName),
    Apply(Box<Type>, Box<Type>),
}

impl Type {
    pub fn is_arrow(&self) -> bool {
        matches!(self, Self::Arrow(..))
    }

    pub fn description(&self) -> String {
        match self {
            Self::Parameter(tp) => format!("Parameter({tp})"),
            Self::Constant(ty) => format!("Constant({ty})"),
            Self::Product(ty) => format!("Product({ty})"),
            Self::Coproduct(ty) => format!("Coproduct({ty})"),
            Self::Arrow(ty0, ty1) => format!("({} -> {})", ty0.description(), ty1.description()),
            Self::Named(nm) => format!("Alias({nm})"),
            Self::Apply(ty0, ty1) => format!("{}[{}]", ty0.description(), ty1.description()),
        }
    }

    pub fn expand_type(self, ctx: &TypingContext) -> Typing<Type> {
        match self {
            Type::Named(name) => ctx
                .lookup(&name.clone().into())
                .ok_or_else(|| TypeError::UndefinedType(name))?
                .instantiate(ctx),
            Type::Apply(constuctor, at) => constuctor.apply_constructor(&mut vec![*at], ctx),
            otherwise => Ok(otherwise),
        }
    }

    fn apply_constructor(self, arguments: &mut Vec<Type>, ctx: &TypingContext) -> Typing<Type> {
        match self {
            Type::Named(name) => {
                let TypeScheme {
                    mut quantifiers,
                    body,
                } = ctx
                    .lookup_scheme(&name.clone().into())
                    .cloned()
                    .ok_or_else(|| TypeError::UndefinedType(name))?;

                let mut subs = Substitutions::default();
                for (param, arg) in quantifiers.drain(..).zip(arguments.drain(..)) {
                    subs.add(param, arg);
                }

                Ok(body.apply(&subs))
            }
            Type::Apply(constructor, at) => {
                arguments.push(*at);
                constructor.apply_constructor(arguments, ctx)
            }
            otherwise => Err(TypeError::WrongKind(otherwise)),
        }
    }

    pub fn generalize(self, ctx: &TypingContext) -> TypeScheme {
        let context_variables = ctx.free_variables();
        let type_variables = self.free_variables();
        let free = type_variables.difference(&context_variables);

        TypeScheme {
            quantifiers: free.cloned().collect(),
            body: self,
        }
    }

    pub fn apply(self, subs: &Substitutions) -> Self {
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
            Self::Coproduct(coproduct) => Self::Coproduct(coproduct.apply(subs)),
            Self::Product(product) => Self::Product(product.apply(subs)),
            Self::Apply(constructor, argument) => {
                Self::Apply(constructor.apply(subs).into(), argument.apply(subs).into())
            }
            trivial => trivial,
        }
    }

    pub fn free_variables(&self) -> HashSet<TypeParameter> {
        match self {
            Self::Parameter(param) => HashSet::from([*param]),
            Self::Arrow(tv0, tv1) => {
                let mut vars = tv0.free_variables();
                vars.extend(tv1.free_variables());
                vars
            }
            Self::Coproduct(c) => c.iter().flat_map(|(_, ty)| ty.free_variables()).collect(),
            Self::Product(x) => match x {
                ProductType::Tuple(TupleType(elements)) => {
                    elements.iter().flat_map(|ty| ty.free_variables()).collect()
                }
                ProductType::Struct(elements) => elements
                    .iter()
                    .flat_map(|(_, ty)| ty.free_variables())
                    .collect(),
            },
            Self::Apply(tv0, tv1) => {
                let mut vars = tv0.free_variables();
                vars.extend(tv1.free_variables());
                vars
            }
            _trivial => HashSet::default(),
        }
    }

    pub fn fresh() -> Type {
        Self::Parameter(TypeParameter::fresh())
    }

    pub fn unify<A>(&self, rhs: &Type, annotation: &A) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        unification::unify(self, rhs, annotation)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter(ty_var) => write!(f, "{ty_var}"),
            Self::Constant(ty) => write!(f, "{ty}"),
            Self::Coproduct(ty) => write!(f, "{ty}"),
            Self::Product(ty) => write!(f, "{ty}"),
            Self::Arrow(ty0, ty1) if ty1.is_arrow() => write!(f, "{ty0}->({ty1})"),
            Self::Arrow(ty0, ty1) => write!(f, "{ty0}->{ty1}"),
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

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash, Ord)]
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
            Self::Unit => "Unit",
            Self::Int => "Int",
            Self::Bool => "Bool",
            Self::Float => "Float",
            Self::Text => "Text",
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

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
pub enum ProductType {
    Tuple(TupleType),
    Struct(Vec<(ast::Identifier, Type)>),
}

impl ProductType {
    fn apply(self, subs: &Substitutions) -> ProductType {
        match self {
            ProductType::Tuple(tuple) => ProductType::Tuple(tuple.apply(subs)),
            ProductType::Struct(elements) => ProductType::Struct(
                elements
                    .into_iter()
                    .map(|(label, ty)| (label, ty.apply(subs)))
                    .collect(),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
pub struct TupleType(pub Vec<Type>);

impl TupleType {
    fn apply(self, subs: &Substitutions) -> Self {
        let Self(elements) = self;
        Self(elements.into_iter().map(|ty| ty.apply(subs)).collect())
    }

    pub fn arity(&self) -> usize {
        let Self(elements) = self;
        elements.len()
    }

    // When and where can I do this?
    fn unspine(self) -> Self {
        let Self(mut elements) = self;
        let first = elements.remove(0);

        Self({
            let mut tail = elements
                .into_iter()
                .flat_map(|el| {
                    if let Type::Product(ProductType::Tuple(spine)) = el {
                        spine.unspine().0
                    } else {
                        vec![el]
                    }
                })
                .collect::<Vec<_>>();

            tail.insert(0, first);
            tail
        })
    }
}

impl fmt::Display for ProductType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(TupleType(elements)) => {
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

// Change this from (String, Type) into (String, Vec<Type>)
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
pub struct CoproductType(Vec<(String, Type)>);

impl CoproductType {
    pub fn new(mut signature: Vec<(String, Type)>) -> Self {
        signature.sort_by(|(p, ..), (q, ..)| p.cmp(q));
        Self(signature)
    }

    fn constructor_signature(&self, name: &ast::Identifier) -> Option<&Type> {
        let Self(constructors) = self;
        constructors
            .iter()
            .find_map(|(constructor, ty)| (&name.as_str() == constructor).then_some(ty))
    }

    fn arity(&self) -> usize {
        let Self(constructors) = self;
        constructors.len()
    }

    pub fn iter(&self) -> Iter<'_, (String, Type)> {
        let Self(constructors) = self;
        constructors.iter()
    }

    pub fn into_iter(self) -> IntoIter<(String, Type)> {
        let Self(constructors) = self;
        constructors.into_iter()
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

impl TypeInference {
    pub fn new(substitutions: Substitutions, inferred_type: Type) -> Self {
        Self {
            substitutions,
            inferred_type,
        }
    }

    pub fn trivially(this: Type) -> Self {
        Self {
            substitutions: Substitutions::default(),
            inferred_type: this,
        }
    }

    pub fn fresh() -> Self {
        Self::trivially(Type::fresh())
    }
}

impl fmt::Display for TypeInference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            substitutions,
            inferred_type,
        } = self;
        write!(f, "{inferred_type} [{substitutions}]")?;

        Ok(())
    }
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
        position: SourceLocation,
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

    #[error("Undefined quantifier {quantifier} in {in_type}")]
    UndefinedQuantifier { quantifier: TypeName, in_type: Type },

    #[error("Undefined quantifier {quantifier} in {in_expression}")]
    UndefinedQuantifierInTypeExpression {
        quantifier: TypeName,
        in_expression: TypeExpression<ParsingInfo>,
    },

    #[error("Superfluous quantifier {quantifier} in {in_type}")]
    SuperfluousQuantification { quantifier: TypeName, in_type: Type },

    #[error("Impossible to match {pattern} with {scrutinee}")]
    PatternMatchImpossible {
        pattern: Pattern<ParsingInfo>,
        scrutinee: Type,
    },

    #[error("Undefined struct label {label} at {position}")]
    UndefinedField {
        position: SourceLocation,
        label: Identifier,
    },

    #[error("Expected {expected_type}, received: {literal}")]
    ExpectedType {
        expected_type: Type,
        literal: Expression<ParsingInfo>,
    },

    #[error("Deconstruction does not cover all of `{scrutinee}` at {at}. Remaining: {residual}")]
    IncompleteDeconstruction {
        at: SourceLocation,
        scrutinee: Expression<ParsingInfo>,
        residual: DomainExpression,
    },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Binding {
    TypeTerm(String),
    ValueTerm(String),
}

impl Binding {
    pub fn is_value_binding(&self) -> bool {
        matches!(self, Self::ValueTerm(..))
    }

    pub fn as_str(&self) -> &str {
        match self {
            Binding::TypeTerm(name) | Binding::ValueTerm(name) => &name,
        }
    }
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeTerm(name) => write!(f, "Type@{name}"),
            Self::ValueTerm(name) => write!(f, "Value@{name}"),
        }
    }
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
pub struct TypingContext {
    bindings: HashMap<Binding, TypeScheme>,
}

impl TypingContext {
    pub fn bind(&mut self, binding: Binding, scheme: TypeScheme) {
        self.bindings.insert(binding, scheme);
    }

    pub fn lookup(&self, binding: &Binding) -> Option<TypeScheme> {
        let scheme = self.bindings.get(binding).cloned()?;
        let scheme = if !binding.is_value_binding() && scheme.is_type_constructor() {
            scheme.into_type_apply_tree(TypeName::new(binding.as_str()))
        } else {
            scheme
        };

        Some(scheme)
    }

    pub fn lookup_scheme(&self, binding: &Binding) -> Option<&TypeScheme> {
        self.bindings.get(binding)
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        let mut bound = HashSet::new();
        let mut free = HashSet::default();

        for TypeScheme { quantifiers, body } in self.bindings.values() {
            bound.extend(quantifiers);
            free.extend(body.free_variables());
        }

        free.difference(&bound).cloned().collect()
    }

    fn apply_substitutions(&self, subs: &Substitutions) -> Self {
        let mut ctx = self.clone();

        for scheme in ctx.bindings.values_mut() {
            *scheme = scheme.clone().apply(subs)
        }

        ctx
    }
}

impl fmt::Display for TypingContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (binding, ty) in &self.bindings {
            match binding {
                Binding::TypeTerm(id) => writeln!(f, "{id} ::= {ty}")?,
                Binding::ValueTerm(id) => writeln!(f, "{id} :: {ty}")?,
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
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
        ast::{self, Apply, Inject, Lambda, Project},
        parser::ParsingInfo,
        typer::{BaseType, ProductType, TupleType, Type, TypeParameter, TypeScheme},
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

    // Weird that it is unable to type this as forall a. \a. a
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
            Type::Product(ProductType::Tuple(TupleType(vec![
                Type::Constant(BaseType::Int),
                Type::Constant(BaseType::Float)
            ])))
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
            Type::Product(ProductType::Tuple(TupleType(vec![
                Type::Constant(BaseType::Int),
                Type::Constant(BaseType::Float)
            ])))
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
            Type::Constant(BaseType::Int).generalize(&ctx),
        );
        ctx.bind(
            Binding::TypeTerm("builtin::Float".to_owned()),
            Type::Constant(BaseType::Float).generalize(&ctx),
        );

        let _e = mk_apply(
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

        let t = ctx
            .infer_type(&ast::Expression::Product(
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
            ))
            .unwrap();

        let expected_type = Type::Product(ProductType::Struct(Vec::from([
            (ast::Identifier::new("id"), mk_identity_type()),
            (ast::Identifier::new("x"), mk_constant_type(BaseType::Int)),
            (ast::Identifier::new("y"), mk_constant_type(BaseType::Float)),
        ])));

        t.inferred_type.unify(&expected_type, &()).unwrap();

        let e = mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0)));
        let t = ctx.infer_type(&e).unwrap();

        // it has not applied the type parameters
        assert_eq!(t.inferred_type, mk_constant_type(BaseType::Float));

        let abs = ast::Expression::Lambda(
            ParsingInfo::default(),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("x"),
                    type_annotation: Some(Type::Constant(BaseType::Float)),
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
            TypeScheme {
                quantifiers: vec![t],
                body: Type::Coproduct(CoproductType::new(vec![
                    ("The".to_owned(), Type::Parameter(t)),
                    ("Nil".to_owned(), Type::Constant(BaseType::Unit)),
                ]))
                .into(),
            },
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
            t.inferred_type.expand_type(&ctx).unwrap(),
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
            t.inferred_type.expand_type(&ctx).unwrap(),
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
            Type::Arrow(Type::Parameter(t).into(), Type::Parameter(t).into()).generalize(&ctx),
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
        gamma.bind(
            Binding::TypeTerm("Id".to_owned()),
            mk_identity_type().generalize(&gamma),
        );

        let apply_to_five_expr = ast::Expression::Lambda(
            ParsingInfo::default(),
            Lambda {
                parameter: ast::Parameter {
                    name: ast::Identifier::new("f"),           // Parameter `f`
                    type_annotation: Some(mk_identity_type()), // Annotate with the type alias
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

        gamma.bind(
            Binding::ValueTerm("id".to_owned()),
            mk_identity_type().generalize(&gamma),
        );

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
