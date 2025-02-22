use std::{
    collections::{HashMap, HashSet},
    fmt,
    slice::Iter,
};

use crate::{
    ast::{self, Expression, Identifier},
    types::typer::Substitutions,
};

/*
    The structure of this file is off.
    The toplevel module ought to be typer.
    What does the types/ typer dichotomy afford us?
    I don't really like the way the associated functions
      are distributed over the types either.
*/

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Parameter(TypeParameter),
    Base(BaseType),
    Product(ProductType),
    Coproduct(CoproductType),
    Function(Box<Type>, Box<Type>),
    Forall(TypeParameter, Box<Type>),
}

impl Type {
    fn apply(self, subs: &typer::Substitutions) -> Self {
        match self {
            // Recurse into apply here to apply the whole chain of
            // substitutions?
            Self::Parameter(param) => subs
                .lookup(&param)
                .cloned()
                .unwrap_or_else(|| Self::Parameter(param)),
            Self::Function(domain, codomain) => {
                Self::Function(Box::new(domain.apply(subs)), Box::new(codomain.apply(subs)))
            }
            Self::Forall(param, body) => {
                // This can be done without a Clone. Can add a closure method instead:
                // subs.except(param, |subs| Self::Forall, etc.)
                let mut subs = subs.clone();
                subs.remove(&param);
                Self::Forall(param, Box::new(body.apply(&subs)))
            }
            Self::Coproduct(coproduct) => Self::Coproduct(coproduct.apply(subs)),
            Self::Product(product) => Self::Product(product.apply(subs)),
            trivial => trivial,
        }
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        match self {
            Self::Parameter(param) => HashSet::from([*param]),
            Self::Function(tv0, tv1) => {
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
                    .values()
                    .flat_map(|ty| ty.free_variables())
                    .collect(),
            },
            Self::Forall(ty_var, ty) => {
                let mut unbound = ty.free_variables();
                unbound.remove(ty_var);
                unbound
            }
            _trivial => HashSet::default(),
        }
    }

    fn instantiate(&self) -> Self {
        let mut map = HashMap::default();
        fn rename(ty: Type, map: &mut HashMap<TypeParameter, Type>) -> Type {
            match ty {
                Type::Forall(ty_var, ty) => {
                    map.insert(ty_var, Type::fresh());
                    rename(*ty, map)
                }
                Type::Parameter(ty_var) => map
                    .get(&ty_var)
                    .cloned()
                    .unwrap_or_else(|| Type::Parameter(ty_var)),
                Type::Function(domain, codomain) => Type::Function(
                    Box::new(rename(*domain, map)),
                    Box::new(rename(*codomain, map)),
                ),
                _ => ty,
            }
        }
        rename(self.clone(), &mut map)
    }

    pub fn fresh() -> Type {
        Self::Parameter(TypeParameter::fresh())
    }

    fn unify_tuples(lhs: &[Type], rhs: &[Type]) -> Result<Substitutions, TypeError> {
        if lhs.len() == rhs.len() {
            let mut substitution = Substitutions::default();
            for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                substitution = substitution.compose(lhs.unify(rhs)?);
            }
            Ok(substitution)
        } else {
            Err(TypeError::BadTupleArity {
                lhs: ProductType::Tuple(lhs.to_vec()),
                rhs: ProductType::Tuple(rhs.to_vec()),
            })
        }
    }

    fn unify_coproducts(
        lhs: &CoproductType,
        rhs: &CoproductType,
    ) -> Result<Substitutions, TypeError> {
        if lhs.arity() == rhs.arity() {
            let mut sub = Substitutions::default();

            for ((lhs_constructor, lhs_ty), (rhs_constructor, rhs_ty)) in lhs.iter().zip(rhs.iter())
            {
                if lhs_constructor == rhs_constructor {
                    sub = sub.compose(lhs_ty.unify(rhs_ty)?)
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

    fn unify(&self, rhs: &Type) -> Result<Substitutions, TypeError> {
        let lhs = self;

        match (lhs, rhs) {
            (Self::Base(t), Self::Base(u)) if t == u => Ok(Substitutions::default()),
            (Self::Parameter(t), Self::Parameter(u)) if t == u => Ok(Substitutions::default()),
            (Self::Parameter(param), ty) | (ty, Self::Parameter(param)) => {
                if ty.free_variables().contains(param) {
                    Err(TypeError::InfiniteType {
                        param: param.clone(),
                        ty: ty.clone(),
                    })
                } else {
                    let mut substitution = Substitutions::default();
                    substitution.add(param.clone(), ty.clone());
                    Ok(substitution)
                }
            }
            (Self::Product(ProductType::Tuple(lhs)), Self::Product(ProductType::Tuple(rhs))) => {
                Self::unify_tuples(lhs, rhs)
            }
            (Self::Product(ProductType::Struct(lhs)), Self::Product(ProductType::Struct(rhs))) => {
                Self::unify_structs(lhs, rhs)
            }
            (Self::Coproduct(lhs), Self::Coproduct(rhs)) => Self::unify_coproducts(lhs, rhs),
            (
                Self::Function(lhs_domain, lhs_codomain),
                Self::Function(rhs_domain, rhs_codomain),
            ) => {
                let domain = lhs_domain.unify(rhs_domain)?;
                let codomain = (*lhs_codomain)
                    .clone()
                    .apply(&domain)
                    .unify(&(*rhs_codomain).clone().apply(&domain))?;

                Ok(domain.compose(codomain))
            }
            (lhs, rhs) => Err(TypeError::UnifyImpossible {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            }),
        }
    }

    // Make a StructProduct (and a TupleProduct) type.
    // todo: this is not ideal. Store a Vec instead of a HashMap
    // This unifies without the nominal part of the type, if it
    // even has one (a name.)
    fn unify_structs(
        lhs: &HashMap<Identifier, Type>,
        rhs: &HashMap<Identifier, Type>,
    ) -> Result<Substitutions, TypeError> {
        if lhs.len() == rhs.len() {
            let mut lhs_fields = lhs.iter().collect::<Vec<_>>();
            lhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut rhs_fields = rhs.iter().collect::<Vec<_>>();
            rhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut substitutions = Substitutions::default();
            for ((lhs_id, lhs_ty), (rhs_id, rhs_ty)) in
                lhs_fields.into_iter().zip(rhs_fields.into_iter())
            {
                if lhs_id == rhs_id {
                    substitutions = substitutions.compose(lhs_ty.unify(rhs_ty)?)
                } else {
                    Err(TypeError::UnifyImpossible {
                        lhs: Type::Product(ProductType::Struct(lhs.clone())),
                        rhs: Type::Product(ProductType::Struct(lhs.clone())),
                    })?
                }
            }

            Ok(substitutions)
        } else {
            Err(TypeError::UnifyImpossible {
                lhs: Type::Product(ProductType::Struct(lhs.clone())),
                rhs: Type::Product(ProductType::Struct(lhs.clone())),
            })
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter(ty_var) => write!(f, "{ty_var}"),
            Self::Base(ty) => write!(f, "{ty}"),
            Self::Coproduct(ty) => write!(f, "{ty}"),
            Self::Product(ty) => write!(f, "{ty}"),
            Self::Function(ty0, ty1) => write!(f, "{ty0}->{ty1}"),
            Self::Forall(ty_var, ty) => write!(f, "forall {ty_var}.{ty}"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductType {
    Tuple(Vec<Type>),
    Struct(HashMap<Identifier, Type>),
}
impl ProductType {
    fn apply(self, subs: &Substitutions) -> ProductType {
        match self {
            ProductType::Tuple(elements) => {
                ProductType::Tuple(elements.into_iter().map(|ty| ty.apply(subs)).collect())
            }
            ProductType::Struct(elements) => ProductType::Struct(
                elements
                    .into_iter()
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
                for element in elements {
                    write!(f, "{element},")?
                }
                write!(f, ")")
            }
            Self::Struct(_elements) => {
                todo!()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
        let Self(constructors) = self;
        Self(
            constructors
                .into_iter()
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

#[derive(Debug)]
pub enum TypeError {
    UndefinedSymbol(ast::Identifier),
    BadProjection {
        base: Type,
        index: ast::ProductIndex,
    },
    InfiniteType {
        param: TypeParameter,
        ty: Type,
    },
    BadTupleArity {
        lhs: ProductType,
        rhs: ProductType,
    },
    IncompatibleCoproducts {
        lhs: CoproductType,
        rhs: CoproductType,
    },
    UnifyImpossible {
        lhs: Type,
        rhs: Type,
    },
    UndefinedType(ast::TypeName),
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
        let Self(map) = self;
        map.insert(binding, ty);
    }

    fn lookup(&self, binding: &Binding) -> Option<&Type> {
        let Self(map) = self;
        map.get(binding)
    }

    pub fn infer_type(&self, e: &Expression) -> Typing {
        typer::infer(e, self)
    }

    fn free_variables(&self) -> HashSet<TypeParameter> {
        let Self(map) = self;
        map.values().flat_map(|ty| ty.free_variables()).collect()
    }

    fn apply(&self, subs: &typer::Substitutions) -> Self {
        let Self(map) = self;

        let mut map = map.clone();
        map.values_mut().for_each(|ty| *ty = ty.clone().apply(subs));

        Self(map)
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
        BaseType, ProductType, Type, TypeError, TypeInference, TypeParameter, Typing, TypingContext,
    };
    use crate::ast::{self, ControlFlow, Expression};

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
    fn infer_lambda(
        ast::Parameter {
            name,
            type_annotation,
        }: &ast::Parameter,
        body: &Expression,
        ctx: &TypingContext,
    ) -> Typing {
        let expected_param_type = if let Some(param_type) = type_annotation {
            ctx.lookup(&param_type.clone().into())
                .cloned()
                .ok_or_else(|| TypeError::UndefinedType(param_type.clone()))?
        } else {
            Type::fresh()
        };

        let mut ctx = ctx.clone();
        println!("Binding {} to {}", name, expected_param_type);
        ctx.bind(name.clone().into(), expected_param_type.clone());

        let body = infer(body, &ctx)?;

        let inferred_param_type = expected_param_type.clone().apply(&body.substitutions);

        let annotation_unification = expected_param_type.unify(&inferred_param_type)?;

        let function_type = generalize_type(
            Type::Function(inferred_param_type.into(), body.inferred_type.into()),
            &ctx,
        );

        Ok(TypeInference {
            substitutions: body.substitutions.compose(annotation_unification),
            inferred_type: function_type,
        })
    }

    fn infer_application(
        function: &ast::Expression,
        argument: &ast::Expression,
        ctx: &TypingContext,
    ) -> Typing {
        let function = infer(function, ctx)?;
        let argument = infer(argument, &ctx.apply(&function.substitutions))?;

        let return_type = Type::fresh();

        let unified_substitutions = function
            .inferred_type
            .instantiate()
            .apply(&argument.substitutions)
            .unify(&Type::Function(
                Box::new(argument.inferred_type),
                Box::new(return_type.clone()),
            ))?;

        let return_type = return_type.apply(&unified_substitutions);

        Ok(TypeInference {
            substitutions: function
                .substitutions
                .compose(argument.substitutions)
                .compose(unified_substitutions),
            inferred_type: return_type,
        })
    }

    pub fn infer(expr: &ast::Expression, ctx: &TypingContext) -> Typing {
        match expr {
            ast::Expression::Variable(binding) | ast::Expression::CallBridge(binding) => {
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
            ast::Expression::Literal(constant) => synthesize_type_of_constant(constant, ctx),
            ast::Expression::SelfReferential {
                name,
                parameter,
                body,
            } => {
                let mut ctx = ctx.clone();
                ctx.bind(name.clone().into(), Type::fresh());
                infer_lambda(parameter, body, &ctx)
            }
            ast::Expression::Lambda { parameter, body } => infer_lambda(parameter, body, ctx),
            ast::Expression::Apply { function, argument } => {
                infer_application(function, argument, ctx)
            }
            ast::Expression::Binding {
                binder,
                bound,
                body,
                ..
            } => infer_binding(binder, bound, body, ctx),
            ast::Expression::Construct {
                name,
                constructor,
                argument,
            } => infer_coproduct(name, constructor, argument, ctx),
            ast::Expression::Product(product) => infer_product(product, ctx),
            ast::Expression::Project { base, index } => infer_projection(base, index, ctx),
            ast::Expression::Sequence { this, and_then } => {
                infer(this, ctx)?;
                infer(and_then, ctx)
            }
            ast::Expression::ControlFlow(control) => infer_control_flow(control, ctx),
        }
    }

    fn infer_control_flow(control: &ControlFlow, ctx: &TypingContext) -> Typing {
        match control {
            ControlFlow::If {
                predicate,
                consequent,
                alternate,
            } => infer_if_expression(predicate, consequent, alternate, ctx),
        }
    }

    fn infer_if_expression(
        predicate: &Expression,
        consequent: &Expression,
        alternate: &Expression,
        ctx: &TypingContext,
    ) -> Typing {
        let predicate_type = infer(predicate, ctx)?;
        let predicate = predicate_type
            .inferred_type
            .unify(&Type::Base(BaseType::Bool))?;

        let consequent = infer(consequent, ctx)?;
        let alternate = infer(alternate, ctx)?;

        let branch = consequent.inferred_type.unify(&alternate.inferred_type)?;

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

    fn infer_binding(
        binding: &ast::Identifier,
        bound: &ast::Expression,
        body: &ast::Expression,
        ctx: &TypingContext,
    ) -> Typing {
        let bound = infer(bound, ctx)?;
        let bound_type = generalize_type(bound.inferred_type, ctx);

        let mut ctx = ctx.clone();
        ctx.bind(binding.clone().into(), bound_type);

        let TypeInference {
            substitutions,
            inferred_type,
        } = infer(body, &ctx.apply(&bound.substitutions))?;

        Ok(TypeInference {
            substitutions: bound.substitutions.compose(substitutions),
            inferred_type,
        })
    }

    fn infer_projection(
        base: &Expression,
        index: &ast::ProductIndex,
        ctx: &TypingContext,
    ) -> Typing {
        let base = infer(base, ctx)?;

        match (&base.inferred_type, index) {
            (Type::Product(ProductType::Tuple(elements)), ast::ProductIndex::Tuple(index))
                if *index < elements.len() =>
            {
                Ok(TypeInference {
                    substitutions: base.substitutions,
                    inferred_type: elements[*index].clone(),
                })
            }
            (Type::Product(ProductType::Struct(elements)), ast::ProductIndex::Struct(id))
                if elements.contains_key(id) =>
            {
                Ok(TypeInference {
                    substitutions: base.substitutions,
                    inferred_type: elements.get(id).cloned().unwrap(),
                })
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

    fn infer_coproduct(
        name: &ast::TypeName,
        constructor: &ast::Identifier,
        argument: &Expression,
        ctx: &TypingContext,
    ) -> Result<TypeInference, TypeError> {
        if let Some(constructed_type @ Type::Coproduct(coproduct)) = ctx
            .lookup(&name.clone().into())
            .map(|ty| ty.instantiate())
            .as_ref()
        {
            let argument = infer(argument, ctx)?;
            if let Some(expected_type) = coproduct.find_constructor(constructor) {
                let substitutions = argument
                    .substitutions
                    .compose(argument.inferred_type.unify(expected_type)?);

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
            Err(TypeError::UndefinedType(name.to_owned()))
        }
    }

    fn infer_product(product: &ast::Product, ctx: &TypingContext) -> Typing {
        match product {
            ast::Product::Tuple(elements) => infer_tuple(elements, ctx),
            ast::Product::Struct { bindings } => infer_struct(bindings, ctx),
        }
    }

    fn infer_struct(
        elements: &HashMap<ast::Identifier, ast::Expression>,
        ctx: &TypingContext,
    ) -> Typing {
        let mut substitutions = Substitutions::default();
        let mut types = Vec::with_capacity(elements.len());

        for (label, initializer) in elements {
            let initializer = ctx.infer_type(initializer)?;

            substitutions = substitutions.compose(initializer.substitutions);
            types.push((label.clone(), initializer.inferred_type));
        }

        Ok(TypeInference {
            substitutions,
            inferred_type: Type::Product(ProductType::Struct(types.into_iter().collect())),
        })
    }

    fn infer_tuple(elements: &[ast::Expression], ctx: &TypingContext) -> Typing {
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
            inferred_type: Type::Base(ty),
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
        ast,
        types::{BaseType, ProductType, Type, TypeParameter},
    };

    fn mk_apply(f: ast::Expression, arg: ast::Expression) -> ast::Expression {
        ast::Expression::Apply {
            function: f.into(),
            argument: arg.into(),
        }
    }

    fn mk_identity() -> ast::Expression {
        ast::Expression::Lambda {
            parameter: ast::Parameter::new(ast::Identifier::new("x")),
            body: ast::Expression::Variable(ast::Identifier::new("x")).into(),
        }
    }

    #[test]
    fn identity() {
        let id = mk_identity();

        let ctx = TypingContext::default();

        let e = mk_apply(id.clone(), ast::Expression::Literal(ast::Constant::Int(10)));
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, Type::Base(BaseType::Int));

        let e = mk_apply(
            id.clone(),
            ast::Expression::Product(ast::Product::Tuple(vec![
                ast::Expression::Literal(ast::Constant::Int(10)),
                ast::Expression::Literal(ast::Constant::Float(1.0)),
            ])),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Tuple(vec![
                Type::Base(BaseType::Int),
                Type::Base(BaseType::Float)
            ]))
        );
    }

    #[test]
    fn tuples() {
        let ctx = TypingContext::default();
        let e = ast::Expression::Product(ast::Product::Tuple(vec![
            ast::Expression::Literal(ast::Constant::Int(1)),
            ast::Expression::Literal(ast::Constant::Float(1.0)),
        ]));

        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Tuple(vec![
                Type::Base(BaseType::Int),
                Type::Base(BaseType::Float)
            ]))
        );

        let e = ast::Expression::Project {
            base: e.into(),
            index: ast::ProductIndex::Tuple(0),
        };
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, Type::Base(BaseType::Int));
    }

    #[test]
    fn applies() {
        let mut ctx = TypingContext::default();
        let e = mk_apply(
            mk_identity(),
            ast::Expression::Product(ast::Product::Struct {
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
            }),
        );
        let t = ctx.infer_type(&e).unwrap();
        let expected_type = Type::Product(ProductType::Struct(HashMap::from([
            (ast::Identifier::new("id"), mk_identity_type()),
            (ast::Identifier::new("x"), mk_trivial_type(BaseType::Int)),
            (ast::Identifier::new("y"), mk_trivial_type(BaseType::Float)),
        ])));

        t.inferred_type.unify(&expected_type).unwrap();

        let e = mk_apply(mk_identity(), mk_constant(ast::Constant::Float(1.0)));
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(t.inferred_type, mk_trivial_type(BaseType::Float));

        let abs = ast::Expression::Lambda {
            parameter: ast::Parameter {
                name: ast::Identifier::new("x"),
                type_annotation: Some(ast::TypeName::new("builtin::Float")),
            },
            body: ast::Expression::Variable(ast::Identifier::new("x")).into(),
        };

        let e = mk_apply(abs, mk_constant(ast::Constant::Float(1.0)));

        ctx.bind(
            Binding::TypeTerm("builtin::Int".to_owned()),
            Type::Base(BaseType::Int),
        );
        ctx.bind(
            Binding::TypeTerm("builtin::Float".to_owned()),
            Type::Base(BaseType::Float),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(t.inferred_type, mk_trivial_type(BaseType::Float));
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
                    ("Nil".to_owned(), Type::Base(BaseType::Unit)),
                ]))
                .into(),
            ),
        );

        let e = ast::Expression::Construct {
            name: ast::TypeName::new("Option"),
            constructor: ast::Identifier::new("The"),
            argument: mk_constant(ast::Constant::Float(1.0)).into(),
        };
        let t = ctx.infer_type(&e).unwrap();

        assert_eq!(
            t.inferred_type,
            Type::Coproduct(CoproductType::new(vec![
                ("The".to_owned(), Type::Base(BaseType::Float)),
                ("Nil".to_owned(), Type::Base(BaseType::Unit)),
            ]))
        )
    }

    #[test]
    fn polymorphic() {
        let mut ctx = TypingContext::default();

        let t = TypeParameter::fresh();
        ctx.bind(
            Binding::ValueTerm("id".to_owned()),
            Type::Function(Type::Parameter(t).into(), Type::Parameter(t).into()).into(),
            //            Type::Forall(
            //                t,
            //            ),
        );

        let e = mk_apply(
            ast::Expression::Variable(ast::Identifier::new("id")),
            ast::Expression::Product(ast::Product::Struct {
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
            }),
        );
        let t = ctx.infer_type(&e).unwrap();
        assert_eq!(
            t.inferred_type,
            Type::Product(ProductType::Struct(HashMap::from([
                (ast::Identifier::new("x"), mk_trivial_type(BaseType::Int)),
                (ast::Identifier::new("y"), mk_trivial_type(BaseType::Float)),
            ])))
        )
    }

    #[test]
    fn rank_n() {
        let mut gamma = TypingContext::default();
        gamma.bind(Binding::TypeTerm("Id".to_owned()), mk_identity_type());

        let apply_to_five_expr = ast::Expression::Lambda {
            parameter: ast::Parameter {
                name: ast::Identifier::new("f"),                 // Parameter `f`
                type_annotation: Some(ast::TypeName::new("Id")), // Annotate with the type alias
            },
            body: Box::new(ast::Expression::Apply {
                function: ast::Expression::Variable(ast::Identifier::new("f")).into(), // Apply `f`
                argument: ast::Expression::Literal(ast::Constant::Int(5)).into(), // Argument: 5
            }),
        };

        let t = gamma.infer_type(&apply_to_five_expr).unwrap();
        println!("t::{t:?}");

        gamma.bind(Binding::ValueTerm("id".to_owned()), mk_identity_type());

        let apply_to_five_to_id = ast::Expression::Apply {
            function: apply_to_five_expr.clone().into(), // Use `applyToFive`
            argument: ast::Expression::Variable(ast::Identifier::new("id")).into(), // Apply it to `id`
        };

        let t = gamma.infer_type(&apply_to_five_to_id).unwrap();
        println!("t::{t:?}");
    }

    fn mk_identity_type() -> Type {
        let ty = TypeParameter::new_for_test(1);
        Type::Function(Type::Parameter(ty).into(), Type::Parameter(ty).into()).into()
    }

    fn mk_trivial_type(ty: BaseType) -> Type {
        Type::Base(ty)
    }

    fn mk_constant(int: ast::Constant) -> ast::Expression {
        ast::Expression::Literal(int)
    }
}
