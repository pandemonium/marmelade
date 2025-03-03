use std::collections::{HashMap, HashSet};

use super::{CoproductType, Parsed, Type, TypeParameter, Typing, TypingContext};
use crate::{
    ast::Identifier,
    typer::{ProductType, TypeError},
};

pub fn unify<A>(
    lhs: &Type,
    rhs: &Type,
    annotation: &A,
    ctx: &TypingContext,
) -> Typing<Substitutions>
where
    A: Parsed,
{
    UnificationContext::new(annotation, ctx).unify(lhs, rhs)
}

struct UnificationContext<'a, A> {
    annotation: &'a A,
    ctx: &'a TypingContext,
    seen: HashSet<Type>,
}

impl<'a, A> UnificationContext<'a, A> {
    fn new(annotation: &'a A, ctx: &'a TypingContext) -> Self {
        Self {
            annotation,
            ctx,
            seen: HashSet::default(),
        }
    }

    fn unify(&mut self, lhs: &Type, rhs: &Type) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        if self.seen.contains(lhs) && self.seen.contains(rhs) {
            return Ok(Substitutions::default());
        } else {
            self.seen.insert(lhs.to_owned());
            self.seen.insert(rhs.to_owned());
        }

        let lhs = &lhs.clone().instantiate().expand(self.ctx)?;
        let rhs = &rhs.clone().instantiate().expand(self.ctx)?;

        println!(
            "unify: expanded {} and {}",
            lhs.description(),
            rhs.description()
        );

        match (lhs, rhs) {
            (Type::Constant(t), Type::Constant(u)) if t == u => Ok(Substitutions::default()),
            (Type::Parameter(t), Type::Parameter(u)) if t == u => {
                println!("unify: `{t}` `{u}`");
                Ok(Substitutions::default())
            }
            (Type::Parameter(param), ty) | (ty, Type::Parameter(param)) => {
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
            (Type::Product(ProductType::Tuple(lhs)), Type::Product(ProductType::Tuple(rhs))) => {
                self.unify_tuples(lhs, rhs)
            }
            (Type::Product(ProductType::Struct(lhs)), Type::Product(ProductType::Struct(rhs))) => {
                self.unify_structs(lhs, rhs)
            }
            (Type::Coproduct(lhs), Type::Coproduct(rhs)) => self.unify_coproducts(lhs, rhs),
            (Type::Arrow(lhs_domain, lhs_codomain), Type::Arrow(rhs_domain, rhs_codomain)) => {
                let domain = self.unify(&lhs_domain.clone().expand(self.ctx)?, rhs_domain)?;

                let codomain = self.unify(
                    &lhs_codomain.clone().apply(&domain),
                    &rhs_codomain.clone().apply(&domain),
                )?;

                Ok(domain.compose(codomain))
            }
            (Type::Apply(lhs_constructor, lhs_at), Type::Apply(rhs_constructor, rhs_at)) => {
                let constructor_substitutions = self.unify(lhs_constructor, rhs_constructor)?;
                let at_substitutions = self.unify(
                    &lhs_at.clone().apply(&constructor_substitutions),
                    &rhs_at.clone().apply(&constructor_substitutions),
                )?;
                Ok(constructor_substitutions.compose(at_substitutions))
            }
            (Type::Named(lhs), Type::Named(rhs)) if lhs == rhs => Ok(Substitutions::default()),
            (lhs, rhs) => Err(TypeError::UnifyImpossible {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                position: { self.annotation.info().location().clone() },
            }),
        }
    }

    // todo: this is not ideal. Store a Vec instead of a HashMap
    // This unifies without the nominal part of the type, if it
    // even has one (a name.)
    fn unify_structs(
        &mut self,
        lhs: &[(Identifier, Type)],
        rhs: &[(Identifier, Type)],
    ) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        if lhs.len() == rhs.len() {
            let mut lhs_fields = lhs.iter().collect::<Vec<_>>();
            lhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut rhs_fields = rhs.iter().collect::<Vec<_>>();
            rhs_fields.sort_by(|(p, _), (q, _)| p.cmp(&q));

            let mut substitutions = Substitutions::default();
            for ((lhs_id, lhs_ty), (rhs_id, rhs_ty)) in
                lhs_fields.drain(..).zip(rhs_fields.drain(..))
            {
                if lhs_id == rhs_id {
                    substitutions = substitutions.compose(self.unify(lhs_ty, rhs_ty)?)
                } else {
                    Err(TypeError::UnifyImpossible {
                        lhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                        rhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                        position: self.annotation.info().location().clone(),
                    })?
                }
            }

            Ok(substitutions)
        } else {
            Err(TypeError::UnifyImpossible {
                lhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                rhs: Type::Product(ProductType::Struct(lhs.to_vec())),
                position: self.annotation.info().location().clone(),
            })
        }
    }

    fn unify_coproducts(
        &mut self,
        lhs: &CoproductType,
        rhs: &CoproductType,
    ) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        if lhs.arity() == rhs.arity() {
            let mut sub = Substitutions::default();

            for ((lhs_constructor, lhs_ty), (rhs_constructor, rhs_ty)) in lhs.iter().zip(rhs.iter())
            {
                if lhs_constructor == rhs_constructor {
                    println!(
                        "unify_coproducts: `{}` `{}`",
                        lhs_ty.description(),
                        rhs_ty.description()
                    );
                    sub = sub.compose(self.unify(lhs_ty, rhs_ty)?)
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

    fn unify_tuples(&mut self, lhs: &[Type], rhs: &[Type]) -> Typing<Substitutions>
    where
        A: Parsed,
    {
        if lhs.len() == rhs.len() {
            let mut substitution = Substitutions::default();
            for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                println!(
                    "unify_tuples: `{}` `{}`",
                    lhs.description(),
                    rhs.description()
                );
                substitution = substitution.compose(self.unify(lhs, rhs)?);
            }
            Ok(substitution)
        } else {
            Err(TypeError::BadTupleArity {
                lhs: ProductType::Tuple(lhs.to_vec()),
                rhs: ProductType::Tuple(rhs.to_vec()),
            })
        }
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
