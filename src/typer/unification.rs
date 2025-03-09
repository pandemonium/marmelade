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
    seen: HashSet<(Type, Type)>,
}

impl<'a, A> UnificationContext<'a, A>
where
    A: Parsed,
{
    fn new(annotation: &'a A, ctx: &'a TypingContext) -> Self {
        println!("-------> New Unification Context");

        Self {
            annotation,
            ctx,
            seen: HashSet::default(),
        }
    }

    fn is_reentrant(&self, pair: &(Type, Type)) -> bool {
        self.seen.contains(pair)
    }

    fn enter(&mut self, pair: (Type, Type)) {
        self.seen.insert(pair);
    }

    fn unify(&mut self, lhs: &Type, rhs: &Type) -> Typing<Substitutions> {
        let uni = match (lhs, rhs) {
            (Type::Constant(t), Type::Constant(u)) if t == u => Ok(Substitutions::default()),
            (Type::Parameter(t), Type::Parameter(u)) if t == u => Ok(Substitutions::default()),
            (Type::Apply(lhs_constructor, lhs_at), Type::Apply(rhs_constructor, rhs_at))
                if lhs_constructor == rhs_constructor && lhs_at == rhs_at =>
            {
                Ok(Substitutions::default())
            }
            _otherwise => self.unify_expanded(lhs.clone(), rhs.clone()),
        };

        uni.inspect(|x| println!("unify unified: {lhs} with {rhs}, with {x:?}"))
    }

    fn unify_expanded(&mut self, lhs: Type, rhs: Type) -> Typing<Substitutions> {
        println!(
            "unify_expanded: {} and {}",
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
                if ty.free_variables().contains(&param) {
                    Err(TypeError::InfiniteType {
                        param: param.clone(),
                        ty: ty.clone(),
                    })
                } else {
                    // Create a nice constructor for this case.
                    let mut substitution = Substitutions::default();
                    substitution.add(param.clone(), ty.clone());
                    Ok(substitution)
                }
            }
            ref relation @ (
                Type::Product(ProductType::Tuple(ref lhs_tuple)),
                Type::Product(ProductType::Tuple(ref rhs_tuple)),
            ) if !self.is_reentrant(&relation) => {
                self.enter(relation.clone());
                self.unify_tuples(lhs_tuple, rhs_tuple)
            }
            ref relation @ (
                Type::Product(ProductType::Struct(ref lhs)),
                Type::Product(ProductType::Struct(ref rhs)),
            ) if !self.is_reentrant(&relation) => {
                self.enter(relation.clone());
                self.unify_structs(&lhs, &rhs)
            }
            ref relation @ (Type::Coproduct(ref lhs), Type::Coproduct(ref rhs))
                if !self.is_reentrant(&relation) =>
            {
                //        self.enter(relation.clone());
                self.unify_coproducts(&lhs, &rhs)
            }
            (Type::Arrow(lhs_domain, lhs_codomain), Type::Arrow(rhs_domain, rhs_codomain)) => {
                //                let lhs_domain = self.expand_type(*lhs_domain.clone())?;
                //                let rhs_domain = self.expand_type(*rhs_domain.clone())?;
                let domain = self.unify(
                    //                    &lhs_domain.clone().expand(self.ctx)?,
                    //                    &rhs_domain.clone().expand(self.ctx)?,
                    &lhs_domain,
                    &rhs_domain,
                )?;

                let codomain = self.unify(
                    &lhs_codomain.clone().apply(&domain),
                    &rhs_codomain.clone().apply(&domain),
                )?;

                Ok(domain.compose(codomain))
            }
            (Type::Apply(lhs_constructor, lhs_at), Type::Apply(rhs_constructor, rhs_at)) => {
                let constructor = self.unify(&lhs_constructor, &rhs_constructor)?;
                let at = self.unify(
                    &lhs_at.clone().apply(&constructor),
                    &rhs_at.clone().apply(&constructor),
                )?;
                Ok(constructor.compose(at))
            }
            (lhs, rhs) if lhs == rhs => Ok(Substitutions::default()),
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
        println!("unify_tuples: ----> {:?} {:?}", lhs, rhs);

        if lhs.len() == rhs.len() {
            let mut substitution = Substitutions::default();
            for (lhs, rhs) in lhs.into_iter().zip(rhs.into_iter()) {
                //                let lhs = &lhs.clone().expand(self.ctx)?;
                //                let rhs = &rhs.clone().expand(self.ctx)?;
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
        println!("addsub: {param} -> {ty}");
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
