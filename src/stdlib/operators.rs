use crate::{
    bridge,
    context::CompileState,
    interpreter::{Base, Interpretation},
    lexer::Operator,
    types::{BaseType, Type, TypeParameter},
};

// Think about the return type of this
pub fn import(env: &mut CompileState) -> Interpretation<()> {
    bridge::define(
        Operator::Equals.id(),
        bridge::PartialRawLambda2 {
            apply: equals,
            signature: make_binary_boolean_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Plus.id(),
        bridge::PartialRawLambda2 {
            apply: plus,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Minus.id(),
        bridge::PartialRawLambda2 {
            apply: minus,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Times.id(),
        bridge::PartialRawLambda2 {
            apply: times,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Divides.id(),
        bridge::PartialRawLambda2 {
            apply: divides,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Modulo.id(),
        bridge::PartialRawLambda2 {
            apply: modulo,
            signature: make_binary_operator(),
        },
        env,
    )
}

fn make_binary_boolean_operator() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Function(
            tp.clone(),
            Type::Function(tp, Type::Base(BaseType::Bool).into()).into(),
        )
        .into(),
    )
}

fn make_binary_operator() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Function(tp.clone(), Type::Function(tp.clone(), tp).into()).into(),
    )
}

pub fn equals(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Bool(p == q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Bool(p == q)),
        (Base::Bool(p), Base::Bool(q)) => Some(Base::Bool(p == q)),
        (Base::Text(p), Base::Text(q)) => Some(Base::Bool(p == q)),
        _otherwise => None,
    }
}

pub fn plus(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Int(p + q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Float(p + q)),
        _otherwise => None,
    }
}

pub fn minus(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Int(p - q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Float(p - q)),
        _otherwise => None,
    }
}

pub fn times(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Int(p * q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Float(p * q)),
        _otherwise => None,
    }
}

pub fn divides(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Int(p / q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Float(p / q)),
        _otherwise => None,
    }
}

pub fn modulo(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Int(p % q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Float(p % q)),
        _otherwise => None,
    }
}
