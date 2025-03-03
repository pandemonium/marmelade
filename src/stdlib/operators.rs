use crate::{
    bridge::*,
    context::CompileState,
    interpreter::{Base, Interpretation},
    lexer::Operator,
    typer::{BaseType, Type, TypeParameter},
};

// Think about the return type of this
pub fn import(env: &mut CompileState) -> Interpretation<()> {
    use Operator::*;

    let raw_lambda2 = |apply, signature| PartialRawLambda2 { apply, signature };

    define(Equals.id(), raw_lambda2(equals, binary_to_bool_type()), env)?;
    define(Gte.id(), raw_lambda2(gte, binary_to_bool_type()), env)?;
    define(Lte.id(), raw_lambda2(lte, binary_to_bool_type()), env)?;
    define(Gt.id(), raw_lambda2(gt, binary_to_bool_type()), env)?;
    define(Lt.id(), raw_lambda2(lt, binary_to_bool_type()), env)?;

    // These are bool -> bool -> bool
    // So they can be added like print_endline instead
    define(And.id(), Lambda2(and), env)?;
    define(Or.id(), Lambda2(or), env)?;
    define(Xor.id(), Lambda2(xor), env)?;

    define(Plus.id(), raw_lambda2(plus, binary_type()), env)?;
    define(Minus.id(), raw_lambda2(minus, binary_type()), env)?;
    define(Times.id(), raw_lambda2(times, binary_type()), env)?;
    define(Divides.id(), raw_lambda2(divides, binary_type()), env)?;
    define(Modulo.id(), raw_lambda2(modulo, binary_type()), env)
}

fn binary_to_bool_type() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Arrow(
            tp.clone(),
            Type::Arrow(tp, Type::Constant(BaseType::Bool).into()).into(),
        )
        .into(),
    )
}

fn binary_type() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Arrow(tp.clone(), Type::Arrow(tp.clone(), tp).into()).into(),
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

pub fn gte(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Bool(p >= q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Bool(p >= q)),
        (Base::Text(p), Base::Text(q)) => Some(Base::Bool(p >= q)),
        _otherwise => None,
    }
}

pub fn lte(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Bool(p <= q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Bool(p <= q)),
        (Base::Text(p), Base::Text(q)) => Some(Base::Bool(p <= q)),
        _otherwise => None,
    }
}

pub fn gt(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Bool(p > q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Bool(p > q)),
        (Base::Text(p), Base::Text(q)) => Some(Base::Bool(p > q)),
        _otherwise => None,
    }
}

pub fn lt(p: Base, q: Base) -> Option<Base> {
    match (p, q) {
        (Base::Int(p), Base::Int(q)) => Some(Base::Bool(p < q)),
        (Base::Float(p), Base::Float(q)) => Some(Base::Bool(p < q)),
        (Base::Text(p), Base::Text(q)) => Some(Base::Bool(p < q)),
        _otherwise => None,
    }
}

fn and(p: bool, q: bool) -> bool {
    p && q
}

fn or(p: bool, q: bool) -> bool {
    p || q
}

fn xor(p: bool, q: bool) -> bool {
    p ^ q
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
