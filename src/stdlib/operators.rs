use crate::{
    bridge::*,
    context::CompileState,
    interpreter::{Base, Interpretation, Value},
    lexer::Operator,
    typer::{BaseType, Type, TypeParameter},
};

// Think about the return type of this
pub fn import(env: &mut CompileState) -> Interpretation<()> {
    use Operator::*;

    //    let raw_lambda2 = |apply, signature| PartialRawLambda2 { apply, signature };

    let base_lambda2 = |apply: fn(Base, Base) -> Option<Base>, signature| PartialRawLambda2 {
        apply: move |p: Value, q: Value| {
            p.try_into_base_type()
                .zip(q.try_into_base_type())
                .and_then(|(p, q)| apply(p, q).map(Value::Base))
        },
        signature,
    };

    define(Equals.id(), base_lambda2(equals, binary_to_bool()), env)?;
    define(Gte.id(), base_lambda2(gte, binary_to_bool()), env)?;
    define(Lte.id(), base_lambda2(lte, binary_to_bool()), env)?;
    define(Gt.id(), base_lambda2(gt, binary_to_bool()), env)?;
    define(Lt.id(), base_lambda2(lt, binary_to_bool()), env)?;

    // These are bool -> bool -> bool
    // So they can be added like print_endline instead
    define(And.id(), Lambda2(and), env)?;
    define(Or.id(), Lambda2(or), env)?;
    define(Xor.id(), Lambda2(xor), env)?;

    define(TupleCons.id(), TupleConsSyntax, env)?;

    define(Plus.id(), base_lambda2(plus, binary()), env)?;
    define(Minus.id(), base_lambda2(minus, binary()), env)?;
    define(Times.id(), base_lambda2(times, binary()), env)?;
    define(Divides.id(), base_lambda2(divides, binary()), env)?;
    define(Modulo.id(), base_lambda2(modulo, binary()), env)
}

fn binary_to_bool() -> Type {
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

fn binary() -> Type {
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

fn tuple_cons(p: Value, q: Value) -> Value {
    Value::Tuple(vec![p, q])
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
