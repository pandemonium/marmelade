use crate::{
    bridge::*,
    context::Linkage,
    interpreter::{Base, Interpretation, Value},
    lexer::Operator,
    typer::{BaseType, Type, TypeParameter, TypeScheme},
};

// Think about the return type of this
pub fn import(linkage: &mut Linkage) -> Interpretation<()> {
    use Operator::*;

    let raw_lambda2 = |apply, signature| PartialRawLambda2 { apply, signature };

    let base_lambda2 = |apply: fn(Base, Base) -> Option<Base>, signature| PartialRawLambda2 {
        apply: move |p: Value, q: Value| {
            p.try_into_base_type()
                .zip(q.try_into_base_type())
                .and_then(|(p, q)| apply(p, q).map(Value::Base))
        },
        signature,
    };

    define(
        Equals.as_identifier(),
        raw_lambda2(equals, binary_to_bool()),
        linkage,
    )?;
    define(
        Gte.as_identifier(),
        base_lambda2(gte, binary_to_bool()),
        linkage,
    )?;
    define(
        Lte.as_identifier(),
        base_lambda2(lte, binary_to_bool()),
        linkage,
    )?;
    define(
        Gt.as_identifier(),
        base_lambda2(gt, binary_to_bool()),
        linkage,
    )?;
    define(
        Lt.as_identifier(),
        base_lambda2(lt, binary_to_bool()),
        linkage,
    )?;

    // These are bool -> bool -> bool
    // So they can be added like print_endline instead
    define(And.as_identifier(), Lambda2(and), linkage)?;
    define(Or.as_identifier(), Lambda2(or), linkage)?;
    define(Xor.as_identifier(), Lambda2(xor), linkage)?;

    define(Tuple.as_identifier(), TupleSyntax, linkage)?;

    define(Plus.as_identifier(), base_lambda2(plus, binary()), linkage)?;
    define(
        Minus.as_identifier(),
        base_lambda2(minus, binary()),
        linkage,
    )?;
    define(
        Times.as_identifier(),
        base_lambda2(times, binary()),
        linkage,
    )?;
    define(
        Division.as_identifier(),
        base_lambda2(divides, binary()),
        linkage,
    )?;
    define(
        Modulo.as_identifier(),
        base_lambda2(modulo, binary()),
        linkage,
    )
}

fn binary_to_bool() -> TypeScheme {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    TypeScheme {
        quantifiers: vec![ty.clone()],
        body: Type::Arrow(
            tp.clone(),
            Type::Arrow(tp, Type::Constant(BaseType::Bool).into()).into(),
        ),
    }
}

fn binary() -> TypeScheme {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));

    TypeScheme {
        quantifiers: vec![ty.clone()],
        body: Type::Arrow(tp.clone(), Type::Arrow(tp.clone(), tp).into()),
    }
}

pub fn equals(p: Value, q: Value) -> Option<Value> {
    match (p, q) {
        (Value::Base(Base::Int(p)), Value::Base(Base::Int(q))) => {
            Some(Value::Base(Base::Bool(p == q)))
        }
        (Value::Base(Base::Float(p)), Value::Base(Base::Float(q))) => {
            Some(Value::Base(Base::Bool(p == q)))
        }
        (Value::Base(Base::Bool(p)), Value::Base(Base::Bool(q))) => {
            Some(Value::Base(Base::Bool(p == q)))
        }
        (Value::Base(Base::Text(p)), Value::Base(Base::Text(q))) => {
            Some(Value::Base(Base::Bool(p == q)))
        }
        (Value::Tuple(mut p), Value::Tuple(mut q)) => {
            let result = p.len() == q.len()
                && p.drain(..)
                    .zip(q.drain(..))
                    .map(|(p, q)| equals(p, q))
                    .all(|v| matches!(v, Some(Value::Base(Base::Bool(true)))));

            Some(Value::Base(Base::Bool(result)))
        }
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
