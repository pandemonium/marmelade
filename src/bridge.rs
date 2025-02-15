use crate::{
    ast::{Expression, Identifier, Parameter},
    interpreter::{Environment, Interpretation, RuntimeError, Scalar, Value},
    types::{TrivialType, Type},
};

pub type CallResult<A> = Result<A, RuntimeError>;

// This thing works well for regular functions but it does not solve
// operators. Is it type check and then monomorphisation that is
// required here? How would it know which symbol to put in there
// base off of types?
// I could always mangle the type signature into its name.
pub trait Bridge {
    fn arity(&self) -> usize;
    fn evaluate(&self, e: &Environment) -> CallResult<Value>;
    fn signature(&self) -> Type;

    fn lambda_tree(&self, target: Identifier) -> Expression {
        (0..self.arity()).rfold(Expression::CallBridge(target), |acc, x| {
            Expression::Lambda {
                parameter: Parameter::new(Identifier::new(&format!("p{x}"))),
                body: acc.into(),
            }
        })
    }
}

pub struct Lambda1<A, R>(fn(A) -> R);

impl<A, R> Bridge for Lambda1<A, R>
where
    A: TryFrom<Value, Error = RuntimeError>,
    R: Into<Value>,
{
    fn arity(&self) -> usize {
        1
    }

    fn evaluate(&self, e: &Environment) -> CallResult<Value> {
        let Self(f) = self;
        Ok(f(e.lookup(&Identifier::new("p0")).cloned()?.try_into()?).into())
    }

    fn signature(&self) -> Type {
        todo!()
    }
}

pub struct Lambda2<A, B, R>(pub fn(A, B) -> R);

impl<A, B, R> Bridge for Lambda2<A, B, R>
where
    A: TryFrom<Value, Error = RuntimeError>,
    B: TryFrom<Value, Error = RuntimeError>,
    R: Into<Value>,
{
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, e: &Environment) -> CallResult<Value> {
        let Self(f) = self;
        Ok(f(
            e.lookup(&Identifier::new("p0")).cloned()?.try_into()?,
            e.lookup(&Identifier::new("p1")).cloned()?.try_into()?,
        )
        .into())
    }

    fn signature(&self) -> Type {
        todo!()
    }
}

// See if I can do something with the Add, Sub, Mul and Div type classes
// for this thing.
// Or could I impl Bridge for all tripples that have Add?
#[derive(Debug, Clone)]
pub struct PartialRawLambda2(pub fn(Scalar, Scalar) -> Option<Scalar>);

impl Bridge for PartialRawLambda2 {
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, e: &Environment) -> CallResult<Value> {
        let Self(f) = self;

        // They have to be scalars.
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?.try_into_scalar();
        let p1 = e.lookup(&Identifier::new("p1")).cloned()?.try_into_scalar();

        p0.zip(p1)
            .and_then(|(p0, p1)| f.clone()(p0, p1))
            .map(Value::Scalar)
            .ok_or_else(|| RuntimeError::InapplicableLamda2) // bring in arguments here later
    }

    fn signature(&self) -> Type {
        todo!()
    }
}

pub fn define<F>(surface_name: Identifier, bridge: F, env: &mut Environment) -> Interpretation<()>
where
    F: Bridge + 'static,
{
    let bridge_name = surface_name.scoped_with("bridge");
    let tree = bridge.lambda_tree(bridge_name.clone());
    env.insert_binding(bridge_name, Value::bridge(bridge));

    // this captures a closure where user functions do not exist
    // But how could this possibly matter?
    let tree = tree.reduce(env)?;
    env.insert_binding(surface_name, tree);

    Ok(())
}

impl TryFrom<Value> for i64 {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Scalar(Scalar::Int(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Trivial(TrivialType::Int)))
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Scalar(Scalar::Int(value))
    }
}

impl TryFrom<Value> for f64 {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Scalar(Scalar::Float(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Trivial(TrivialType::Int)))
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Scalar(Scalar::Float(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn open_file(x: i64) -> i64 {
        println!("open_file: {x}");
        1
    }

    #[test]
    fn playtime() {
        let x = define(
            Identifier::new("open_file"),
            Lambda1(open_file),
            &mut Environment::default(),
        )
        .unwrap();
    }
}
