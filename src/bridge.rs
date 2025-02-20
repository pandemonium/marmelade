use crate::{
    ast::{Expression, Identifier, Parameter},
    context::InterpretationContext,
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
        // Call into the typer here?
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
        // Call into the typer here?
        todo!()
    }
}

// See if I can do something with the Add, Sub, Mul and Div type classes
// for this thing.
// Or could I impl Bridge for all tripples that have Add?
#[derive(Debug, Clone)]
pub struct PartialRawLambda2 {
    pub apply: fn(Scalar, Scalar) -> Option<Scalar>,
    pub signature: Type,
}

impl Bridge for PartialRawLambda2 {
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, e: &Environment) -> CallResult<Value> {
        // They have to be scalars.
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?.try_into_scalar();
        let p1 = e.lookup(&Identifier::new("p1")).cloned()?.try_into_scalar();

        p0.zip(p1)
            .and_then(|(p0, p1)| self.apply.clone()(p0, p1))
            .map(Value::Scalar)
            .ok_or_else(|| RuntimeError::InapplicableLamda2) // bring in arguments here later
    }

    // Can I implement this?
    // let t = Type::fresh();
    // Type::Function(t.into(), t.into())
    // and then generalize_type lifts a Forall?
    // But these are not forall a. a -> a -> a, though.
    // Because there are hidden constraints.
    // Can I type these without introducing constraints?
    // Should I just say that all three types are fresh?
    // But should I check it, then?
    fn signature(&self) -> Type {
        self.signature.clone()
    }
}

// This function must take something like or otherwise named CompilationContext instead.
// Because it has to insert a type binding.
// struct CompilationContext {
//     env: Environment,
//     ctx: TypingContext,
// }
pub fn define<B>(
    surface_name: Identifier,
    bridge: B,
    InterpretationContext {
        typing_context,
        interpreter_environment,
    }: &mut InterpretationContext,
) -> Interpretation<()>
where
    B: Bridge + 'static,
{
    let bridge_name = surface_name.scoped_with("bridge");

    typing_context.bind(bridge_name.clone().into(), bridge.signature());
    typing_context.bind(surface_name.clone().into(), bridge.signature());

    let tree = bridge.lambda_tree(bridge_name.clone());
    interpreter_environment.insert_binding(bridge_name, Value::bridge(bridge));

    let tree = tree.reduce(interpreter_environment)?;
    interpreter_environment.insert_binding(surface_name, tree);

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

    // Bring this test back, but fix Lambda1::signature and Lambda2::signature first
    //    #[test]
    fn playtime() {
        let _x = define(
            Identifier::new("open_file"),
            Lambda1(open_file),
            &mut InterpretationContext::default(),
        )
        .unwrap();
    }
}
