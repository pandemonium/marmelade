use crate::{
    ast::{Expression, Identifier, Lambda, Parameter},
    context::CompileState,
    interpreter::{Base, Environment, Interpretation, RuntimeError, Value},
    types::{BaseType, Type, TypeParameter},
};

pub type CallResult<A> = Result<A, RuntimeError>;

#[derive(Debug, Clone)]
pub struct BridgingInfo {
    pub intended_target: Identifier,
}

impl BridgingInfo {
    pub fn new(id: Identifier) -> Self {
        Self {
            intended_target: id,
        }
    }
}

pub trait Bridge {
    fn arity(&self) -> usize;
    fn evaluate(&self, e: &Environment) -> CallResult<Value>;
    fn signature(&self) -> Type;

    // What to put in Expression here? They are synthetic
    fn lambda_tree(&self, target: Identifier) -> Expression<BridgingInfo> {
        let info = BridgingInfo::new(target.clone());
        (0..self.arity()).rfold(Expression::CallBridge(info.clone(), target), |acc, x| {
            Expression::Lambda(
                info.clone(),
                Lambda {
                    parameter: Parameter::new(Identifier::new(&format!("p{x}"))),
                    body: acc.into(),
                },
            )
        })
    }
}

trait TypeBridge {
    const TYPE: Type;

    fn lift_type() -> Type {
        Self::TYPE
    }
}

impl TypeBridge for String {
    const TYPE: Type = Type::Base(BaseType::Text);
}

impl TypeBridge for () {
    const TYPE: Type = Type::Base(BaseType::Unit);
}

impl TypeBridge for i64 {
    const TYPE: Type = Type::Base(BaseType::Int);
}

impl TypeBridge for f64 {
    const TYPE: Type = Type::Base(BaseType::Float);
}

impl TypeBridge for bool {
    const TYPE: Type = Type::Base(BaseType::Bool);
}

pub struct Lambda1<A, R>(pub fn(A) -> R);

impl<A, R> Bridge for Lambda1<A, R>
where
    A: TryFrom<Value, Error = RuntimeError> + TypeBridge,
    R: Into<Value> + TypeBridge,
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
        Type::Function(A::lift_type().into(), R::lift_type().into())
    }
}

pub struct Lambda2<A, B, R>(pub fn(A, B) -> R);

impl<A, B, R> Bridge for Lambda2<A, B, R>
where
    A: TryFrom<Value, Error = RuntimeError> + TypeBridge,
    B: TryFrom<Value, Error = RuntimeError> + TypeBridge,
    R: Into<Value> + TypeBridge,
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
        Type::Function(
            A::lift_type().into(),
            Type::Function(B::lift_type().into(), R::lift_type().into()).into(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct RawLambda1<R>(pub fn(Value) -> R);

impl<R> Bridge for RawLambda1<R>
where
    R: TypeBridge + Into<Value>,
{
    fn arity(&self) -> usize {
        1
    }

    fn evaluate(&self, e: &Environment) -> CallResult<Value> {
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?;
        Ok(self.0.clone()(p0).into())
    }

    fn signature(&self) -> Type {
        let ty = TypeParameter::fresh();
        Type::Forall(
            ty.clone(),
            Type::Function(Type::Parameter(ty).into(), R::lift_type().into()).into(),
        )
    }
}

// See if I can do something with the Add, Sub, Mul and Div type classes
// for this thing.
// Or could I impl Bridge for all tripples that have Add?
#[derive(Debug, Clone)]
pub struct PartialRawLambda2 {
    pub apply: fn(Base, Base) -> Option<Base>,
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
            .map(Value::Base)
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
    CompileState {
        typing_context,
        interpreter_environment,
        ..
    }: &mut CompileState,
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

impl TryFrom<Value> for () {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Unit) = value {
            Ok(())
        } else {
            Err(RuntimeError::ExpectedType(Type::Base(BaseType::Unit)))
        }
    }
}

impl From<()> for Value {
    fn from(_value: ()) -> Self {
        Value::Base(Base::Unit)
    }
}

impl TryFrom<Value> for String {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Text(s)) = value {
            Ok(s)
        } else {
            Err(RuntimeError::ExpectedType(Type::Base(BaseType::Text)))
        }
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::Base(Base::Text(value))
    }
}

impl TryFrom<Value> for i64 {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Int(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Base(BaseType::Int)))
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Base(Base::Int(value))
    }
}

impl TryFrom<Value> for f64 {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Float(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Base(BaseType::Int)))
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Base(Base::Float(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn _open_file(x: i64) -> i64 {
        println!("open_file: {x}");
        1
    }

    // Bring this test back, but fix Lambda1::signature and Lambda2::signature first
    //    #[test]
    fn _playtime() {
        let _x = define(
            Identifier::new("open_file"),
            Lambda1(_open_file),
            &mut CompileState::default(),
        )
        .unwrap();
    }
}
