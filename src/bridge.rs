use std::fmt;

use crate::{
    ast::{Expression, Identifier, Lambda, Parameter},
    context::Linkage,
    interpreter::{Base, Environment, Interpretation, RuntimeError, Value},
    lexer::SourceLocation,
    parser::ParsingInfo,
    typer::{BaseType, Parsed, ProductType, TupleType, Type, TypeParameter, TypeScheme},
};

#[derive(Debug, Copy, Clone)]
pub struct BridgeInfo;

impl fmt::Display for BridgeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BridgeInfo")
    }
}

static BRIDGE_PARSING_INFO: ParsingInfo = ParsingInfo {
    position: SourceLocation { row: 0, column: 0 },
};

impl Parsed for BridgeInfo {
    fn info(&self) -> &ParsingInfo {
        &BRIDGE_PARSING_INFO
    }
}

pub trait Bridge {
    fn arity(&self) -> usize;
    fn evaluate(&self, e: &Environment) -> Interpretation;
    fn synthesize_type(&self) -> TypeScheme;

    fn generate_lambda_tree(&self, target: Identifier) -> Expression<BridgeInfo> {
        (0..self.arity()).rfold(
            Expression::InvokeBridge(BridgeInfo, target),
            |expression, parameter_index| {
                Expression::Lambda(
                    BridgeInfo,
                    Lambda {
                        parameter: Parameter::new(Identifier::new(&format!("p{parameter_index}"))),
                        body: expression.into(),
                    },
                )
            },
        )
    }
}

trait TypeBridge {
    const TYPE: Type;

    fn synthesize_type() -> Type {
        Self::TYPE
    }
}

impl TypeBridge for String {
    const TYPE: Type = Type::Constant(BaseType::Text);
}

impl TypeBridge for () {
    const TYPE: Type = Type::Constant(BaseType::Unit);
}

impl TypeBridge for i64 {
    const TYPE: Type = Type::Constant(BaseType::Int);
}

impl TypeBridge for f64 {
    const TYPE: Type = Type::Constant(BaseType::Float);
}

impl TypeBridge for bool {
    const TYPE: Type = Type::Constant(BaseType::Bool);
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

    fn evaluate(&self, e: &Environment) -> Interpretation {
        let Self(f) = self;
        Ok(f(e.lookup(&Identifier::new("p0")).cloned()?.try_into()?).into())
    }

    fn synthesize_type(&self) -> TypeScheme {
        TypeScheme::from_constant(Type::Arrow(
            A::synthesize_type().into(),
            R::synthesize_type().into(),
        ))
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

    fn evaluate(&self, e: &Environment) -> Interpretation {
        let Self(f) = self;
        Ok(f(
            e.lookup(&Identifier::new("p0")).cloned()?.try_into()?,
            e.lookup(&Identifier::new("p1")).cloned()?.try_into()?,
        )
        .into())
    }

    fn synthesize_type(&self) -> TypeScheme {
        TypeScheme::from_constant(Type::Arrow(
            A::synthesize_type().into(),
            Type::Arrow(B::synthesize_type().into(), R::synthesize_type().into()).into(),
        ))
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

    fn evaluate(&self, e: &Environment) -> Interpretation {
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?;
        Ok(self.0(p0).into())
    }

    fn synthesize_type(&self) -> TypeScheme {
        let alpha = TypeParameter::fresh();
        TypeScheme::new(
            &[alpha],
            Type::Arrow(Type::Parameter(alpha).into(), R::synthesize_type().into()),
        )
    }
}

pub struct TupleSyntax;

impl Bridge for TupleSyntax {
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, e: &Environment) -> Interpretation {
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?;
        let p1 = e.lookup(&Identifier::new("p1")).cloned()?;

        let tuple = if let Value::Tuple(mut xs) = p1 {
            xs.insert(0, p0);
            Value::Tuple(xs)
        } else {
            Value::Tuple(vec![p0, p1])
        };

        Ok(tuple)
    }

    fn synthesize_type(&self) -> TypeScheme {
        let p = TypeParameter::fresh();
        let q = TypeParameter::fresh();
        let body = Type::Arrow(
            Type::Parameter(p).into(),
            Type::Arrow(
                Type::Parameter(q).into(),
                Type::Product(ProductType::Tuple(TupleType(vec![
                    Type::Parameter(p),
                    Type::Parameter(q),
                ])))
                .into(),
            )
            .into(),
        );

        TypeScheme::new(&[p, q], body)
    }
}

// See if I can do something with the Add, Sub, Mul and Div type classes
// for this thing.
// Or could I impl Bridge for all tripples that have Add?
// Why is this Base only?
#[derive(Debug, Clone)]
pub struct PartialRawLambda2<F> {
    pub apply: F,
    pub signature: TypeScheme,
}

impl<F> Bridge for PartialRawLambda2<F>
where
    F: Clone + FnOnce(Value, Value) -> Option<Value>,
{
    fn arity(&self) -> usize {
        2
    }

    fn evaluate(&self, e: &Environment) -> Interpretation {
        // Is there any way to do this without the happy path clones?!
        let p0 = e.lookup(&Identifier::new("p0")).cloned()?;
        let p1 = e.lookup(&Identifier::new("p1")).cloned()?;

        self.apply.clone()(p0.clone(), p1.clone())
            .ok_or_else(|| RuntimeError::InapplicableLamda2 { fst: p0, snd: p1 }.into())
    }

    fn synthesize_type(&self) -> TypeScheme {
        self.signature.clone()
    }
}

pub fn define<B>(
    syntactical_name: Identifier,
    bridge: B,
    linkage: &mut Linkage,
) -> Interpretation<()>
where
    B: Bridge + 'static,
{
    let bridge_name = syntactical_name.mangle("bridge");

    linkage.bind_type(bridge_name.clone().into(), bridge.synthesize_type());
    linkage.bind_type(syntactical_name.clone().into(), bridge.synthesize_type());

    let lambda_expression = bridge.generate_lambda_tree(bridge_name.clone());
    linkage.bind_value(bridge_name, Value::bridge(bridge));

    let tree = lambda_expression.reduce(&mut linkage.interpreter_environment)?;
    linkage.bind_value(syntactical_name, tree);

    Ok(())
}

impl TryFrom<Value> for () {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if Value::Base(Base::Unit) == value {
            Ok(())
        } else {
            Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Unit)))
        }
    }
}

impl From<()> for Value {
    fn from(_value: ()) -> Self {
        Self::Base(Base::Unit)
    }
}

impl TryFrom<Value> for String {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Text(s)) = value {
            Ok(s)
        } else {
            Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Text)))
        }
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::Base(Base::Text(value))
    }
}

impl TryFrom<Value> for i64 {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Int(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Int)))
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
            Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Int)))
        }
    }
}

impl<A, B> From<(A, B)> for Value
where
    A: Into<Self>,
    B: Into<Self>,
{
    fn from((p, q): (A, B)) -> Self {
        Self::Tuple(vec![p.into(), q.into()])
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Base(Base::Float(value))
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Base(Base::Bool(value))
    }
}

impl TryFrom<Value> for bool {
    type Error = RuntimeError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        if let Value::Base(Base::Bool(x)) = value {
            Ok(x)
        } else {
            Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Bool)))
        }
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
            &mut Linkage::default(),
        )
        .unwrap();
    }
}
