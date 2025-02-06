use std::{fmt, marker::PhantomData};

use crate::{
    ast::Identifier,
    interpreter::{Environment, Interpretation, RuntimeError, Value},
    lexer::Operator,
    synthetics::{CallResult, Lambda2, SyntheticStub},
    types::Type,
};

// Make a generic version of this
struct OperatorBridge<A, F> {
    operator: Operator,
    closure: F,
    tag: PhantomData<A>,
}

impl<A, F> OperatorBridge<A, F>
where
    F: Clone + FnOnce(A, A) -> A + 'static,
{
    pub fn new(operator: Operator, closure: F) -> Self {
        Self {
            operator,
            closure,
            tag: PhantomData::default(),
        }
    }
}

impl<A, F> fmt::Debug for OperatorBridge<A, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Op(symbol: {:?}, ..)", self.operator)
    }
}

impl<A, F> SyntheticStub for OperatorBridge<A, F>
where
    A: Clone + 'static,
    F: Clone + FnOnce(A, A) -> A + 'static,
    A: TryFrom<Value, Error = RuntimeError>,
    A: Into<Value>,
{
    fn surface_binder(&self) -> Identifier {
        Identifier::new(&self.operator.function_identifier())
    }

    fn apply(&self, capture: &Environment) -> CallResult<Value> {
        self.apply2(capture)
    }

    fn signature(&self) -> Type {
        todo!()
    }
}

impl<A, F> Lambda2 for OperatorBridge<A, F>
where
    A: Clone + 'static,
    F: Clone + FnOnce(A, A) -> A + 'static,
    A: TryFrom<Value, Error = RuntimeError>,
    A: Into<Value>,
{
    type P0 = A;
    type P1 = A;
    type R = A;

    fn apply_inner(&self, p0: Self::P0, p1: Self::P1) -> Self::R {
        self.closure.clone()(p0, p1)
    }
}

pub fn define(env: &mut Environment) -> Interpretation<()> {
    // Could I polymorph this even more by traiting up the underlying operators?
    // A: Add, A: Mul, etc?
    OperatorBridge::<i64, _>::new(Operator::Plus, |p, q| p + q).define(env)?;
    OperatorBridge::<i64, _>::new(Operator::Minus, |p, q| p - q).define(env)?;
    OperatorBridge::<i64, _>::new(Operator::Times, |p, q| p * q).define(env)?;
    OperatorBridge::<i64, _>::new(Operator::Divides, |p, q| p / q).define(env)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Expression as E, Identifier},
        interpreter::{Environment, Scalar},
        stdlib,
    };

    #[test]
    fn plus() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(3),
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn minus() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("-")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(-1),
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn times() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("*")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(2),
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn divides() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("/")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(0),
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }
}
