use std::fmt;

use crate::{
    ast::{Expression, Identifier, Parameter},
    interpreter::{Environment, Interpretation, RuntimeError, Scalar, Value},
    types::{TrivialType, Type},
};

pub type CallResult<A> = Result<A, RuntimeError>;

pub trait SyntheticStub
where
    Self: fmt::Debug,
{
    fn surface_binder(&self) -> Identifier;

    fn stub_binder(&self) -> Identifier {
        self.surface_binder().scoped_with("synthetic")
    }

    fn apply(&self, capture: &Environment) -> CallResult<Value>;

    fn signature(&self) -> Type;
}

// Does this have to be a trait? I don't think it does
pub trait Lambda2: SyntheticStub + Sized + 'static {
    type P0: TryFrom<Value, Error = RuntimeError>;
    type P1: TryFrom<Value, Error = RuntimeError>;
    type R: Into<Value>;

    fn apply2(&self, capture: &Environment) -> CallResult<Value> {
        Ok(self
            .apply_inner(
                capture
                    .lookup(&Identifier::new("p0"))
                    .cloned()?
                    .try_into()?,
                capture
                    .lookup(&Identifier::new("p1"))
                    .cloned()?
                    .try_into()?,
            )
            .into())
    }

    fn apply_inner(&self, p0: Self::P0, p1: Self::P1) -> Self::R;

    fn define(self, env: &mut Environment) -> Interpretation<()> {
        let stub_binder = self.stub_binder();
        let surface_binder = self.surface_binder();

        env.insert_binding(stub_binder.clone(), Value::synthetic_stub(self));
        let tree = synthesize_lambda2_tree(&stub_binder).reduce(env)?;
        env.insert_binding(surface_binder, tree);

        Ok(())
    }
}

fn synthesize_lambda2_tree(stub_binder: &Identifier) -> Expression {
    Expression::Lambda {
        parameter: Parameter::new(Identifier::new("p0")),
        body: Expression::Lambda {
            parameter: Parameter::new(Identifier::new("p1")),
            body: Expression::InvokeSynthetic(stub_binder.clone()).into(),
        }
        .into(),
    }
}
