use std::collections::VecDeque;

use crate::{
    ast::{Constant, Expression, Identifier, Parameter},
    types::Type,
};

#[derive(Debug, Clone)]
pub enum Value {
    Trivial(TrivialValue),
    Closure {
        parameter: Identifier,
        capture: Environment,
        body: Expression,
    },
}

#[derive(Debug, Clone)]
pub enum TrivialValue {
    Int(i64),
    Float(f64),
    Text(String),
}

impl From<Constant> for TrivialValue {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(x) => Self::Int(x),
            Constant::Float(x) => Self::Float(x),
            Constant::Text(x) => Self::Text(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Environment {
    state: VecDeque<(Identifier, Value)>,
}

impl Environment {
    pub fn insert_binding(&mut self, binder: Identifier, value: Value) {
        self.state.push_front((binder, value));
    }

    pub fn lookup(&self, id: &Identifier) -> InterpreterResult<&Value> {
        self.state
            .iter()
            .find_map(|(binder, value)| (binder == id).then_some(value))
            .ok_or_else(|| RuntimeError::UndefinedSymbol(id.clone()))
    }
}

pub enum RuntimeError {
    UndefinedSymbol(Identifier),
    ExpectedType(Type),
    ExpectedFunction,
}

pub type InterpreterResult<A> = Result<A, RuntimeError>;

impl Expression {
    pub fn reduce(self, env: &mut Environment) -> InterpreterResult<Value> {
        match self {
            Self::Variable(id) => env.lookup(&id).cloned(),
            Self::Literal(constant) => Ok(Value::Trivial(constant.into())),
            Self::Lambda { parameter, body } => Ok(Value::Closure {
                parameter: parameter.name,
                capture: env.clone(),
                body: *body,
            }),
            Self::Apply { function, argument } => {
                // How do I solve builtins and that stuff?
                // Introduce ApplySpecial for the builtin stuff?
                if let Value::Closure {
                    parameter,
                    mut capture,
                    body,
                } = function.reduce(env)?
                {
                    let binding = argument.reduce(env)?;
                    capture.insert_binding(parameter, binding);
                    body.reduce(&mut capture)
                } else {
                    /* Some typing information would be nice */
                    Err(RuntimeError::ExpectedFunction)
                }
            }
            Self::Construct {
                name,
                constructor,
                argument,
            } => todo!(),
            Self::Product(product) => todo!(),
            Self::Project { base, index } => todo!(),
            Self::Binding {
                binder,
                bound,
                body,
                ..
            } => {
                let bound = bound.reduce(env)?;
                env.insert_binding(binder, bound);
                body.reduce(env)
                // remove binder here, before returning?
            }
            Self::Sequence { this, and_then } => {
                this.reduce(env)?;
                and_then.reduce(env)
            }
            Self::ControlFlow(control_flow) => todo!(),
        }
    }
}
