use std::collections::VecDeque;

use crate::{
    ast::{Constant, ControlFlow, Expression, Identifier, Parameter},
    types::{TrivialType, Type},
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
    Bool(bool),
}

impl From<Constant> for TrivialValue {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(x) => Self::Int(x),
            Constant::Float(x) => Self::Float(x),
            Constant::Text(x) => Self::Text(x),
            Constant::Bool(x) => Self::Bool(x),
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

pub type InterpreterResult<A = Value> = Result<A, RuntimeError>;

impl Expression {
    pub fn reduce(self, env: &mut Environment) -> InterpreterResult {
        match self {
            Self::Variable(id) => env.lookup(&id).cloned(),
            Self::Literal(constant) => Ok(Value::Trivial(constant.into())),
            Self::Lambda { parameter, body } => reduce_lambda(parameter.name, *body, env),
            Self::Apply { function, argument } => reduce_apply(*function, *argument, env),
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
            } => reduce_binding(binder, *bound, *body, env),
            Self::Sequence { this, and_then } => {
                this.reduce(env)?;
                and_then.reduce(env)
            }
            Self::ControlFlow(control) => reduce_control_flow(control, env),
        }
    }
}

fn reduce_control_flow(control: ControlFlow, env: &mut Environment) -> InterpreterResult {
    match control {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            if let Value::Trivial(TrivialValue::Bool(test)) = predicate.reduce(env)? {
                if test {
                    consequent.reduce(env)
                } else {
                    alternate.reduce(env)
                }
            } else {
                Err(RuntimeError::ExpectedType(Type::Trivial(TrivialType::Bool)))
            }
        }
    }
}

fn reduce_binding(
    binder: Identifier,
    bound: Expression,
    body: Expression,
    env: &mut Environment,
) -> InterpreterResult {
    let bound = bound.reduce(env)?;
    env.insert_binding(binder, bound);
    // remove binder here, before returning?
    body.reduce(env)
}

fn reduce_apply(
    function: Expression,
    argument: Expression,
    env: &mut Environment,
) -> InterpreterResult {
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
        // Quantify a good function type here
        Err(RuntimeError::ExpectedFunction)
    }
}

fn reduce_lambda(param: Identifier, body: Expression, env: &mut Environment) -> InterpreterResult {
    Ok(Value::Closure {
        parameter: param,
        capture: env.clone(),
        body,
    })
}
