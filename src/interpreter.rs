use std::{borrow::Cow, collections::VecDeque, rc::Rc};

use thiserror::Error;

use crate::{
    ast::{
        CompilationUnit, Constant, ControlFlow, Declaration, Expression, Identifier,
        ModuleDeclarator, ValueDeclarator,
    },
    synthetics::SyntheticStub,
    types::{TrivialType, Type},
};

pub struct Program {
    environment: Environment,
    entry_point: Expression,
}

struct ProgamLoader;

impl ProgamLoader {
    pub fn load(self, program: CompilationUnit, prelude: Environment) -> Program {
        match program {
            CompilationUnit::Implicit(module) => self.load_module(module, prelude),
            _otherwise => todo!(),
        }
    }

    fn load_module(self, module: ModuleDeclarator, mut env: Environment) -> Program {
        let matrix = module.dependency_matrix();

        let is_defined = |id: &Identifier| env.lookup(id).is_ok();

        // process declarations starting at those whose dependencies
        // are already in the environment

        //        module.declarations

        for decl in module.declarations {
            match decl {
                Declaration::Value {
                    binder,
                    declarator: ValueDeclarator::Constant(constant),
                    ..
                } => (),
                Declaration::Value {
                    binder,
                    declarator: ValueDeclarator::Function(function),
                    ..
                } => (),
                _otherwise => todo!(),
            }
        }

        Program {
            environment: env,
            entry_point: todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Scalar(Scalar),
    Closure {
        parameter: Identifier,
        capture: Environment,
        body: Expression,
    },
    SyntheticBridge {
        stub: Rc<dyn SyntheticStub>,
    },
}

impl Value {
    pub fn try_into_scalar(self) -> Option<Scalar> {
        if let Self::Scalar(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn synthetic_stub<S>(stub: S) -> Self
    where
        S: SyntheticStub + 'static,
    {
        Self::SyntheticBridge {
            stub: Rc::new(stub),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl From<Constant> for Scalar {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(x) => Self::Int(x),
            Constant::Float(x) => Self::Float(x),
            Constant::Text(x) => Self::Text(x),
            Constant::Bool(x) => Self::Bool(x),
        }
    }
}

// Is there a Rust-blessed persistent Cons-list such that
// I can share the tail? Will a Cow do here? Cow<VecDeque> ?
#[derive(Debug, Clone, Default)]
pub struct Environment {
    state: Cow<'static, VecDeque<(Identifier, Value)>>,
}

impl Environment {
    pub fn insert_binding(&mut self, binder: Identifier, bound: Value) {
        self.state.to_mut().push_front((binder, bound));
    }

    pub fn lookup(&self, id: &Identifier) -> Interpretation<&Value> {
        self.state
            .iter()
            .find_map(|(binder, bound)| (binder == id).then_some(bound))
            .ok_or_else(|| RuntimeError::UndefinedSymbol(id.clone()))
    }
}

#[derive(Debug, PartialEq, Error)]
pub enum RuntimeError {
    #[error("Undefined symbol {0}")]
    UndefinedSymbol(Identifier),

    #[error("Expected type {0}")]
    ExpectedType(Type),

    #[error("Expected a function type")]
    ExpectedFunction,

    #[error("Expected a synthetic closure {0}")]
    ExpectedSynthetic(Identifier),
}

pub type Interpretation<A = Value> = Result<A, RuntimeError>;

impl Expression {
    pub fn reduce(self, env: &mut Environment) -> Interpretation {
        match self {
            Self::Variable(id) => env.lookup(&id).cloned(),
            Self::InvokeSynthetic(id) => evaluate_synthetic(id, env),
            Self::Literal(constant) => immediate(constant),
            Self::Lambda { parameter, body } => close_over_environment(parameter.name, *body, env),
            Self::Apply { function, argument } => apply_function(*function, *argument, env),
            Self::Construct { .. } => todo!(),
            Self::Product(..) => todo!(),
            Self::Project { .. } => todo!(),
            Self::Binding {
                binder,
                bound,
                body,
                ..
            } => reduce_binding(binder, *bound, *body, env),
            Self::Sequence { this, and_then } => sequence(this, and_then, env),
            Self::ControlFlow(control) => reduce_control_flow(control, env),
        }
    }
}

fn immediate(constant: Constant) -> Interpretation {
    Ok(Value::Scalar(constant.into()))
}

fn sequence(
    this: Box<Expression>,
    and_then: Box<Expression>,
    env: &mut Environment,
) -> Interpretation {
    this.reduce(env)?;
    and_then.reduce(env)
}

fn evaluate_synthetic(id: Identifier, env: &mut Environment) -> Interpretation {
    if let Value::SyntheticBridge { stub } = env.lookup(&id)? {
        stub.clone().apply(env)
    } else {
        Err(RuntimeError::ExpectedSynthetic(id))
    }
}

fn reduce_control_flow(control: ControlFlow, env: &mut Environment) -> Interpretation {
    match control {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            if let Value::Scalar(Scalar::Bool(test)) = predicate.reduce(env)? {
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
) -> Interpretation {
    let bound = bound.reduce(env)?;
    env.insert_binding(binder, bound);
    // remove binder here, before returning?
    body.reduce(env)
}

fn apply_function(
    function: Expression,
    argument: Expression,
    env: &mut Environment,
) -> Interpretation {
    match function.reduce(env)? {
        Value::Closure {
            parameter,
            mut capture,
            body,
        } => {
            let binding = argument.reduce(env)?;
            capture.insert_binding(parameter, binding);
            body.reduce(&mut capture)
        }
        _otherwise => Err(RuntimeError::ExpectedFunction),
    }
}

fn close_over_environment(
    param: Identifier,
    body: Expression,
    env: &mut Environment,
) -> Interpretation {
    Ok(Value::Closure {
        parameter: param,
        capture: env.clone(),
        body,
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Expression, Identifier},
        interpreter::{Environment, RuntimeError, Scalar, Value},
        stdlib,
    };

    #[test]
    fn reduce_literal() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        assert_eq!(
            Scalar::Int(1),
            Expression::Literal(Constant::Int(1))
                .reduce(&mut env)
                .unwrap()
                .try_into_scalar()
                .unwrap(),
        );
    }

    #[test]
    fn reduce_with_variables() {
        let mut env = Environment::default();
        stdlib::define(&mut env).unwrap();

        env.insert_binding(Identifier::new("x"), Value::Scalar(Scalar::Int(1)));

        assert_eq!(
            Scalar::Int(1),
            Expression::Variable(Identifier::new("x"))
                .reduce(&mut env)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );

        assert_eq!(
            RuntimeError::UndefinedSymbol(Identifier::new("y")),
            Expression::Variable(Identifier::new("y"))
                .reduce(&mut env)
                .unwrap_err()
        )
    }
}
