use std::{borrow::Cow, collections::VecDeque, rc::Rc};
use thiserror::Error;

use crate::{
    ast::{
        CompilationUnit, Constant, ControlFlow, Declaration, DependencyMatrix, Expression,
        Identifier, ModuleDeclarator, ValueDeclarator,
    },
    synthetics::SyntheticStub,
    types::{TrivialType, Type},
};

pub type Loaded<A> = Result<A, LoadError>;

// Todo: which ones are involved in the cycle or are unresolved?
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("Cyclic dependencies")]
    DependencyCycle,

    #[error("Unsatisfied dependencies")]
    UnsatisfiedDependencies,

    #[error("Runtime error initialzing the module")]
    InitializationError(#[from] RuntimeError),

    #[error("Dependency resolution failed")]
    DependencyResolutionFailed,
}

pub struct Interpreter {
    prelude: Environment,
}

impl Interpreter {
    pub fn new(prelude: Environment) -> Self {
        Self { prelude }
    }

    pub fn load_and_run(self, program: CompilationUnit) -> Loaded<Value> {
        match program {
            CompilationUnit::Implicit(module) => {
                // Typing has to happen for this to feel nice. TBD.
                match self.load_module(module)?.lookup(&Identifier::new("main"))? {
                    Value::Closure {
                        parameter,
                        capture,
                        body,
                    } => todo!(),
                    Value::SyntheticBridge { stub } => todo!(),
                    scalar => Ok(scalar.clone()),
                }
            }
            _otherwise => todo!(),
        }
    }

    fn load_module(self, module: ModuleDeclarator) -> Loaded<Environment> {
        ModuleLoader::try_initializing(&module, self.prelude)?.resolve_dependencies()
    }
}

struct ModuleLoader<'a> {
    module: &'a ModuleDeclarator,
    matrix: DependencyMatrix<'a>,
    resolved: Environment,
}

impl<'a> ModuleLoader<'a> {
    fn try_initializing(module: &'a ModuleDeclarator, global: Environment) -> Loaded<Self> {
        let matrix = module.dependency_matrix();

        if !matrix.is_wellformed() {
            if !matrix.is_acyclic() {
                Err(LoadError::DependencyCycle)
            } else if !matrix.is_satisfiable() {
                Err(LoadError::UnsatisfiedDependencies)
            } else {
                unreachable!()
            }
        } else {
            Ok(Self {
                module,
                matrix,
                resolved: global,
            })
        }
    }

    fn find_resolvable(&'a self) -> Option<&'a &'a Identifier> {
        self.matrix.find(|id| {
            self.matrix
                .dependencies(id)
                .unwrap_or_default()
                .iter()
                // Optimize this
                .all(|id| self.resolved.is_defined(id))
        })
    }

    fn is_resolved(&self) -> bool {
        // Optimize this
        self.matrix
            .satisfies(|dependency| self.resolved.is_defined(dependency))
    }

    fn try_resolve(&mut self, id: &Identifier) -> Loaded<()> {
        if let Some(Declaration::Value { declarator, .. }) = self.module.find_value_declaration(id)
        {
            // That this has to clone the Expressions is not ideal
            match declarator {
                ValueDeclarator::Constant(constant) => {
                    let env = &mut self.resolved;
                    let reduced = constant.initializer.clone().reduce(env)?;
                    env.insert_binding(id.clone(), reduced);
                }
                ValueDeclarator::Function(function) => {
                    let env = &mut self.resolved;
                    let tree = function.clone().into_lambda_tree().reduce(env)?;
                    env.insert_binding(id.clone(), tree);
                }
            }

            Ok(())
        } else {
            panic!("Attempt to resolve un-resolvable declaration {id}")
        }
    }

    fn resolve_dependencies(mut self) -> Loaded<Environment> {
        while !self.is_resolved() {
            if let Some(resolvable) = self.find_resolvable().cloned() {
                self.try_resolve(&resolvable.clone())?;
            } else {
                Err(LoadError::DependencyResolutionFailed)?
            }
        }

        Ok(self.resolved)
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

    pub fn is_defined(&self, id: &Identifier) -> bool {
        self.state.iter().any(|(defined, ..)| defined == id)
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
