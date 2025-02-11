use std::{borrow::Cow, collections::VecDeque, fmt, rc::Rc};
use thiserror::Error;

use crate::{
    ast::{
        CompilationUnit, Constant, ControlFlow, Declaration, DependencyMatrix, Expression,
        Identifier, ModuleDeclarator, ValueDeclarator,
    },
    bridge::Bridge,
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
                    Value::Closure { .. } => todo!(),
                    Value::Bridge { .. } => todo!(),
                    scalar => Ok(scalar.clone()),
                }
            }
            _otherwise => todo!(),
        }
    }

    fn load_module(self, module: ModuleDeclarator) -> Loaded<Environment> {
        ModuleLoader::try_initializing(&module, self.prelude)?.resolve_dependencies()
    }

    fn _patch_with_prelude(
        &self,
        module: ModuleDeclarator,
        _prelude: &Environment,
    ) -> ModuleDeclarator {
        println!("patch_with_prelude: Not patching!!!");
        //        module.declarations.push(Declaration::ImportModule {
        //            position: Location::default(),
        //            exported_symbols: prelude.symbols().into_iter().cloned().collect::<Vec<_>>(),
        //        });
        module
    }
}

struct ModuleLoader<'a> {
    module: &'a ModuleDeclarator,
    matrix: DependencyMatrix<'a>,
    resolved: Environment,
}

impl<'a> ModuleLoader<'a> {
    fn try_initializing(module: &'a ModuleDeclarator, prelude: Environment) -> Loaded<Self> {
        let resolver = |id: &Identifier| prelude.is_defined(id);

        let matrix = module.dependency_matrix();
        if !matrix.is_wellformed(resolver) {
            if !matrix.is_acyclic() {
                Err(LoadError::DependencyCycle)
            } else if !matrix.is_satisfiable(resolver) {
                Err(LoadError::UnsatisfiedDependencies)
            } else {
                unreachable!()
            }
        } else {
            Ok(Self {
                module,
                matrix,
                resolved: prelude,
            })
        }
    }

    fn resolve_dependencies(mut self) -> Loaded<Environment> {
        let mut unresolved = self.matrix.nodes().drain(..).collect::<VecDeque<_>>();

        while let Some(resolvable) = unresolved.pop_back() {
            if self.try_resolve(&resolvable.clone()).is_err() {
                unresolved.push_front(resolvable);
            }
        }

        Ok(self.resolved)
    }

    fn try_resolve(&mut self, id: &Identifier) -> Loaded<()> {
        if let Some(Declaration::Value { declarator, .. }) = self.module.find_value_declaration(id)
        {
            self.resolve_value_binding(id, declarator)
        } else {
            panic!("Unable to resolve declaration: `{id}` - not implemented")
        }
    }

    fn resolve_value_binding(
        &mut self,
        id: &Identifier,
        declarator: &ValueDeclarator,
    ) -> Result<(), LoadError> {
        // That this has to clone the Expressions is not ideal

        let expression = match declarator.clone() {
            ValueDeclarator::Constant(constant) => constant.initializer,
            ValueDeclarator::Function(function) => function.into_lambda_tree(),
        };

        let env = &mut self.resolved;
        let value = expression.reduce(env)?;
        env.insert_binding(id.clone(), value);

        Ok(())
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
    Bridge {
        target: BridgeDebug,
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

    pub fn bridge<B>(bridge: B) -> Self
    where
        B: Bridge + 'static,
    {
        Self::Bridge {
            target: BridgeDebug(Rc::new(bridge)),
        }
    }
}

#[derive(Clone)]
pub struct BridgeDebug(Rc<dyn Bridge + 'static>);

impl fmt::Debug for BridgeDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(b) = self;
        // Could display the type here too, I guess
        write!(f, "Bridge(Lamda{}(..))", b.arity())
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

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::Text(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{x}"),
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

    pub fn symbols(&self) -> Vec<&Identifier> {
        self.state.iter().map(|(id, ..)| id).collect::<Vec<_>>()
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

    #[error("Not applicable")]
    InapplicableLamda2,
}

pub type Interpretation<A = Value> = Result<A, RuntimeError>;

impl Expression {
    pub fn reduce(self, env: &mut Environment) -> Interpretation {
        match self {
            Self::Variable(id) => env.lookup(&id).cloned(),
            Self::CallBridge(id) => evaluate_bridge(id, env),
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

fn evaluate_bridge(id: Identifier, env: &mut Environment) -> Interpretation {
    if let Value::Bridge {
        // Do away with this sucker. Impl Deref.
        target: BridgeDebug(bridge),
    } = env.lookup(&id)?
    {
        bridge.evaluate(env)
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
        stdlib::import(&mut env).unwrap();

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
        stdlib::import(&mut env).unwrap();

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
