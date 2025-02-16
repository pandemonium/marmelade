use std::{
    borrow::Cow,
    cell::{OnceCell, RefCell},
    collections::{HashMap, VecDeque},
    fmt,
    ops::Deref,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};
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

    #[error("Runtime error initialzing the module {0}")]
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
                let env = self.load_module(module)?;
                match env.lookup(&Identifier::new("main"))? {
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
            if self
                .try_resolve(&resolvable.clone())
                .inspect_err(|e| println!("resolve_dependencies: resolving {resolvable} {e}"))
                .is_err()
            {
                unresolved.push_front(resolvable);
            }
        }

        //        self.try_resolve(&Identifier::new("factorial"))?;
        //        self.try_resolve(&Identifier::new("main"))?;
        //
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
            ValueDeclarator::Function(function) => function.into_lambda_tree(id.clone()),
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
    Closure(Closure),
    RecursiveClosure(RecursiveClosure),
    Bridge { target: BridgeDebug },
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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Closure(closure) => writeln!(f, "closure {closure}"),
            Self::RecursiveClosure(closure) => writeln!(f, "closure {closure}"),
            Self::Bridge { target } => write!(f, "{target:?}"),
        }
    }
}

// Turn SelfReferentialLambda into this
#[derive(Debug, Clone)]
pub struct RecursiveClosure {
    pub name: Identifier,
    pub inner: Rc<RefCell<Closure>>,
}

impl fmt::Display for RecursiveClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { name, inner } = self;
        write!(f, "{name} -> {inner:?}")
    }
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub parameter: Identifier,
    pub capture: Environment,
    pub body: Expression,
}

impl fmt::Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            parameter,
            capture,
            body,
        } = self;
        writeln!(f, "\\{parameter}. ")?;
        write!(f, "{capture}")?;
        write!(f, "{body}")
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

#[derive(Debug, Clone, Default)]
pub struct Environment {
    enclosing: Option<Rc<Environment>>,
    leaf: Vec<(Identifier, Value)>,
}

impl Environment {
    pub fn make_child(self: Rc<Environment>) -> Self {
        Self {
            enclosing: self.into(),
            leaf: Vec::default(),
        }
    }

    pub fn insert_binding(&mut self, binder: Identifier, bound: Value) {
        self.leaf.push((binder, bound));
    }

    pub fn lookup(&self, id: &Identifier) -> Interpretation<&Value> {
        self.leaf
            .iter()
            .rev()
            .find_map(|(binder, bound)| (binder == id).then_some(bound))
            .map(Ok)
            .unwrap_or_else(|| {
                println!("looking: enclosing: {id}");
                self.enclosing.as_ref().map_or_else(
                    || Err(RuntimeError::UndefinedSymbol(id.clone())),
                    |env| env.lookup(id),
                )
            })
    }

    pub fn is_defined(&self, id: &Identifier) -> bool {
        self.lookup(id).is_ok()
    }

    pub fn symbols(&self) -> Vec<&Identifier> {
        let mut boofer = self
            .leaf
            .iter()
            .rev()
            .map(|(id, ..)| id)
            .collect::<Vec<_>>();

        if let Some(enclosing) = self.enclosing.as_ref() {
            boofer.extend(enclosing.symbols());
        }

        boofer
    }

    fn remove_binding(&mut self, binder: &Identifier) {
        if let Some(pos) = self.leaf.iter().rposition(|(b, _)| b == binder) {
            self.leaf.remove(pos);
        }
    }
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "     ")?;
        for (binder, binding) in self.leaf.iter().rev() {
            write!(f, "{binder} = {binding},")?;
        }

        if let Some(enclosing) = self.enclosing.as_ref() {
            writeln!(f, "")?;
            write!(f, "{enclosing}")?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Error)]
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
            Self::SelfReferential {
                name,
                parameter,
                body,
            } => make_recursive_closure(name, parameter.name, *body, env.clone()),
            Self::Lambda { parameter, body } => make_closure(parameter.name, *body, env.clone()),
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

fn make_recursive_closure(
    name: Identifier,
    parameter: Identifier,
    body: Expression,
    capture: Environment,
) -> Result<Value, RuntimeError> {
    let closure = Rc::new(RefCell::new(Closure {
        parameter,
        capture,
        body,
    }));

    closure.borrow_mut().capture.insert_binding(
        name.clone(),
        Value::RecursiveClosure(RecursiveClosure {
            name,
            inner: Rc::clone(&closure),
        }),
    );

    let closure = closure.borrow();
    Ok(Value::Closure(closure.clone()))
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
    env.insert_binding(binder.clone(), bound);
    let retval = body.reduce(env);
    env.remove_binding(&binder);
    retval
}

fn apply_function(
    function: Expression,
    argument: Expression,
    env: &mut Environment,
) -> Interpretation {
    match function.reduce(env)? {
        Value::Closure(Closure {
            parameter,
            mut capture,
            body,
        }) => {
            let binding = argument.reduce(env)?;
            capture.insert_binding(parameter.clone(), binding);

            let retval = body.reduce(&mut capture);
            capture.remove_binding(&parameter);
            retval
        }
        Value::RecursiveClosure(RecursiveClosure { name, inner }) => {
            let binding = argument.reduce(env)?;

            let mut inner = { inner.borrow_mut().clone() };
            let parameter = inner.parameter.clone();
            inner.capture.insert_binding(parameter.clone(), binding);

            inner.body.clone().reduce(&mut inner.capture)
        }
        _otherwise => Err(RuntimeError::ExpectedFunction),
    }
}

fn make_closure(param: Identifier, body: Expression, env: Environment) -> Interpretation {
    Ok(Value::Closure(Closure {
        parameter: param,
        capture: env,
        body,
    }))
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, ControlFlow, Expression, Identifier, Parameter},
        interpreter::{Environment, RuntimeError, Scalar, Value},
        lexer::Location,
        stdlib,
    };

    use super::Closure;

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

    fn make_fix() -> Expression {
        Expression::Apply {
            function: Box::new(Expression::Lambda {
                parameter: Parameter::new(Identifier::new("x")),
                body: Box::new(Expression::Apply {
                    function: Box::new(Expression::Variable(Identifier::new("x"))),
                    argument: Box::new(Expression::Variable(Identifier::new("x"))),
                }),
            }),
            argument: Box::new(Expression::Lambda {
                parameter: Parameter::new(Identifier::new("x")),
                body: Box::new(Expression::Apply {
                    function: Box::new(Expression::Variable(Identifier::new("x"))),
                    argument: Box::new(Expression::Variable(Identifier::new("x"))),
                }),
            }),
        }
    }

    fn make_fix_value(env: Environment) -> Value {
        Value::Closure(Closure {
            parameter: Identifier::new("f"),
            capture: env.clone(),
            body: Expression::Apply {
                function: Box::new(Expression::Lambda {
                    parameter: Parameter::new(Identifier::new("x")),
                    body: Box::new(Expression::Apply {
                        function: Box::new(Expression::Variable(Identifier::new("f"))),
                        argument: Box::new(Expression::Lambda {
                            parameter: Parameter::new(Identifier::new("y")),
                            body: Box::new(Expression::Apply {
                                function: Box::new(Expression::Variable(Identifier::new("x"))),
                                argument: Box::new(Expression::Variable(Identifier::new("x"))),
                            }),
                        }),
                    }),
                }),
                argument: Box::new(Expression::Lambda {
                    parameter: Parameter::new(Identifier::new("x")),
                    body: Box::new(Expression::Apply {
                        function: Box::new(Expression::Variable(Identifier::new("f"))),
                        argument: Box::new(Expression::Lambda {
                            parameter: Parameter::new(Identifier::new("y")),
                            body: Box::new(Expression::Apply {
                                function: Box::new(Expression::Variable(Identifier::new("x"))),
                                argument: Box::new(Expression::Variable(Identifier::new("x"))),
                            }),
                        }),
                    }),
                }),
            },
        })
    }

    #[test]
    fn eval_fix() {
        let factorial = Expression::Lambda {
            parameter: Parameter::new(Identifier::new("x")),
            body: Expression::ControlFlow(ControlFlow::If {
                predicate: Expression::Apply {
                    function: Expression::Variable(Identifier::new("==")).into(),
                    argument: Expression::Variable(Identifier::new("x")).into(),
                }
                .into(),
                consequent: Expression::Literal(Constant::Int(1)).into(),
                alternate: Expression::Binding {
                    postition: Location::default(), // Placeholder for actual location
                    binder: Identifier::new("xx"),
                    bound: Expression::Apply {
                        function: Expression::Apply {
                            function: Expression::Variable(Identifier::new("-")).into(),
                            argument: Expression::Variable(Identifier::new("x")).into(),
                        }
                        .into(),
                        argument: Expression::Literal(Constant::Int(1)).into(),
                    }
                    .into(),
                    body: Expression::Apply {
                        function: Expression::Apply {
                            function: Expression::Variable(Identifier::new("*")).into(),
                            argument: Expression::Variable(Identifier::new("x")).into(),
                        }
                        .into(),
                        argument: Expression::Apply {
                            function: Expression::Variable(Identifier::new("factorial")).into(),
                            argument: Expression::Variable(Identifier::new("xx")).into(),
                        }
                        .into(),
                    }
                    .into(),
                }
                .into(),
            })
            .into(),
        };
    }

    //    #[test]
    fn fixed_factorial() {
        let factorial = Expression::Apply {
            function: Expression::Variable(Identifier::new("fix")).into(),
            argument: Expression::Lambda {
                parameter: Parameter::new(Identifier::new("fact")),
                body: Expression::Lambda {
                    parameter: Parameter::new(Identifier::new("x")),
                    body: Expression::ControlFlow(ControlFlow::If {
                        predicate: Expression::Apply {
                            function: Expression::Apply {
                                function: Expression::Variable(Identifier::new("==")).into(),
                                argument: Expression::Variable(Identifier::new("x")).into(),
                            }
                            .into(),
                            argument: Expression::Literal(Constant::Int(0)).into(),
                        }
                        .into(),
                        consequent: Expression::Literal(Constant::Int(1)).into(),
                        alternate: Expression::Binding {
                            postition: Location::default(), // Placeholder for actual location
                            binder: Identifier::new("xx"),
                            bound: Expression::Apply {
                                function: Expression::Apply {
                                    function: Expression::Variable(Identifier::new("-")).into(),
                                    argument: Expression::Variable(Identifier::new("x")).into(),
                                }
                                .into(),
                                argument: Expression::Literal(Constant::Int(1)).into(),
                            }
                            .into(),
                            body: Expression::Apply {
                                function: Expression::Apply {
                                    function: Expression::Variable(Identifier::new("*")).into(),
                                    argument: Expression::Variable(Identifier::new("x")).into(),
                                }
                                .into(),
                                argument: Expression::Apply {
                                    function: Expression::Variable(Identifier::new("fact")).into(),
                                    argument: Expression::Variable(Identifier::new("xx")).into(),
                                }
                                .into(),
                            }
                            .into(),
                        }
                        .into(),
                    })
                    .into(),
                }
                .into(),
            }
            .into(),
        };

        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

        env.insert_binding(
            Identifier::new("fix"),
            make_fix_value(Environment::default()),
        );

        let reduced_fact = factorial.reduce(&mut env).unwrap();
        env.insert_binding(Identifier::new("factorial"), reduced_fact);

        let e = Expression::Apply {
            function: Expression::Variable(Identifier::new("factorial")).into(),
            argument: Expression::Literal(Constant::Int(1)).into(),
        };

        assert_eq!(
            Scalar::Int(127),
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }
}
