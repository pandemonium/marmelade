use std::{any::Any, cell::RefCell, collections::HashMap, fmt, rc::Rc};
use thiserror::Error;

use crate::{
    ast::{
        Apply, Binding, CompilationUnit, Constant, ControlFlow, Declaration, DeconstructInto,
        Expression, Identifier, ImportModule, Inject, Lambda, MatchClause, ModuleDeclarator,
        Pattern, Product, ProductIndex, Project, SelfReferential, Sequence, StructPattern,
        TypeDeclaration, TypeDeclarator, TypeName,
    },
    bridge::Bridge,
    interpreter::module::ModuleResolver,
    parser::ParseError,
    typer::{BaseType, Parsed, Type, TypeChecker, TypeError, Typing, TypingContext},
};

mod module;

pub use module::DependencyGraph;

pub type Resolved<A> = Result<A, ResolutionError>;

// Todo: which ones are involved in the cycle or are unresolved?
#[derive(Debug, Error)]
pub enum ResolutionError {
    #[error("Cyclic dependencies")]
    DependencyCycle,

    #[error("Unsatisfied dependencies")]
    UnsatisfiedDependencies,

    #[error("Runtime error initialzing the module {0}")]
    InitializationError(#[from] RuntimeError),

    #[error("Dependency resolution failed")]
    DependencyResolutionFailed,

    #[error("Type error {0}")]
    TypeError(#[from] TypeError),

    #[error("Parse error {0}")]
    ParseError(#[from] ParseError),
}

pub struct Interpreter {
    prelude: Environment,
}

impl Interpreter {
    pub fn new(prelude: Environment) -> Self {
        Self { prelude }
    }

    pub fn load_and_run<A>(
        self,
        typing_context: TypingContext,
        program: CompilationUnit<A>,
    ) -> Resolved<Value>
    where
        A: Clone + Parsed + fmt::Debug + fmt::Display,
    {
        match program {
            CompilationUnit::Implicit(annotation, module) => {
                let env = self.load_module(&annotation, typing_context, module)?;

                match env.lookup(&Identifier::new("main"))? {
                    Value::Closure { .. } => todo!(),
                    Value::Bridge { .. } => todo!(),
                    scalar => Ok(scalar.clone()),
                }
            }
            _otherwise => todo!(),
        }
    }

    fn load_module<A>(
        self,
        annotation: &A,
        mut typing_context: TypingContext,
        mut module: ModuleDeclarator<A>,
    ) -> Resolved<Environment>
    where
        A: Clone + Parsed + fmt::Debug + fmt::Display,
    {
        self.inject_prelude(annotation, &mut module);
        self.inject_types_and_synthetics(annotation, &mut module, &mut typing_context)?;

        let type_checker = TypeChecker::new(typing_context);

        ModuleResolver::initialize(&module, self.prelude)?
            .type_check(type_checker)?
            .resolve()
    }

    fn inject_prelude<A>(&self, annotation: &A, module: &mut ModuleDeclarator<A>)
    where
        A: Clone,
    {
        println!("Loading prelude...");
        module.declarations.push(Declaration::ImportModule(
            annotation.clone(),
            ImportModule {
                exported_symbols: self
                    .prelude
                    .symbols()
                    .drain(..)
                    .cloned()
                    .collect::<Vec<_>>(),
            },
        ));
    }

    fn inject_types_and_synthetics<A>(
        &self,
        annotation: &A,
        module: &mut ModuleDeclarator<A>,
        typing_context: &mut TypingContext,
    ) -> Typing<()>
    where
        A: fmt::Display + fmt::Debug + Clone + Parsed,
    {
        println!("Loading types and synthesizing constructors ...");
        let mut injections = vec![];
        for decl in &module.declarations {
            if let Declaration::Type(
                _,
                TypeDeclaration {
                    binding,
                    declarator,
                },
            ) = decl
            {
                match declarator {
                    TypeDeclarator::Coproduct(_, coproduct) => {
                        let coproduct = coproduct.make_implementation_module(
                            annotation,
                            TypeName::new(&binding.as_str()),
                            typing_context,
                        )?;
                        println!(
                            "inject_types_and_synthetics: `{binding}` `{}`",
                            coproduct.type_constructor
                        );
                        typing_context.bind(coproduct.name.into(), coproduct.type_constructor);
                        for constructor in coproduct.constructors {
                            injections.push(Declaration::Value(annotation.clone(), constructor));
                        }
                    }
                    TypeDeclarator::Struct(_, record) => {
                        let scheme = record.synthesize_type(typing_context)?;
                        println!("inject_types_and_synthetics: `{binding}` `{scheme}`");
                        typing_context.bind(TypeName::new(&binding.as_str()).into(), scheme);
                    }
                    _otherwise => todo!(),
                }
            }
        }

        module.declarations.extend(injections);

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Base(Base),
    Coproduct {
        name: TypeName,
        constructor: Identifier,
        value: Box<Value>,
    },
    Tuple(Vec<Value>),
    Struct(Vec<(Identifier, Value)>),
    Closure(Closure),
    RecursiveClosure(RecursiveClosure),
    Bridge {
        target: BridgeDebug,
    },
}

impl Value {
    pub fn try_into_base_type(self) -> Option<Base> {
        if let Self::Base(s) = self {
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
            Self::Base(scalar) => write!(f, "{scalar}"),
            Self::Coproduct {
                name,
                constructor,
                value,
            } => write!(f, "{name}::{constructor} {value}"),
            Self::Tuple(elements) => write!(
                f,
                "{}",
                elements
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Struct(fields) => write!(
                f,
                "{}",
                fields
                    .iter()
                    .map(|(field, value)| format!("{field}: {value}"))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Closure(closure) => writeln!(f, "closure {closure:?}"),
            Self::RecursiveClosure(closure) => writeln!(f, "closure {closure:?}"),
            Self::Bridge { target } => write!(f, "{target:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveClosure {
    pub name: Identifier, // Name does not seem used.
    pub inner: Rc<RefCell<Closure>>,
}

impl fmt::Display for RecursiveClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { name, inner } = self;
        let inner = inner.borrow();
        write!(f, "{name} -> {inner}")
    }
}

#[derive(Clone, PartialEq)]
pub struct Closure {
    pub parameter: Identifier,
    pub capture: Environment,
    pub body: Expression<()>,
}

impl fmt::Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Closure")
            .field("parameter", &self.parameter)
            .field("body", &self.body)
            .finish()
    }
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
        write!(f, "{body:?}")
    }
}

#[derive(Clone)]
pub struct BridgeDebug(Rc<dyn Bridge + 'static>);

impl PartialEq for BridgeDebug {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id()
    }
}

impl fmt::Debug for BridgeDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(b) = self;
        // Could display the type here too, I guess
        write!(f, "Bridge(Lamda{}(..))", b.arity())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Base {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Unit,
}

impl From<Constant> for Base {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(x) => Self::Int(x),
            Constant::Float(x) => Self::Float(x),
            Constant::Text(x) => Self::Text(x),
            Constant::Bool(x) => Self::Bool(x),
            Constant::Unit => Self::Unit,
        }
    }
}

impl fmt::Display for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::Text(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{x}"),
            Self::Unit => write!(f, "()"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Environment {
    parent: Option<Rc<Environment>>,
    leaf: Vec<(Identifier, Value)>,
}

impl Environment {
    pub fn into_parent(self: Environment) -> Self {
        Self {
            parent: Rc::new(self).into(),
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
                self.parent.as_ref().map_or_else(
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

        if let Some(enclosing) = self.parent.as_ref() {
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

        if let Some(enclosing) = self.parent.as_ref() {
            writeln!(f)?;
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

    #[error("Expected a function type, got: {got}")]
    ExpectedFunction { got: Value },

    #[error("Expected a synthetic closure {0}")]
    ExpectedSynthetic(Identifier),

    #[error("Function not applicable to {fst} and {snd}")]
    InapplicableLamda2 { fst: Value, snd: Value },

    #[error("{scrutinee} did not match any case")]
    ExpectedMatch { scrutinee: Value },

    #[error("Expected struct field {0}")]
    ExpectedProductIndex(ProductIndex),

    #[error("Must not project {lhs} with {rhs}")]
    BadProjection { lhs: Value, rhs: ProductIndex },
}

pub type Interpretation<A = Value> = Result<A, RuntimeError>;

impl<A> Expression<A>
where
    A: fmt::Debug + Clone,
{
    // I would like to remove mut here. This means that reduce_binding has to stop
    // in place-mutating this environment, to instead clone. But I want a smart clone
    // here. Environment::with_binding(<<closure>>)?
    pub fn reduce(self, env: &mut Environment) -> Interpretation {
        match self {
            Self::Variable(_, id) => env.lookup(&id).cloned(),
            Self::InvokeBridge(_, id) => invoke_bridge(id, env),
            Self::Literal(_, constant) => Ok(reduce_immediate(constant)),
            Self::SelfReferential(
                _,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => make_recursive_closure(name, parameter.name, body.erase_annotation(), env.clone()),
            Self::Lambda(_, Lambda { parameter, body }) => {
                make_closure(parameter.name, *body, env.clone())
            }
            Self::Apply(_, Apply { function, argument }) => {
                apply_function(*function, *argument, env)
            }
            Self::Inject(_, inject) => reduce_inject_coproduct(inject, env),
            Self::Product(_, node) => reduce_product(node, env),
            Self::Project(_, Project { base, index }) => reduce_projection(*base, index, env),
            Self::Binding(
                _,
                Binding {
                    binder,
                    bound,
                    body,
                    ..
                },
            ) => reduce_binding(binder, *bound, *body, env),
            Self::Sequence(_, Sequence { this, and_then }) => {
                reduce_sequence(*this, *and_then, env)
            }
            Self::ControlFlow(_, control) => reduce_control_flow(control, env),
            Self::DeconstructInto(_, deconstruct) => reduce_deconstruction(deconstruct, env),
        }
    }
}

fn reduce_projection<A>(
    base: Expression<A>,
    index: ProductIndex,
    env: &mut Environment,
) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    let base = base.reduce(env)?;
    match (base, index) {
        (Value::Struct(elements), ref index @ ProductIndex::Struct(ref rhs)) => elements
            .iter()
            .find_map(|(lhs, element)| (lhs == rhs).then_some(element))
            .cloned()
            .ok_or_else(|| RuntimeError::ExpectedProductIndex(index.clone())),
        (Value::Tuple(elements), index @ ProductIndex::Tuple(rhs)) => elements
            .get(rhs)
            .cloned()
            .ok_or_else(|| RuntimeError::ExpectedProductIndex(index)),
        (lhs, rhs) => Err(RuntimeError::BadProjection { lhs, rhs }),
    }
}

fn reduce_deconstruction<A>(
    DeconstructInto {
        scrutinee,
        match_clauses,
    }: DeconstructInto<A>,
    env: &mut Environment,
) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    let scrutinee = scrutinee.reduce(env)?;

    match_clauses
        .into_iter()
        .find_map(|clause| clause.match_with(&scrutinee))
        .ok_or_else(|| RuntimeError::ExpectedMatch { scrutinee })
        .and_then(|matched| reduce_consequent(matched, env))
}

fn reduce_consequent<A>(matched: Match<A>, env: &mut Environment) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    let mut env = env.clone();
    for (binding, value) in matched.bindings {
        env.insert_binding(binding, value);
    }

    matched.consequent.reduce(&mut env)
}

pub struct Match<A> {
    pub bindings: Vec<(Identifier, Value)>,
    pub consequent: Expression<A>,
}

impl<A> MatchClause<A> {
    pub fn match_with(self, scrutinee: &Value) -> Option<Match<A>>
    where
        A: fmt::Debug + Clone,
    {
        Some(Match {
            bindings: extract_matched_bindings(scrutinee, self.pattern)?,
            consequent: *self.consequent,
        })
    }
}

fn extract_matched_bindings<A>(
    scrutinee: &Value,
    pattern: Pattern<A>,
) -> Option<Vec<(Identifier, Value)>>
where
    A: fmt::Debug,
{
    match (scrutinee, pattern) {
        (
            Value::Coproduct {
                constructor, value, ..
            },
            Pattern::Coproduct(annotation, pattern),
        ) if constructor == &pattern.constructor => {
            extract_matched_bindings(value, Pattern::Tuple(annotation, pattern.argument))
        }
        (Value::Tuple(elements), Pattern::Tuple(_, pattern))
            if elements.len() == pattern.elements.len() =>
        {
            let mut bindings = Vec::with_capacity(elements.len());
            for (value, pattern) in elements.iter().zip(pattern.elements) {
                bindings.extend(extract_matched_bindings(value, pattern)?);
            }

            Some(bindings)
        }
        (Value::Struct(field_values), Pattern::Struct(_, StructPattern { fields }))
            if fields.len() == field_values.len() =>
        {
            let field_values = field_values
                .iter()
                .map(|(field, value)| (field, value))
                .collect::<HashMap<_, _>>();

            let mut bindings = Vec::with_capacity(fields.len());
            for (field, pattern) in fields {
                let value = field_values.get(&field)?;
                bindings.extend(extract_matched_bindings(value, pattern)?);
            }

            Some(bindings)
        }
        (_, Pattern::Literally(this)) => (scrutinee == &reduce_immediate(this)).then(Vec::new),
        (_, Pattern::Otherwise(binding)) => Some(vec![(binding, scrutinee.clone())]),
        _otherwise => None,
    }
}

impl<A> Pattern<A> {}

fn reduce_product<A>(node: Product<A>, env: &mut Environment) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    match node {
        Product::Tuple(mut elements) => Ok(Value::Tuple(
            // flatten here?
            // could mean more than one flattenings.
            elements
                .drain(..)
                .map(|e| e.reduce(env))
                .collect::<Interpretation<_>>()?,
        )),
        Product::Struct(mut bindings) => Ok(Value::Struct(
            bindings
                .drain(..)
                .map(|(field, expr)| expr.reduce(env).map(|v| (field, v)))
                .collect::<Interpretation<_>>()?,
        )),
    }
}

fn reduce_inject_coproduct<A>(node: Inject<A>, env: &mut Environment) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    Ok(Value::Coproduct {
        constructor: node.constructor,
        value: node.argument.reduce(env)?.into(),
        name: node.name.clone(),
    })
}

fn make_recursive_closure(
    name: Identifier,
    parameter: Identifier,
    body: Expression<()>,
    capture: Environment,
) -> Interpretation {
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

fn reduce_immediate(constant: Constant) -> Value {
    Value::Base(constant.into())
}

fn reduce_sequence<A>(
    this: Expression<A>,
    and_then: Expression<A>,
    env: &mut Environment,
) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    this.reduce(env)?;
    and_then.reduce(env)
}

fn invoke_bridge(id: Identifier, env: &mut Environment) -> Interpretation {
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

fn reduce_control_flow<A>(control: ControlFlow<A>, env: &mut Environment) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    match control {
        ControlFlow::If {
            predicate,
            consequent,
            alternate,
        } => {
            if let Value::Base(Base::Bool(test)) = predicate.reduce(env)? {
                if test {
                    consequent.reduce(env)
                } else {
                    alternate.reduce(env)
                }
            } else {
                Err(RuntimeError::ExpectedType(Type::Constant(BaseType::Bool)))
            }
        }
    }
}

fn reduce_binding<A>(
    binder: Identifier,
    bound: Expression<A>,
    body: Expression<A>,
    env: &mut Environment,
) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    // This ought to clone it, though?
    let bound = bound.reduce(env)?;
    env.insert_binding(binder.clone(), bound);
    let retval = body.reduce(env);
    env.remove_binding(&binder);
    retval
}

fn apply_function<A>(
    function: Expression<A>,
    argument: Expression<A>,
    env: &mut Environment,
) -> Interpretation
where
    A: fmt::Debug + Clone,
{
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
        Value::RecursiveClosure(RecursiveClosure { inner, .. }) => {
            let binding = argument.reduce(env)?;

            let mut inner = { inner.borrow_mut().clone() };

            let parameter = inner.parameter.clone();
            inner.capture.insert_binding(parameter.clone(), binding);

            let retval = inner.body.clone().reduce(&mut inner.capture);
            inner.capture.remove_binding(&parameter);

            retval
        }
        otherwise => Err(RuntimeError::ExpectedFunction { got: otherwise }),
    }
}

fn make_closure<A>(param: Identifier, body: Expression<A>, env: Environment) -> Interpretation {
    Ok(Value::Closure(Closure {
        parameter: param,
        capture: env,
        body: body.erase_annotation(),
    }))
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Apply, Binding, Constant, ControlFlow, Expression, Identifier, Lambda, Parameter},
        context::Linkage,
        interpreter::{Base, Environment, RuntimeError, Value},
        stdlib,
    };

    use super::Closure;

    #[test]
    fn reduce_literal() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        assert_eq!(
            Base::Int(1),
            Expression::Literal((), Constant::Int(1))
                .reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap(),
        );
    }

    #[test]
    fn reduce_with_variables() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        context
            .interpreter_environment
            .insert_binding(Identifier::new("x"), Value::Base(Base::Int(1)));

        assert_eq!(
            Base::Int(1),
            Expression::Variable((), Identifier::new("x"))
                .reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );

        assert_eq!(
            RuntimeError::UndefinedSymbol(Identifier::new("y")),
            Expression::Variable((), Identifier::new("y"))
                .reduce(&mut context.interpreter_environment)
                .unwrap_err()
        )
    }

    fn _make_fix() -> Expression<()> {
        Expression::Apply(
            (),
            Apply {
                function: Box::new(Expression::Lambda(
                    (),
                    Lambda {
                        parameter: Parameter::new(Identifier::new("x")),
                        body: Box::new(Expression::Apply(
                            (),
                            Apply {
                                function: Box::new(Expression::Variable((), Identifier::new("x"))),
                                argument: Box::new(Expression::Variable((), Identifier::new("x"))),
                            },
                        )),
                    },
                )),
                argument: Box::new(Expression::Lambda(
                    (),
                    Lambda {
                        parameter: Parameter::new(Identifier::new("x")),
                        body: Box::new(Expression::Apply(
                            (),
                            Apply {
                                function: Box::new(Expression::Variable((), Identifier::new("x"))),
                                argument: Box::new(Expression::Variable((), Identifier::new("x"))),
                            },
                        )),
                    },
                )),
            },
        )
    }

    fn _make_fix_value(env: Environment) -> Value {
        Value::Closure(Closure {
            parameter: Identifier::new("f"),
            capture: env.clone(),
            body: Expression::Apply(
                (),
                Apply {
                    function: Box::new(Expression::Lambda(
                        (),
                        Lambda {
                            parameter: Parameter::new(Identifier::new("x")),
                            body: Box::new(Expression::Apply(
                                (),
                                Apply {
                                    function: Box::new(Expression::Variable(
                                        (),
                                        Identifier::new("f"),
                                    )),
                                    argument: Box::new(Expression::Lambda(
                                        (),
                                        Lambda {
                                            parameter: Parameter::new(Identifier::new("y")),
                                            body: Box::new(Expression::Apply(
                                                (),
                                                Apply {
                                                    function: Box::new(Expression::Variable(
                                                        (),
                                                        Identifier::new("x"),
                                                    )),
                                                    argument: Box::new(Expression::Variable(
                                                        (),
                                                        Identifier::new("x"),
                                                    )),
                                                },
                                            )),
                                        },
                                    )),
                                },
                            )),
                        },
                    )),
                    argument: Box::new(Expression::Lambda(
                        (),
                        Lambda {
                            parameter: Parameter::new(Identifier::new("x")),
                            body: Box::new(Expression::Apply(
                                (),
                                Apply {
                                    function: Box::new(Expression::Variable(
                                        (),
                                        Identifier::new("f"),
                                    )),
                                    argument: Box::new(Expression::Lambda(
                                        (),
                                        Lambda {
                                            parameter: Parameter::new(Identifier::new("y")),
                                            body: Box::new(Expression::Apply(
                                                (),
                                                Apply {
                                                    function: Box::new(Expression::Variable(
                                                        (),
                                                        Identifier::new("x"),
                                                    )),
                                                    argument: Box::new(Expression::Variable(
                                                        (),
                                                        Identifier::new("x"),
                                                    )),
                                                },
                                            )),
                                        },
                                    )),
                                },
                            )),
                        },
                    )),
                },
            ),
        })
    }

    #[test]
    fn eval_fix() {
        let _factorial = Expression::Lambda(
            (),
            Lambda {
                parameter: Parameter::new(Identifier::new("x")),
                body: Expression::ControlFlow(
                    (),
                    ControlFlow::If {
                        predicate: Expression::Apply(
                            (),
                            Apply {
                                function: Expression::Variable((), Identifier::new("==")).into(),
                                argument: Expression::Variable((), Identifier::new("x")).into(),
                            },
                        )
                        .into(),
                        consequent: Expression::Literal((), Constant::Int(1)).into(),
                        alternate: Expression::Binding(
                            (),
                            Binding {
                                binder: Identifier::new("xx"),
                                bound: Expression::Apply(
                                    (),
                                    Apply {
                                        function: Expression::Apply(
                                            (),
                                            Apply {
                                                function: Expression::Variable(
                                                    (),
                                                    Identifier::new("-"),
                                                )
                                                .into(),
                                                argument: Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )
                                                .into(),
                                            },
                                        )
                                        .into(),
                                        argument: Expression::Literal((), Constant::Int(1)).into(),
                                    },
                                )
                                .into(),
                                body: Expression::Apply(
                                    (),
                                    Apply {
                                        function: Expression::Apply(
                                            (),
                                            Apply {
                                                function: Expression::Variable(
                                                    (),
                                                    Identifier::new("*"),
                                                )
                                                .into(),
                                                argument: Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )
                                                .into(),
                                            },
                                        )
                                        .into(),
                                        argument: Expression::Apply(
                                            (),
                                            Apply {
                                                function: Expression::Variable(
                                                    (),
                                                    Identifier::new("factorial"),
                                                )
                                                .into(),
                                                argument: Expression::Variable(
                                                    (),
                                                    Identifier::new("xx"),
                                                )
                                                .into(),
                                            },
                                        )
                                        .into(),
                                    },
                                )
                                .into(),
                            },
                        )
                        .into(),
                    },
                )
                .into(),
            },
        );
    }

    //    #[test]
    fn _fixed_factorial() {
        let factorial = Expression::Apply(
            (),
            Apply {
                function: Expression::Variable((), Identifier::new("fix")).into(),
                argument: Expression::Lambda(
                    (),
                    Lambda {
                        parameter: Parameter::new(Identifier::new("fact")),
                        body: Expression::Lambda(
                            (),
                            Lambda {
                                parameter: Parameter::new(Identifier::new("x")),
                                body: Expression::ControlFlow(
                                    (),
                                    ControlFlow::If {
                                        predicate: Expression::Apply(
                                            (),
                                            Apply {
                                                function: Expression::Apply(
                                                    (),
                                                    Apply {
                                                        function: Expression::Variable(
                                                            (),
                                                            Identifier::new("=="),
                                                        )
                                                        .into(),
                                                        argument: Expression::Variable(
                                                            (),
                                                            Identifier::new("x"),
                                                        )
                                                        .into(),
                                                    },
                                                )
                                                .into(),
                                                argument: Expression::Literal((), Constant::Int(0))
                                                    .into(),
                                            },
                                        )
                                        .into(),
                                        consequent: Expression::Literal((), Constant::Int(1))
                                            .into(),
                                        alternate: Expression::Binding(
                                            (),
                                            Binding {
                                                binder: Identifier::new("xx"),
                                                bound: Expression::Apply(
                                                    (),
                                                    Apply {
                                                        function: Expression::Apply(
                                                            (),
                                                            Apply {
                                                                function: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("-"),
                                                                )
                                                                .into(),
                                                                argument: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("x"),
                                                                )
                                                                .into(),
                                                            },
                                                        )
                                                        .into(),
                                                        argument: Expression::Literal(
                                                            (),
                                                            Constant::Int(1),
                                                        )
                                                        .into(),
                                                    },
                                                )
                                                .into(),
                                                body: Expression::Apply(
                                                    (),
                                                    Apply {
                                                        function: Expression::Apply(
                                                            (),
                                                            Apply {
                                                                function: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("*"),
                                                                )
                                                                .into(),
                                                                argument: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("x"),
                                                                )
                                                                .into(),
                                                            },
                                                        )
                                                        .into(),
                                                        argument: Expression::Apply(
                                                            (),
                                                            Apply {
                                                                function: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("fact"),
                                                                )
                                                                .into(),
                                                                argument: Expression::Variable(
                                                                    (),
                                                                    Identifier::new("xx"),
                                                                )
                                                                .into(),
                                                            },
                                                        )
                                                        .into(),
                                                    },
                                                )
                                                .into(),
                                            },
                                        )
                                        .into(),
                                    },
                                )
                                .into(),
                            },
                        )
                        .into(),
                    },
                )
                .into(),
            },
        );

        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        context.interpreter_environment.insert_binding(
            Identifier::new("fix"),
            _make_fix_value(Environment::default()),
        );

        let reduced_fact = factorial
            .reduce(&mut context.interpreter_environment)
            .unwrap();
        context
            .interpreter_environment
            .insert_binding(Identifier::new("factorial"), reduced_fact);

        let e = Expression::Apply(
            (),
            Apply {
                function: Expression::Variable((), Identifier::new("factorial")).into(),
                argument: Expression::Literal((), Constant::Int(1)).into(),
            },
        );

        assert_eq!(
            Base::Int(127),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }
}
