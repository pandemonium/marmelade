use std::collections::{hash_map, HashSet};
use std::{any::Any, cell::RefCell, collections::HashMap, fmt, rc::Rc};
use thiserror::Error;

use crate::{
    ast::{
        Apply, Binding, CompilationUnit, Constant, ControlFlow, Coproduct, Declaration,
        DeconstructInto, Expression, Identifier, ImportModule, Inject, Lambda, MatchClause,
        ModuleDeclarator, Pattern, Product, ProductIndex, Project, SelfReferential, Sequence,
        Struct, StructPattern, TypeDeclaration, TypeDeclarator, TypeName, ValueDeclaration,
        ValueDeclarator,
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

    #[error("{root} has an unsatisfied dependency on {dependency}")]
    UnsatisfiedDependency {
        root: Identifier,
        dependency: Identifier,
    },
}

pub struct ModuleMap<'a, A>(HashMap<Identifier, &'a ModuleDeclarator<A>>);

impl<'a, A> ModuleMap<'a, A> {
    pub fn contains(&self, id: &Identifier) -> bool {
        let Self(map) = self;
        map.contains_key(id)
    }

    pub fn names(&self) -> HashSet<Identifier> {
        let Self(map) = self;
        map.keys().cloned().collect()
    }
}

impl<'a, A> IntoIterator for &'a ModuleMap<'a, A> {
    type Item = (&'a Identifier, &'a &'a ModuleDeclarator<A>);
    type IntoIter = hash_map::Iter<'a, Identifier, &'a ModuleDeclarator<A>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, A> From<Vec<(Identifier, &'a ModuleDeclarator<A>)>> for ModuleMap<'a, A> {
    fn from(value: Vec<(Identifier, &'a ModuleDeclarator<A>)>) -> Self {
        Self(value.into_iter().collect())
    }
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
        A: Copy + Parsed + fmt::Debug + fmt::Display,
    {
        let environment = self.load_compilation_unit(program, typing_context)?;

        match environment.lookup(&Identifier::new("main"))? {
            Value::Closure { .. } => todo!(),
            Value::Bridge { .. } => todo!(),
            scalar => Ok(scalar.clone()),
        }
    }

    fn load_compilation_unit<A>(
        self,
        CompilationUnit {
            annotation,
            mut main,
        }: CompilationUnit<A>,
        mut typing_context: TypingContext,
    ) -> Resolved<Environment>
    where
        A: Copy + Parsed + fmt::Debug + fmt::Display,
    {
        self.inject_prelude(&annotation, &mut main);
        self.inject_modules(&annotation, &mut main, &mut typing_context)?;
        self.resolve_names(&mut main)?;

        let type_checker = TypeChecker::new(typing_context);
        ModuleResolver::initialize(&main, self.prelude)?
            .type_check(type_checker)?
            .into_environment()
    }

    fn resolve_names<A>(&self, main: &mut ModuleDeclarator<A>) -> Resolved<()>
    where
        A: Copy + Parsed + fmt::Debug + fmt::Display,
    {
        let module_map = Self::discover_modules(&main).names();

        main.declarations = main.declarations.drain(..).fold(vec![], |mut acc, decl| {
            let value_decl = if let Declaration::Value(annotation, declaration) = decl {
                Declaration::Value(
                    annotation,
                    declaration.map_expression(|expr| expr.resolve_names(&module_map)),
                )
            } else {
                decl
            };

            acc.push(value_decl);
            acc
        });

        Ok(())
    }

    fn inject_modules<'a, A>(
        &self,
        annotation: &A,
        main: &'a mut ModuleDeclarator<A>,
        typing_context: &mut TypingContext,
    ) -> Resolved<()>
    where
        A: Copy + Parsed + fmt::Debug + fmt::Display,
    {
        let module_map = Self::discover_modules(&main);

        let mut injections = vec![];
        for (name, module) in &module_map {
            let initializer =
                self.inject_module_structures(annotation, name.clone(), &module, typing_context);

            let declarations =
                self.inject_module_types_and_synthetics(annotation, main, typing_context)?;

            injections.push(initializer);
            injections.extend(declarations);
        }

        main.declarations.extend(injections);

        Ok(())
    }

    fn inject_module_structures<'a, A>(
        &self,
        parsing_info: &A,
        module_binder: Identifier,
        module: &'a ModuleDeclarator<A>,
        _typing_context: &mut TypingContext,
    ) -> Declaration<A>
    where
        A: Copy + Parsed + fmt::Debug + fmt::Display,
    {
        let module_declarations = module
            .declarations
            .iter()
            .filter_map(|decl| match decl {
                Declaration::Value(_, decl) => {
                    Some((decl.binder.clone(), decl.declarator.expression.clone()))
                }
                _otherwise => None,
            })
            .collect();

        Declaration::Value(
            *parsing_info,
            ValueDeclaration {
                binder: module_binder,
                type_signature: None,
                declarator: ValueDeclarator {
                    expression: Expression::Product(
                        *parsing_info,
                        Product::Struct(module_declarations),
                    ),
                },
            },
        )
    }

    fn discover_modules<'a, A>(module: &'a ModuleDeclarator<A>) -> ModuleMap<'a, A> {
        let mut modules = vec![(module.name.clone(), module)];

        for decl in &module.declarations {
            if let Declaration::Module(_, module) = decl {
                modules.push((module.name.suffix_with(&module.name.as_str()), module));
            }

            if let Declaration::Type(_, tpe) = decl {
                match &tpe.declarator {
                    TypeDeclarator::Coproduct(
                        _,
                        Coproduct {
                            associated_module: Some(m),
                            ..
                        },
                    ) => modules.push((m.name.clone(), m)),
                    TypeDeclarator::Struct(
                        _,
                        Struct {
                            associated_module: Some(m),
                            ..
                        },
                    ) => modules.push((m.name.clone(), m)),
                    _otherwise => (),
                }
            }
        }

        ModuleMap::from(modules)
    }

    fn inject_prelude<A>(&self, parsing_info: &A, module: &mut ModuleDeclarator<A>)
    where
        A: Copy,
    {
        println!("Loading prelude...");
        module.declarations.push(Declaration::ImportModule(
            *parsing_info,
            ImportModule {
                exported_symbols: self
                    .prelude
                    .symbols()
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            },
        ));
    }

    fn inject_module_types_and_synthetics<A>(
        &self,
        parsing_info: &A,
        module: &ModuleDeclarator<A>,
        typing_context: &mut TypingContext,
    ) -> Typing<Vec<Declaration<A>>>
    where
        A: fmt::Display + fmt::Debug + Copy + Parsed,
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
                    TypeDeclarator::Coproduct(_, coproduct) => self.inject_coproduct(
                        parsing_info,
                        coproduct,
                        binding,
                        &mut injections,
                        typing_context,
                    )?,

                    TypeDeclarator::Struct(_, record) => self.inject_struct(
                        parsing_info,
                        binding,
                        record,
                        &mut injections,
                        typing_context,
                    )?,

                    _otherwise => todo!(),
                }
            }
        }

        Ok(injections)
    }

    fn inject_coproduct<A>(
        &self,
        parsing_info: &A,
        coproduct: &Coproduct<A>,
        binding: &Identifier,
        injections: &mut Vec<Declaration<A>>,
        typing_context: &mut TypingContext,
    ) -> Typing<()>
    where
        A: fmt::Display + fmt::Debug + Copy + Parsed,
    {
        let coproduct = coproduct.make_implementation_module(
            parsing_info,
            TypeName::new(&binding.as_str()),
            typing_context,
        )?;

        println!(
            "inject_types_and_synthetics: `{binding}` `{}`",
            coproduct.declared_type
        );

        typing_context.bind(coproduct.name.into(), coproduct.declared_type);

        for constructor in coproduct.constructors {
            injections.push(Declaration::Value(parsing_info.clone(), constructor));
        }

        Ok(())
    }

    fn inject_struct<A>(
        &self,
        _parsing_info: &A,
        binding: &Identifier,
        record: &Struct<A>,
        _injections: &mut Vec<Declaration<A>>,
        typing_context: &mut TypingContext,
    ) -> Typing<()>
    where
        A: fmt::Display + fmt::Debug + Copy + Parsed,
    {
        let scheme = record.synthesize_type(typing_context)?;
        println!("inject_types_and_synthetics: `{binding}` `{scheme}`");
        typing_context.bind(TypeName::new(&binding.as_str()).into(), scheme);

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
        (_, Pattern::Literally(_, this)) => (scrutinee == &reduce_immediate(this)).then(Vec::new),
        (_, Pattern::Otherwise(_, binding)) => Some(vec![(binding, scrutinee.clone())]),
        _otherwise => None,
    }
}

impl<A> Pattern<A> {}

fn reduce_product<A>(node: Product<A>, env: &mut Environment) -> Interpretation
where
    A: fmt::Debug + Clone,
{
    match node {
        Product::Tuple(elements) => Ok(Value::Tuple(
            elements
                .into_iter()
                .map(|e| e.reduce(env))
                .collect::<Interpretation<_>>()?,
        )),
        Product::Struct(bindings) => Ok(Value::Struct(
            bindings
                .into_iter()
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
