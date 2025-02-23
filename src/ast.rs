use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::{
    interpreter::DependencyGraph,
    lexer::{self, SourcePosition},
    parser::ParsingInfo,
};

#[derive(Debug)]
pub enum CompilationUnit<A> {
    Implicit(ModuleDeclarator<A>),
    Library {
        modules: Vec<ModuleDeclarator<A>>,
        main: ModuleDeclarator<A>,
    },
}
impl<A> CompilationUnit<A>
where
    A: Clone,
{
    pub fn map<B>(self, f: fn(A) -> B) -> CompilationUnit<B> {
        match self {
            Self::Implicit(module) => CompilationUnit::<B>::Implicit(module.map(f)),
            Self::Library { mut modules, main } => CompilationUnit::<B>::Library {
                modules: modules.drain(..).map(|m| m.map(f)).collect(),
                main: main.map(f),
            },
        }
    }
}

impl<A> fmt::Display for CompilationUnit<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Implicit(module) => writeln!(f, "{module}"),
            Self::Library { modules, main } => {
                for m in modules {
                    writeln!(f, "{m}")?;
                }
                write!(f, "{main}")
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ModuleDeclarator<A> {
    pub position: SourcePosition,
    pub name: Identifier,
    pub declarations: Vec<Declaration<A>>,
    // pub main: Expression,
    // I would like this, but I have to think a little more about it
}

impl<A> ModuleDeclarator<A>
where
    A: Clone,
{
    pub fn find_value_declaration<'a>(&'a self, id: &'a Identifier) -> Option<&'a Declaration<A>> {
        self.declarations
            .iter()
            .find(|decl| matches!(decl, Declaration::Value { binder, .. } if binder == id))
    }

    pub fn dependency_graph(&self) -> DependencyGraph {
        DependencyGraph::from_declarations(&self.declarations)
    }

    fn map<B>(mut self, f: fn(A) -> B) -> ModuleDeclarator<B> {
        ModuleDeclarator::<B> {
            position: self.position,
            name: self.name,
            declarations: self.declarations.drain(..).map(|d| d.map(f)).collect(),
        }
    }
}

impl<A> fmt::Display for ModuleDeclarator<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            name, declarations, ..
        } = self;
        writeln!(f, "module {name}")?;
        for decl in declarations {
            writeln!(f, "{decl}")?;
        }

        writeln!(f, "end {name}")
    }
}

// Could this be an enum?
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier(String);

impl Identifier {
    pub fn new(x: &str) -> Self {
        Self(x.to_owned())
    }

    pub fn as_str(&self) -> &str {
        let Self(x) = self;
        x
    }

    pub fn scoped_with(&self, scope: &str) -> Self {
        let Self(id) = self;
        Self(format!("{scope}::{id}"))
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(id) = self;
        write!(f, "{id}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeName(String);

impl TypeName {
    pub fn as_str(&self) -> &str {
        let Self(x) = self;
        x
    }

    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(name) = self;
        write!(f, "{name}")
    }
}

#[derive(Debug, PartialEq)]
pub enum Declaration<A> {
    Value {
        position: SourcePosition,
        binder: Identifier,
        declarator: ValueDeclarator<A>,
    },
    Type {
        position: SourcePosition,
        binding: Identifier,
        declarator: TypeDeclarator,
    },
    Module(ModuleDeclarator<A>),
    ImportModule {
        position: SourcePosition,
        exported_symbols: Vec<Identifier>,
    },
    // Use()    ??
}

impl<A> Declaration<A>
where
    A: Clone,
{
    pub fn position(&self) -> &SourcePosition {
        match self {
            Self::Value { position, .. }
            | Self::Type { position, .. }
            | Self::Module(ModuleDeclarator { position, .. })
            | Self::ImportModule { position, .. } => position,
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Declaration<B> {
        match self {
            Self::Value {
                position,
                binder,
                declarator,
            } => Declaration::<B>::Value {
                position,
                binder,
                declarator: declarator.map(f),
            },
            Self::Type {
                position,
                binding,
                declarator,
            } => Declaration::<B>::Type {
                position,
                binding,
                declarator,
            },
            Self::Module(declarator) => Declaration::<B>::Module(declarator.map(f)),
            Self::ImportModule {
                position,
                exported_symbols,
            } => Declaration::<B>::ImportModule {
                position,
                exported_symbols,
            },
        }
    }
}

impl<A> fmt::Display for Declaration<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value {
                binder, declarator, ..
            } => write!(f, "{binder} = {declarator}"),
            Self::Type {
                binding,
                declarator,
                ..
            } => write!(f, "{binding} = {declarator}"),
            Self::Module(module) => write!(f, "{module}"),
            Self::ImportModule {
                exported_symbols, ..
            } => {
                write!(f, "import ")?;
                for sym in exported_symbols {
                    write!(f, "{sym},")?;
                }
                //                writeln!(f, "")
                Ok(())
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum TypeDeclarator {
    Alias {
        alias: Identifier,
        aliased: TypeName,
    },
    Coproduct(Vec<Constructor>),
    Struct(Vec<StructField>),
}

impl fmt::Display for TypeDeclarator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alias { alias, aliased } => write!(f, "type alias {alias} = {aliased}"),
            Self::Coproduct(constructors) => {
                for c in constructors {
                    write!(f, "| {c}")?;
                }
                Ok(())
            }
            Self::Struct(fields) => {
                writeln!(f, "struct {{")?;
                for StructField {
                    name,
                    type_annotation,
                } in fields
                {
                    writeln!(f, "{name} :: {type_annotation}")?;
                }
                writeln!(f, "}}")
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct StructField {
    pub name: Identifier,
    pub type_annotation: TypeName,
}

#[derive(Debug, PartialEq)]
pub struct Constructor {
    pub name: Identifier,
    pub signature: Vec<TypeName>,
}

impl fmt::Display for Constructor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { name, signature } = self;
        write!(f, "{name} :: (")?;
        if signature.len() > 1 {
            write!(f, "{}", &signature[0])?;
            for ty in &signature[1..] {
                write!(f, " * {ty}")?;
            }
        } else {
            for ty in signature {
                write!(f, "{ty}")?;
            }
        }
        writeln!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueDeclarator<A> {
    Constant(ConstantDeclarator<A>),
    Function(FunctionDeclarator<A>),
}

impl<A> ValueDeclarator<A>
where
    A: Clone,
{
    pub fn dependencies(&self) -> HashSet<&Identifier> {
        let mut free = match self {
            Self::Constant(decl) => decl.free_identifiers(),
            Self::Function(decl) => decl.free_identifiers(),
        };
        free.drain().collect()
    }

    fn map<B>(self, f: fn(A) -> B) -> ValueDeclarator<B> {
        match self {
            Self::Constant(constant) => ValueDeclarator::<B>::Constant(constant.map(f)),
            Self::Function(function) => ValueDeclarator::<B>::Function(function.map(f)),
        }
    }
}

impl<A> fmt::Display for ValueDeclarator<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(constant) => write!(f, "{constant}"),
            Self::Function(function) => write!(f, "{function}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantDeclarator<A> {
    pub initializer: Expression<A>,
    pub type_annotation: Option<TypeName>,
}

impl<A> ConstantDeclarator<A> {
    pub fn free_identifiers(&self) -> HashSet<&Identifier> {
        self.initializer.free_identifiers()
    }

    fn map<B>(self, f: fn(A) -> B) -> ConstantDeclarator<B> {
        ConstantDeclarator::<B> {
            initializer: self.initializer.map(f),
            type_annotation: self.type_annotation,
        }
    }
}

impl<A> fmt::Display for ConstantDeclarator<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            initializer,
            type_annotation,
        } = self;
        write!(f, "{initializer}")?;
        if let Some(ty) = type_annotation {
            write!(f, "[{ty}]")?;
        }

        //        writeln!(f, "")
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDeclarator<A> {
    pub parameters: Vec<Parameter>,
    pub return_type_annotation: Option<TypeName>,
    pub body: Expression<A>,
}

impl<A> FunctionDeclarator<A>
where
    A: Clone,
{
    // does this function really go here?
    pub fn into_lambda_tree(mut self, self_name: Identifier) -> Expression<A> {
        self.parameters
            .drain(..)
            .rev()
            .fold(self.body, |body, parameter| {
                Expression::SelfReferential(
                    body.annotation().clone(),
                    SelfReferential {
                        name: self_name.clone(),
                        parameter,
                        body: body.into(),
                    },
                )
            })
    }

    pub fn free_identifiers(&self) -> HashSet<&Identifier> {
        let mut free = self.body.free_identifiers();
        for param in &self.parameters {
            free.remove(&param.name);
        }
        free
    }

    fn map<B>(self, f: fn(A) -> B) -> FunctionDeclarator<B> {
        FunctionDeclarator::<B> {
            parameters: self.parameters,
            return_type_annotation: self.return_type_annotation,
            body: self.body.map(f),
        }
    }
}

impl<A> fmt::Display for FunctionDeclarator<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            parameters,
            return_type_annotation,
            body,
        } = self;

        write!(f, "(")?;
        for p in parameters {
            write!(f, "{p}, ")?;
        }
        write!(f, ")")?;

        if let Some(ty) = return_type_annotation {
            write!(f, " -> {ty}")?;
        }

        writeln!(f, " =")?;
        write!(f, "{body}")
    }
}

// these can be pattern matches too
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Option<TypeName>,
}

impl Parameter {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            type_annotation: None,
        }
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            name,
            type_annotation,
        } = self;
        write!(f, "{name}")?;

        if let Some(ty) = type_annotation {
            write!(f, ": {ty}")?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression<A> {
    Variable(A, Identifier),
    CallBridge(A, Identifier),
    Literal(A, Constant),
    SelfReferential(A, SelfReferential<A>),
    Lambda(A, Lambda<A>),
    Apply(A, Apply<A>),
    Construct(A, Construct<A>),
    Product(A, Product<A>),
    Project(A, Project<A>),
    Binding(A, Binding<A>),
    Sequence(A, Sequence<A>),
    ControlFlow(A, ControlFlow<A>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfReferential<A> {
    pub name: Identifier,
    pub parameter: Parameter,
    pub body: Box<Expression<A>>,
}
impl<A> SelfReferential<A> {
    fn map<B>(self, f: fn(A) -> B) -> SelfReferential<B> {
        SelfReferential::<B> {
            name: self.name,
            parameter: self.parameter,
            body: self.body.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lambda<A> {
    pub parameter: Parameter,
    pub body: Box<Expression<A>>,
}
impl<A> Lambda<A> {
    fn map<B>(self, f: fn(A) -> B) -> Lambda<B> {
        Lambda::<B> {
            parameter: self.parameter,
            body: self.body.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Apply<A> {
    pub function: Box<Expression<A>>,
    pub argument: Box<Expression<A>>,
}
impl<A> Apply<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Apply<B> {
        Apply::<B> {
            function: self.function.map(f).into(),
            argument: self.argument.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Construct<A> {
    pub name: TypeName,
    pub constructor: Identifier,
    pub argument: Box<Expression<A>>,
}
impl<A> Construct<A> {
    fn map<B>(self, f: fn(A) -> B) -> Construct<B> {
        Construct::<B> {
            name: self.name,
            constructor: self.constructor,
            argument: self.argument.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Project<A> {
    pub base: Box<Expression<A>>,
    pub index: ProductIndex,
}
impl<A> Project<A> {
    fn map<B>(self, f: fn(A) -> B) -> Project<B> {
        Project::<B> {
            base: self.base.map(f).into(),
            index: self.index,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding<A> {
    pub binder: Identifier,
    pub bound: Box<Expression<A>>,
    pub body: Box<Expression<A>>,
}
impl<A> Binding<A> {
    fn map<B>(self, f: fn(A) -> B) -> Binding<B> {
        Binding::<B> {
            binder: self.binder,
            bound: self.bound.map(f).into(),
            body: self.body.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sequence<A> {
    pub this: Box<Expression<A>>,
    pub and_then: Box<Expression<A>>,
}
impl<A> Sequence<A> {
    fn map<B>(self, f: fn(A) -> B) -> Sequence<B> {
        Sequence::<B> {
            this: self.this.map(f).into(),
            and_then: self.and_then.map(f).into(),
        }
    }
}

impl Expression<ParsingInfo> {
    pub fn position(&self) -> &lexer::SourcePosition {
        &self.parsing_info().position
    }

    pub fn parsing_info(&self) -> &ParsingInfo {
        self.annotation()
    }
}

impl<A> Expression<A> {
    pub fn annotation(&self) -> &A {
        match self {
            Self::Variable(annotation, ..) => annotation,
            Self::CallBridge(annotation, ..) => annotation,
            Self::Literal(annotation, ..) => annotation,
            Self::SelfReferential(annotation, ..) => annotation,
            Self::Lambda(annotation, ..) => annotation,
            Self::Apply(annotation, ..) => annotation,
            Self::Construct(annotation, ..) => annotation,
            Self::Product(annotation, ..) => annotation,
            Self::Project(annotation, ..) => annotation,
            Self::Binding(annotation, ..) => annotation,
            Self::Sequence(annotation, ..) => annotation,
            Self::ControlFlow(annotation, ..) => annotation,
        }
    }

    pub fn free_identifiers<'a>(&'a self) -> HashSet<&'a Identifier> {
        let mut free_identifiers = HashSet::default();
        self.find_unbound(&mut HashSet::default(), &mut free_identifiers);
        free_identifiers
    }

    fn find_unbound<'a>(
        &'a self,
        bound: &mut HashSet<&'a Identifier>,
        free: &mut HashSet<&'a Identifier>,
    ) {
        match self {
            Self::Variable(_, id) => {
                if !bound.contains(id) {
                    free.insert(id);
                }
            }
            Self::CallBridge(_, id) => {
                free.insert(id);
            }
            Self::Lambda(_, Lambda { parameter, body }) => {
                // This is probably not correct
                // I have to remove this after looking in "body
                bound.insert(&parameter.name);
                body.find_unbound(bound, free);
            }
            Self::Apply(_, Apply { function, argument }) => {
                function.find_unbound(bound, free);
                argument.find_unbound(bound, free);
            }
            Self::Construct(_, Construct { argument, .. }) => argument.find_unbound(bound, free),
            Self::Product(_, Product::Tuple(expressions)) => {
                for e in expressions {
                    e.find_unbound(bound, free);
                }
            }
            Self::Product(_, Product::Struct { bindings }) => {
                for e in bindings.values() {
                    e.find_unbound(bound, free);
                }
            }
            Self::Project(_, Project { base, .. }) => base.find_unbound(bound, free),
            Self::Binding(
                _,
                Binding {
                    binder,
                    bound: bound_expr,
                    body,
                    ..
                },
            ) => {
                bound_expr.find_unbound(bound, free);
                bound.insert(binder);
                body.find_unbound(bound, free);
            }
            Self::Sequence(_, Sequence { this, and_then }) => {
                this.find_unbound(bound, free);
                and_then.find_unbound(bound, free);
            }
            Self::ControlFlow(
                _,
                ControlFlow::If {
                    predicate,
                    consequent,
                    alternate,
                },
            ) => {
                predicate.find_unbound(bound, free);
                consequent.find_unbound(bound, free);
                alternate.find_unbound(bound, free);
            }
            _otherwise => (),
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Expression<B> {
        match self {
            Self::Variable(x, info) => Expression::<B>::Variable(f(x), info),
            Self::CallBridge(x, info) => Expression::<B>::CallBridge(f(x), info),
            Self::Literal(x, info) => Expression::<B>::Literal(f(x), info),
            Self::SelfReferential(x, info) => Expression::<B>::SelfReferential(f(x), info.map(f)),
            Self::Lambda(x, info) => Expression::<B>::Lambda(f(x), info.map(f)),
            Self::Apply(x, info) => Expression::<B>::Apply(f(x), info.map(f)),
            Self::Construct(x, info) => Expression::<B>::Construct(f(x), info.map(f)),
            Self::Product(x, info) => Expression::<B>::Product(f(x), info.map(f)),
            Self::Project(x, info) => Expression::<B>::Project(f(x), info.map(f)),
            Self::Binding(x, info) => Expression::<B>::Binding(f(x), info.map(f)),
            Self::Sequence(x, info) => Expression::<B>::Sequence(f(x), info.map(f)),
            Self::ControlFlow(x, info) => Expression::<B>::ControlFlow(f(x), info.map(f)),
        }
    }

    pub fn erase_annotation(self) -> Expression<()> {
        self.map(|_| ())
    }
}

impl<A> fmt::Display for Expression<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(_, id) => write!(f, "{id}"),
            Expression::CallBridge(_, id) => write!(f, "call {id}"),
            Expression::Literal(_, c) => write!(f, "{c}"),
            Expression::SelfReferential(_, SelfReferential { name, body, .. }) => {
                write!(f, "-----> {name}->[{body}]")
            }
            Expression::Lambda(_, Lambda { parameter, body }) => {
                write!(f, "lambda \\{parameter}. {body}")
            }
            Expression::Apply(_, Apply { function, argument }) => {
                write!(f, "{function} {argument}")
            }
            Expression::Construct(
                _,
                Construct {
                    name,
                    constructor,
                    argument,
                },
            ) => write!(f, "{name}::{constructor} {argument}"),
            Expression::Product(_, product) => write!(f, "{product}"),
            Expression::Project(_, Project { base, index }) => write!(f, "{base}.{index}"),
            Expression::Binding(
                _,
                Binding {
                    binder,
                    bound,
                    body,
                    ..
                },
            ) => write!(f, "let {binder} = {bound} in {body}"),
            Expression::Sequence(_, Sequence { this, and_then }) => {
                writeln!(f, "{this}\n{and_then}")
            }
            Expression::ControlFlow(_, control) => writeln!(f, "{control}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlow<A> {
    If {
        predicate: Box<Expression<A>>,
        consequent: Box<Expression<A>>,
        alternate: Box<Expression<A>>,
    },
}
impl<A> ControlFlow<A> {
    fn map<B>(self, f: fn(A) -> B) -> ControlFlow<B> {
        match self {
            Self::If {
                predicate,
                consequent,
                alternate,
            } => ControlFlow::<B>::If {
                predicate: predicate.map(f).into(),
                consequent: consequent.map(f).into(),
                alternate: alternate.map(f).into(),
            },
        }
    }
}

impl<A> fmt::Display for ControlFlow<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::If {
            predicate,
            consequent,
            alternate,
        } = self;
        writeln!(f, "if {predicate}\nthen {consequent}\nelse\n{alternate}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductIndex {
    Tuple(usize),
    Struct(Identifier),
}

impl fmt::Display for ProductIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(index) => write!(f, "{index}"),
            Self::Struct(id) => write!(f, "{id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Unit,
}

impl fmt::Display for Constant {
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

impl From<lexer::Literal> for Constant {
    fn from(value: lexer::Literal) -> Self {
        match value {
            lexer::Literal::Integer(x) => Self::Int(x),
            lexer::Literal::Text(x) => Self::Text(x),
            lexer::Literal::Bool(x) => Self::Bool(x),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Product<A> {
    Tuple(Vec<Expression<A>>),
    Struct {
        bindings: HashMap<Identifier, Expression<A>>,
    },
}
impl<A> Product<A> {
    fn map<B>(self, f: fn(A) -> B) -> Product<B> {
        match self {
            Self::Tuple(mut expressions) => {
                Product::<B>::Tuple(expressions.drain(..).map(|x| x.map(f)).collect())
            }
            Self::Struct { mut bindings } => Product::<B>::Struct {
                bindings: bindings.drain().map(|(k, v)| (k, v.map(f))).collect(),
            },
        }
    }
}

impl<A> fmt::Display for Product<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(expressions) => {
                write!(f, "(")?;
                for e in expressions {
                    write!(f, "{e},")?;
                }
                write!(f, ")")
            }
            Self::Struct { bindings } => {
                write!(f, "{{")?;
                for (field, value) in bindings {
                    write!(f, "{field}: {value},")?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::SourcePosition;

    use super::{
        Constant, ConstantDeclarator, Declaration, Expression, Identifier, ModuleDeclarator,
        ValueDeclarator,
    };

    #[test]
    fn cyclic_dependencies() {
        // I should parse text instead of this
        let m = ModuleDeclarator {
            position: SourcePosition::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("foo")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("quux")),
                        type_annotation: None,
                    }),
                },
            ],
        };

        // Still broken
        assert!(!m.dependency_graph().is_acyclic());
    }

    #[test]
    fn satisfiable() {
        let m = ModuleDeclarator {
            position: SourcePosition::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("frobnicator")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("frobnicator"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Literal((), Constant::Int(1)),
                        type_annotation: None,
                    }),
                },
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(dep_mat.is_satisfiable(|_| false));
    }

    #[test]
    fn unsatisfiable() {
        let m = ModuleDeclarator {
            position: SourcePosition::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("frobnicator")),
                        type_annotation: None,
                    }),
                },
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(!dep_mat.is_satisfiable(|_| false));
    }

    #[test]
    fn top_of_the_day() {
        let m = ModuleDeclarator {
            position: SourcePosition::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable((), Identifier::new("frobnicator")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: SourcePosition::default(),
                    binder: Identifier::new("frobnicator"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Literal((), Constant::Int(1)),
                        type_annotation: None,
                    }),
                },
            ],
        };

        let graph = m.dependency_graph();

        let order = graph.compute_resolution_order();

        // This order is not stable. Find a better way.
        assert_eq!(
            vec![
                &Identifier::new("frobnicator"),
                &Identifier::new("bar"),
                &Identifier::new("foo"),
                &Identifier::new("quux"),
            ],
            order
        );
    }
}
