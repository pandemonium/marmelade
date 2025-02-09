use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::lexer::{self, Location};

#[derive(Debug)]
pub enum CompilationUnit {
    Implicit(ModuleDeclarator),
    Library {
        modules: Vec<ModuleDeclarator>,
        main: ModuleDeclarator,
    },
}

#[derive(Debug, PartialEq)]
pub struct ModuleDeclarator {
    pub position: Location,
    pub name: Identifier,
    pub declarations: Vec<Declaration>,
    // pub main: Expression,
    // I would like this, but I have to think a little more about it
}

impl ModuleDeclarator {
    pub fn find_value_declaration<'a>(&'a self, id: &'a Identifier) -> Option<&'a Declaration> {
        self.declarations
            .iter()
            .find(|decl| matches!(decl, Declaration::Value { binder, .. } if binder == id))
    }

    pub fn dependency_matrix(&self) -> DependencyMatrix {
        DependencyMatrix::from_declarations(&self.declarations)
    }
}

// Where is the stdlib stuff? Or other libraries.
// Is a Use also a pathway to Declarations?

// I want to compute where to start typing the compilation unit
// Which is at a function which only has dependencies to symbols which
// already have known types
#[derive(Debug)]
pub struct DependencyMatrix<'a> {
    outbound_dependencies: HashMap<&'a Identifier, Vec<&'a Identifier>>,
    inbound_dependencies: HashMap<&'a Identifier, Vec<&'a Identifier>>,
}

impl<'a> DependencyMatrix<'a> {
    pub fn from_declarations(decls: &'a [Declaration]) -> Self {
        let mut outbound = HashMap::default();
        let mut inbound: HashMap<&'a Identifier, Vec<&'a Identifier>> = HashMap::default();

        for decl in decls {
            match decl {
                Declaration::Value {
                    binder, declarator, ..
                } => {
                    let deps = declarator.dependencies();
                    for dep in &deps {
                        inbound.entry(dep).or_default().push(binder);
                    }
                    outbound.insert(binder, deps);
                }
                Declaration::ImportModule {
                    exported_symbols, ..
                } => {
                    for dep in exported_symbols {
                        outbound.insert(dep, Vec::default());
                    }
                }
                _otherwise => (),
            }
        }

        Self {
            outbound_dependencies: outbound,
            inbound_dependencies: inbound,
        }
    }

    pub fn is_wellformed<F>(&'a self, is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        self.is_acyclic() && self.is_satisfiable(is_external)
    }

    pub fn is_satisfiable<F>(&'a self, mut is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        self.outbound_dependencies.values().all(|deps| {
            deps.iter().all(|dep| {
                let retval = self.outbound_dependencies.contains_key(dep) || is_external(dep);

                if !retval {
                    println!("is_satisfiable: `{dep}` not found");
                    println!("is_satisfiable: {:?}", self.outbound_dependencies)
                }

                retval
            })
        })
    }

    pub fn is_acyclic(&'a self) -> bool {
        !self
            .outbound_dependencies
            .keys()
            .into_iter()
            .any(|id| self.is_cyclic(id, id, &mut HashSet::default()))
    }

    fn is_cyclic(
        &self,
        needle: &'a Identifier,
        node: &'a Identifier,
        seen: &mut HashSet<&'a Identifier>,
    ) -> bool {
        seen.insert(node);

        let mut is_in_subtree = |id| !seen.contains(id) && self.is_cyclic(needle, id, seen);

        self.dependencies(node)
            .unwrap_or_default()
            .iter()
            .any(|&child| needle == child || is_in_subtree(child))
    }

    pub fn nodes(&self) -> Vec<&'a Identifier> {
        self.outbound_dependencies.keys().cloned().collect()
    }

    pub fn find<F>(&'a self, mut p: F) -> Option<&'a &'a Identifier>
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.outbound_dependencies.keys().find(|id| p(id))
    }

    pub fn satisfies<F>(&'a self, mut p: F) -> bool
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.outbound_dependencies.keys().all(|id| p(id))
    }

    pub fn dependencies(&self, d: &'a Identifier) -> Option<&[&'a Identifier]> {
        self.outbound_dependencies.get(d).map(Vec::as_slice)
    }

    pub fn depends_on(&self, d: &'a Identifier) -> Option<&[&'a Identifier]> {
        self.inbound_dependencies.get(d).map(Vec::as_slice)
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

    pub fn scoped_with(&self, scope: String) -> Self {
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

#[derive(Debug, PartialEq)]
pub enum Declaration {
    Value {
        position: Location,
        binder: Identifier,
        declarator: ValueDeclarator,
    },
    Type {
        position: Location,
        binding: Identifier,
        declarator: TypeDeclarator,
    },
    Module(ModuleDeclarator),
    ImportModule {
        position: Location,
        exported_symbols: Vec<Identifier>,
    },
    // Use()    ??
}

impl Declaration {
    pub fn position(&self) -> &Location {
        match self {
            Self::Value { position, .. }
            | Self::Type { position, .. }
            | Self::Module(ModuleDeclarator { position, .. })
            | Self::ImportModule { position, .. } => position,
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

#[derive(Debug, Clone, PartialEq)]
pub enum ValueDeclarator {
    Constant(ConstantDeclarator),
    Function(FunctionDeclarator),
}

impl ValueDeclarator {
    pub fn dependencies(&self) -> Vec<&Identifier> {
        let mut free = match self {
            Self::Constant(decl) => decl.free_identifiers(),
            Self::Function(decl) => decl.free_identifiers(),
        };
        free.drain().collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantDeclarator {
    pub initializer: Expression,
    pub type_annotation: Option<TypeName>,
}

impl ConstantDeclarator {
    pub fn free_identifiers(&self) -> HashSet<&Identifier> {
        self.initializer.free_identifiers()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDeclarator {
    pub parameters: Vec<Parameter>,
    pub return_type_annotation: Option<TypeName>,
    pub body: Expression,
}

impl FunctionDeclarator {
    // does this function really go here?
    pub fn into_lambda_tree(self) -> Expression {
        self.parameters
            .into_iter()
            .rev()
            .fold(self.body, |body, parameter| Expression::Lambda {
                parameter,
                body: body.into(),
            })
    }

    pub fn free_identifiers(&self) -> HashSet<&Identifier> {
        let mut free = self.body.free_identifiers();
        for param in &self.parameters {
            free.remove(&param.name);
        }
        free
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Variable(Identifier),
    InvokeSynthetic(Identifier),
    Literal(Constant),
    Lambda {
        parameter: Parameter,
        body: Box<Expression>,
    },
    Apply {
        function: Box<Expression>,
        argument: Box<Expression>,
    },
    Construct {
        name: TypeName,
        constructor: Identifier,
        argument: Box<Expression>,
    },
    Product(Product),
    Project {
        base: Box<Expression>,
        index: ProductIndex,
    },
    Binding {
        postition: lexer::Location,
        binder: Identifier,
        bound: Box<Expression>,
        body: Box<Expression>,
    },
    Sequence {
        this: Box<Expression>,
        and_then: Box<Expression>,
    },
    ControlFlow(ControlFlow),
}

impl Expression {
    pub fn position(&self) -> Option<&lexer::Location> {
        if let Self::Binding { postition, .. } = self {
            Some(postition)
        } else {
            None
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
            Self::Variable(id) => {
                if !bound.contains(id) {
                    free.insert(id);
                }
            }
            Self::InvokeSynthetic(id) => {
                free.insert(id);
            }
            Self::Lambda { parameter, body } => {
                // This is probably not correct
                // I have to remove this after looking in "body
                bound.insert(&parameter.name);
                body.find_unbound(bound, free);
            }
            Self::Apply { function, argument } => {
                function.find_unbound(bound, free);
                argument.find_unbound(bound, free);
            }
            Self::Construct { argument, .. } => argument.find_unbound(bound, free),
            Self::Product(Product::Tuple(expressions)) => {
                for e in expressions {
                    e.find_unbound(bound, free);
                }
            }
            Self::Product(Product::Struct { bindings }) => {
                for e in bindings.values() {
                    e.find_unbound(bound, free);
                }
            }
            Self::Project { base, .. } => base.find_unbound(bound, free),
            Self::Binding {
                binder,
                bound: bound_expr,
                body,
                ..
            } => {
                bound_expr.find_unbound(bound, free);
                bound.insert(binder);
                body.find_unbound(bound, free);
            }
            Self::Sequence { this, and_then } => {
                this.find_unbound(bound, free);
                and_then.find_unbound(bound, free);
            }
            Self::ControlFlow(ControlFlow::If {
                predicate,
                consequent,
                alternate,
            }) => {
                predicate.find_unbound(bound, free);
                consequent.find_unbound(bound, free);
                alternate.find_unbound(bound, free);
            }
            _otherwise => (),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlow {
    If {
        predicate: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductIndex {
    Tuple(usize),
    Struct(Identifier),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
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
pub enum Product {
    Tuple(Vec<Expression>),
    Struct {
        bindings: HashMap<Identifier, Expression>,
    },
}

#[cfg(test)]
mod tests {
    use crate::lexer::Location;

    use super::{
        Constant, ConstantDeclarator, Declaration, Expression, Identifier, ModuleDeclarator,
        ValueDeclarator,
    };

    #[test]
    fn cyclic_dependencies() {
        // I should parse text instead of this
        let m = ModuleDeclarator {
            position: Location::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("foo")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("quux")),
                        type_annotation: None,
                    }),
                },
            ],
        };

        assert!(!m.dependency_matrix().is_acyclic());
    }

    #[test]
    fn satisfiable() {
        let m = ModuleDeclarator {
            position: Location::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("frobnicator")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("frobnicator"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Literal(Constant::Int(1)),
                        type_annotation: None,
                    }),
                },
            ],
        };

        let dep_mat = m.dependency_matrix();
        assert!(dep_mat.is_acyclic());
        assert!(dep_mat.is_satisfiable(|_| false));
    }

    #[test]
    fn unsatisfiable() {
        let m = ModuleDeclarator {
            position: Location::default(),
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("foo"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("quux"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("bar")),
                        type_annotation: None,
                    }),
                },
                Declaration::Value {
                    position: Location::default(),
                    binder: Identifier::new("bar"),
                    declarator: ValueDeclarator::Constant(ConstantDeclarator {
                        initializer: Expression::Variable(Identifier::new("frobnicator")),
                        type_annotation: None,
                    }),
                },
            ],
        };

        let dep_mat = m.dependency_matrix();
        assert!(dep_mat.is_acyclic());
        assert!(!dep_mat.is_satisfiable(|_| false));
    }
}
