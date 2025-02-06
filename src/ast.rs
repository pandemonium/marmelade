use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::lexer::{self, Location};

#[derive(Debug)]
pub enum CompilationUnit {
    Implicit(Module),
    Library { modules: Vec<Module>, main: Module },
}

#[derive(Debug, PartialEq)]
pub struct Module {
    pub position: Location,
    pub name: Identifier,
    pub declarations: Vec<Declaration>,
    // pub main: Expression,
    // I would like this, but I have to think a little more about it
}

impl Module {
    pub fn find_value_declaration(&self, id: Identifier) -> Option<&Declaration> {
        self.declarations
            .iter()
            .find(|decl| matches!(decl, Declaration::Value { binder, .. } if binder == &id))
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
    Module(Module),
}

impl Declaration {
    pub fn position(&self) -> &Location {
        match self {
            Self::Value { position, .. }
            | Self::Type { position, .. }
            | Self::Module(Module { position, .. }) => position,
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

#[derive(Debug, PartialEq)]
pub enum ValueDeclarator {
    Constant(ConstantDeclarator),
    Function(FunctionDeclarator),
}

impl ValueDeclarator {
    // how does it find which functions this expression depends on?
    // or is variables enough?
    // So all free symbols?
    // All symbols that are neither parameters nor local variables are
    // free symbols
    // They are more than these really because shadowing
    // so I would really need to "interpret" my way down the tree
    // So starting with an empty set of symbols as closed
    // add all parameters
    // then add any Expr::Var as free, unless in closed
    // add any let binder to closed
    fn dependencies(&self) -> Vec<&Identifier> {
        match self {
            Self::Constant(constant_declarator) => todo!(),
            Self::Function(function_declarator) => todo!(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct ConstantDeclarator {
    pub initializer: Expression,
    pub type_annotation: TypeName,
}

#[derive(Debug, PartialEq)]
pub struct FunctionDeclarator {
    pub parameters: Vec<Parameter>,
    pub return_type_annotation: Option<TypeName>,
    pub body: Expression,
}

impl FunctionDeclarator {
    // does this function really go here?
    fn into_lambda_tree(self) -> Expression {
        self.parameters
            .into_iter()
            .rev()
            .fold(self.body, |body, parameter| Expression::Lambda {
                parameter,
                body: body.into(),
            })
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
        self.find_free_identifiers(&mut HashSet::default(), &mut free_identifiers);
        free_identifiers
    }

    fn find_free_identifiers<'a>(
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
                bound.insert(&parameter.name);
                body.find_free_identifiers(bound, free);
            }
            Self::Apply { function, argument } => {
                function.find_free_identifiers(bound, free);
                argument.find_free_identifiers(bound, free);
            }
            Self::Construct { argument, .. } => argument.find_free_identifiers(bound, free),
            Self::Product(Product::Tuple(expressions)) => {
                for e in expressions {
                    e.find_free_identifiers(bound, free);
                }
            }
            Self::Product(Product::Struct { bindings }) => {
                for e in bindings.values() {
                    e.find_free_identifiers(bound, free);
                }
            }
            Self::Project { base, .. } => base.find_free_identifiers(bound, free),
            Self::Binding {
                binder,
                bound: bound_expr,
                body,
                ..
            } => {
                bound_expr.find_free_identifiers(bound, free);
                bound.insert(binder);
                body.find_free_identifiers(bound, free);
            }
            Self::Sequence { this, and_then } => {
                this.find_free_identifiers(bound, free);
                and_then.find_free_identifiers(bound, free);
            }
            Self::ControlFlow(ControlFlow::If {
                predicate,
                consequent,
                alternate,
            }) => {
                predicate.find_free_identifiers(bound, free);
                consequent.find_free_identifiers(bound, free);
                alternate.find_free_identifiers(bound, free);
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
