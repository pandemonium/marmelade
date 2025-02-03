use std::collections::HashMap;

use crate::lexer::{self, Location};

pub enum CompilationUnit {
    Implicit(Module),
    Library { modules: Vec<Module>, main: Module },
}

#[derive(Debug, PartialEq)]
pub struct Module {
    pub position: Location,
    pub name: Identifier,
    pub declarations: Vec<Declaration>,
    // pub main: Expression,// I would like this, but I have to think a little more about it
}

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
    Constant {
        initializer: Expression,
        type_annotation: TypeName,
    },
    Function {
        parameters: Vec<Parameter>,
        return_type_annotation: Option<TypeName>,
        body: Expression, // what is this? A Nested lambda?
    },
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
