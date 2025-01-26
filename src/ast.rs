use std::collections::HashMap;

pub struct Module {
    pub declarations: Vec<Declaration>,
    pub main: Expression,
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

#[derive(Debug, Clone)]
pub struct TypeName(String);

impl TypeName {
    pub fn as_str(&self) -> &str {
        let Self(x) = self;
        x
    }
}

pub enum Declaration {
    Value {
        binding: Identifier,
        declarator: ValueDeclarator,
    },
    Type {
        binding: Identifier,
        declarator: TypeDeclarator,
    },
}

pub enum TypeDeclarator {
    Alias {
        alias: Identifier,
        aliased: TypeName,
    },
    Coproduct(Vec<Constructor>),
    Struct(Vec<StructField>),
}

pub struct StructField {
    name: Identifier,
    type_annotation: TypeName,
}

pub struct Constructor {
    name: Identifier,
    signature: Vec<TypeName>,
}

pub enum ValueDeclarator {
    Constant {
        initializer: Expression,
        type_annotation: TypeName,
    },
    Function {
        parameters: Vec<Parameter>,
        return_type_annotation: Option<TypeName>,
    },
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
    Coproduct {
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
        binding: Identifier,
        bound: Box<Expression>,
        body: Box<Expression>,
    },
    ControlFlow(ControlFlow),
}

#[derive(Debug, Clone)]
pub enum ControlFlow {
    If {
        predicate: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Box<Expression>,
    },
}

#[derive(Debug, Clone)]
pub enum ProductIndex {
    Tuple(usize),
    Struct(Identifier),
}

#[derive(Debug, Clone)]
pub enum Constant {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub enum Product {
    Tuple(Vec<Expression>),
    Struct {
        bindings: HashMap<Identifier, Expression>,
    },
}
