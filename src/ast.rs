use std::{
    collections::{HashMap, HashSet, VecDeque},
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

impl fmt::Display for CompilationUnit {
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

    pub fn dependency_graph(&self) -> DependencyGraph {
        DependencyGraph::from_declarations(&self.declarations)
    }
}

impl fmt::Display for ModuleDeclarator {
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

#[derive(Debug)]
pub struct DependencyGraph<'a> {
    dependencies: HashMap<&'a Identifier, Vec<&'a Identifier>>,
}

impl<'a> DependencyGraph<'a> {
    pub fn from_declarations(decls: &'a [Declaration]) -> Self {
        let mut outbound = HashMap::with_capacity(decls.len());

        for decl in decls {
            match decl {
                Declaration::Value {
                    binder, declarator, ..
                } => {
                    outbound.insert(binder, declarator.dependencies().into_iter().collect());
                }
                Declaration::ImportModule {
                    exported_symbols, ..
                } => {
                    println!("from_declarations: {exported_symbols:?}");

                    for dep in exported_symbols {
                        outbound.entry(dep).or_default();
                    }
                }
                _otherwise => (),
            }
        }

        Self {
            dependencies: outbound,
        }
    }

    // Think about whether or not this consumes self.
    pub fn compute_resolution_order(&self) -> Vec<&'a Identifier> {
        let mut boofer = Vec::with_capacity(self.dependencies.len());

        fn exclude_self_referentials<'a>(
            node: &'a Identifier,
            edges: &Vec<&'a Identifier>,
        ) -> HashSet<&'a Identifier> {
            edges
                .iter()
                .filter(|&&edge| edge != node)
                .map(|&x| x)
                .collect::<HashSet<_>>()
        }

        let mut graph = self
            .dependencies
            .iter()
            .map(|(&node, edges)| (node, exclude_self_referentials(node, edges)))
            .collect::<Vec<_>>();

        // Look in to doing away with this go-between structure and make the lookups
        // directly in graph instead
        let mut in_degrees = graph
            .iter()
            .map(|(node, edges)| (*node, edges.len()))
            .collect::<HashMap<_, _>>();

        let mut queue = Vec::with_capacity(in_degrees.len());

        loop {
            let independents = in_degrees
                .iter()
                .filter_map(|(&node, edges)| (*edges == 0).then_some(node))
                .collect::<Vec<_>>();

            if independents.is_empty() {
                // if in_degrees is not empty here, then there were cycles.
                println!("compute_resolution_order: {in_degrees:?}");
                break;
            }

            queue.extend(independents);

            for independent in queue.drain(..) {
                for (node, edges) in graph.iter_mut() {
                    if edges.remove(independent) {
                        if let Some(v) = in_degrees.get_mut(node) {
                            *v -= 1;
                        }
                    }
                }

                in_degrees.remove(independent);
                boofer.push(independent);
            }
        }

        boofer
    }

    pub fn is_wellformed<F>(&'a self, is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        /*self.is_acyclic() &&*/
        self.is_satisfiable(is_external)
    }

    pub fn is_satisfiable<F>(&'a self, mut is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        self.dependencies.values().all(|deps| {
            deps.iter().all(|dep| {
                let retval = self.dependencies.contains_key(dep) || is_external(dep);

                if !retval {
                    println!("is_satisfiable: `{dep}` not found");
                    println!("is_satisfiable: {:?}", self.dependencies)
                }

                retval
            })
        })
    }

    pub fn is_acyclic(&self) -> bool {
        let mut visited = HashSet::new();

        self.dependencies
            .keys()
            .all(|child| !self.is_cyclic(child, &mut visited, &mut HashSet::new()))
    }

    fn is_cyclic(
        &self,
        node: &'a Identifier,
        seen: &mut HashSet<&'a Identifier>,
        path: &mut HashSet<&'a Identifier>,
    ) -> bool {
        if path.contains(node) {
            path.len() > 1
        //            true
        } else if seen.contains(node) {
            false
        } else {
            path.insert(node);
            seen.insert(node);

            let has_cycle = self
                .dependencies(node)
                .unwrap_or_default()
                .iter()
                .any(|&child| self.is_cyclic(child, seen, path));

            path.remove(node);

            has_cycle
        }
    }

    pub fn nodes(&self) -> Vec<&'a Identifier> {
        self.dependencies.keys().cloned().collect()
    }

    pub fn find<F>(&'a self, mut p: F) -> Option<&'a &'a Identifier>
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies.keys().find(|id| p(id))
    }

    pub fn satisfies<F>(&'a self, mut p: F) -> bool
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies.keys().all(|id| p(id))
    }

    pub fn dependencies(&self, d: &'a Identifier) -> Option<&[&'a Identifier]> {
        self.dependencies.get(d).map(Vec::as_slice)
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

impl fmt::Display for Declaration {
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
pub enum ValueDeclarator {
    Constant(ConstantDeclarator),
    Function(FunctionDeclarator),
}

impl ValueDeclarator {
    pub fn dependencies(&self) -> HashSet<&Identifier> {
        let mut free = match self {
            Self::Constant(decl) => decl.free_identifiers(),
            Self::Function(decl) => decl.free_identifiers(),
        };
        free.drain().collect()
    }
}

impl fmt::Display for ValueDeclarator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(constant) => write!(f, "{constant}"),
            Self::Function(function) => write!(f, "{function}"),
        }
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

impl fmt::Display for ConstantDeclarator {
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
pub struct FunctionDeclarator {
    pub parameters: Vec<Parameter>,
    pub return_type_annotation: Option<TypeName>,
    pub body: Expression,
}

impl FunctionDeclarator {
    // does this function really go here?
    pub fn into_lambda_tree(self, self_name: Identifier) -> Expression {
        self.parameters
            .into_iter()
            .rev()
            .fold(self.body, |body, parameter| Expression::SelfReferential {
                name: self_name.clone(),
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

impl fmt::Display for FunctionDeclarator {
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
pub enum Expression {
    Variable(Identifier),
    CallBridge(Identifier),
    Literal(Constant),
    SelfReferential {
        name: Identifier,
        parameter: Parameter,
        body: Box<Expression>,
    },
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
            Self::CallBridge(id) => {
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

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(id) => write!(f, "{id}"),
            Expression::CallBridge(id) => write!(f, "call {id}"),
            Expression::Literal(c) => write!(f, "{c}"),
            Expression::SelfReferential { name, body, .. } => write!(f, "-----> {name}->[{body}]"),
            Expression::Lambda { parameter, body } => write!(f, "lambda \\{parameter}. {body}"),
            Expression::Apply { function, argument } => write!(f, "{function} {argument}"),
            Expression::Construct {
                name,
                constructor,
                argument,
            } => write!(f, "{name}::{constructor} {argument}"),
            Expression::Product(product) => write!(f, "{product}"),
            Expression::Project { base, index } => write!(f, "{base}.{index}"),
            Expression::Binding {
                binder,
                bound,
                body,
                ..
            } => write!(f, "let {binder} = {bound} in {body}"),
            Expression::Sequence { this, and_then } => writeln!(f, "{this}\n{and_then}"),
            Expression::ControlFlow(control) => writeln!(f, "{control}"),
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

impl fmt::Display for ControlFlow {
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
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::Text(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{x}"),
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
pub enum Product {
    Tuple(Vec<Expression>),
    Struct {
        bindings: HashMap<Identifier, Expression>,
    },
}

impl fmt::Display for Product {
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

        // Still broken
        assert!(!m.dependency_graph().is_acyclic());
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

        let dep_mat = m.dependency_graph();
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

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(!dep_mat.is_satisfiable(|_| false));
    }

    #[test]
    fn top_of_the_day() {
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
