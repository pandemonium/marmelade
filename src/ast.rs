use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
};

use crate::{
    interpreter::DependencyGraph,
    lexer::{self, SourceLocation},
    parser::ParsingInfo,
    typer::{
        CoproductType, ProductType, Type, TypeError, TypeParameter, TypeScheme, Typing,
        TypingContext,
    },
};

#[derive(Debug)]
pub struct LibraryDeclarator<A> {
    pub modules: Vec<ModuleDeclarator<A>>,
    pub main: ModuleDeclarator<A>,
}

#[derive(Debug)]
pub enum CompilationUnit<A> {
    Implicit(A, ModuleDeclarator<A>),
    Library(A, LibraryDeclarator<A>),
}
impl<A> CompilationUnit<A>
where
    A: fmt::Debug + Clone,
{
    pub fn map<B>(self, f: fn(A) -> B) -> CompilationUnit<B> {
        match self {
            Self::Implicit(a, module) => CompilationUnit::Implicit(f(a), module.map(f)),
            Self::Library(a, LibraryDeclarator { mut modules, main }) => CompilationUnit::Library(
                f(a),
                LibraryDeclarator {
                    modules: modules.drain(..).map(|m| m.map(f)).collect(),
                    main: main.map(f),
                },
            ),
        }
    }
}

impl<A> fmt::Display for CompilationUnit<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Implicit(_, module) => writeln!(f, "{module}"),
            Self::Library(_, LibraryDeclarator { modules, main }) => {
                for m in modules {
                    writeln!(f, "{m}")?;
                }
                write!(f, "{main}")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModuleDeclarator<A> {
    pub name: Identifier,
    pub declarations: Vec<Declaration<A>>,
    // pub main: Expression,
    // I would like this, but I have to think a little more about it
}

impl<A> ModuleDeclarator<A>
where
    A: fmt::Debug + Clone,
{
    pub fn find_value_declaration<'a>(&'a self, id: &'a Identifier) -> Option<&'a Declaration<A>> {
        self.declarations
            .iter()
            .find(|decl| matches!(decl, Declaration::Value(_, value) if &value.binder == id))
    }

    pub fn dependency_graph(&self) -> DependencyGraph {
        DependencyGraph::from_declarations(&self.declarations)
    }

    fn map<B>(mut self, f: fn(A) -> B) -> ModuleDeclarator<B> {
        ModuleDeclarator {
            name: self.name,
            declarations: self.declarations.drain(..).map(|d| d.map(f)).collect(),
        }
    }
}

impl<A> fmt::Display for ModuleDeclarator<A>
where
    A: fmt::Display,
{
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
pub enum Identifier {
    Atom(String),
    Select(Box<Identifier>, String),
}

impl Identifier {
    pub fn new(x: &str) -> Self {
        Self::Atom(x.to_owned())
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Atom(this) => this.to_owned(),
            Self::Select(parent, this) => format!("{}::{this}", parent.as_str()),
        }
    }

    pub fn scoped_with(&self, scope: &str) -> Self {
        Self::Select(Self::Atom(scope.to_owned()).into(), self.as_str())
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct ValueDeclaration<A> {
    pub binder: Identifier,
    pub type_signature: Option<TypeSignature<A>>,
    pub declarator: ValueDeclarator<A>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeSignature<A> {
    pub quantifier: Option<UniversalQuantifier>,
    pub body: TypeExpression<A>,
}

impl<A> TypeSignature<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> TypeSignature<B> {
        TypeSignature {
            quantifier: self.quantifier,
            body: self.body.map(f),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeDeclaration<A> {
    pub binding: Identifier,
    pub declarator: TypeDeclarator<A>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportModule {
    pub exported_symbols: Vec<Identifier>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Declaration<A> {
    Value(A, ValueDeclaration<A>),
    Type(A, TypeDeclaration<A>),
    Module(A, ModuleDeclarator<A>),
    ImportModule(A, ImportModule),
    // Use()    ??
}

impl Declaration<ParsingInfo> {
    pub fn position(&self) -> &SourceLocation {
        self.parsing_info().location()
    }

    pub fn parsing_info(&self) -> &ParsingInfo {
        match self {
            Self::Value(annotation, _)
            | Self::Type(annotation, _)
            | Self::Module(annotation, _)
            | Self::ImportModule(annotation, _) => annotation,
        }
    }
}

impl<A> Declaration<A>
where
    A: fmt::Debug + Clone,
{
    pub fn map<B>(self, f: fn(A) -> B) -> Declaration<B> {
        match self {
            Self::Value(
                a,
                ValueDeclaration {
                    binder,
                    declarator,
                    type_signature,
                },
            ) => Declaration::Value(
                f(a),
                ValueDeclaration {
                    binder,
                    type_signature: type_signature.map(|signature| signature.map(f)),
                    declarator: declarator.map(f),
                },
            ),
            Self::Type(
                a,
                TypeDeclaration {
                    binding,
                    declarator,
                },
            ) => Declaration::Type(
                f(a),
                TypeDeclaration {
                    binding,
                    declarator: declarator.map(f),
                },
            ),
            Self::Module(a, declarator) => Declaration::Module(f(a), declarator.map(f)),
            Self::ImportModule(a, ImportModule { exported_symbols }) => {
                Declaration::ImportModule(f(a), ImportModule { exported_symbols })
            }
        }
    }
}

impl<A> fmt::Display for Declaration<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(
                _,
                ValueDeclaration {
                    binder, declarator, ..
                },
            ) => write!(f, "{binder} = {declarator}"),
            Self::Type(
                _,
                TypeDeclaration {
                    binding,
                    declarator,
                    ..
                },
            ) => write!(f, "{binding} = {declarator}"),
            Self::Module(_, module) => write!(f, "{module}"),
            Self::ImportModule(
                _,
                ImportModule {
                    exported_symbols, ..
                },
            ) => {
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

#[derive(Clone, Debug, PartialEq, Default)]
pub struct UniversalQuantifier(pub Vec<TypeName>);

impl UniversalQuantifier {
    pub fn add(self, quantifier: TypeName) -> Self {
        let Self(mut quantifiers) = self;
        quantifiers.push(quantifier);
        Self(quantifiers)
    }

    pub fn type_variables(&self) -> &[TypeName] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Coproduct<A> {
    pub forall: UniversalQuantifier,
    pub constructors: Vec<Constructor<A>>,
}

impl<A> Coproduct<A>
where
    A: Clone,
{
    pub fn map<B>(self, f: fn(A) -> B) -> Coproduct<B> {
        let Self {
            forall,
            mut constructors,
        } = self;
        Coproduct {
            forall,
            constructors: constructors.drain(..).map(|c| c.map(f)).collect(),
        }
    }

    pub fn make_implementation_module(
        &self,
        annotation: A,
        self_name: TypeName,
    ) -> Typing<CoproductModule<A>> {
        let declaring_type = self.synthesize_type()?;
        println!("make_implementation_module: {declaring_type}");

        Ok(CoproductModule {
            name: self_name.clone(),
            declaring_type,
            constructors: self
                .constructors
                .iter()
                .map(|constructor| constructor.make_function(annotation.clone(), self_name.clone()))
                .collect(),
        })
    }

    fn synthesize_type(&self) -> Typing<TypeScheme> {
        // This maps names to type parameters
        let mut universals = HashMap::default();
        let coproduct_type = self.synthesize_coproduct_type(&mut universals);
        let superfluous = self
            .forall
            .type_variables()
            .iter()
            .filter(|&var| !universals.contains_key(var.as_str()))
            .cloned()
            .collect::<Vec<_>>();

        if superfluous.is_empty() {
            self.make_type_scheme(universals, coproduct_type)
        } else {
            // Only reports the first
            Err(TypeError::SuperfluousQuantification {
                quantifier: superfluous.into_iter().next().expect("safe"),
                in_type: coproduct_type.clone(),
            })
        }
    }

    fn make_type_scheme(
        &self,
        universals: HashMap<String, TypeParameter>,
        body: Type,
    ) -> Typing<TypeScheme> {
        let mut boofer = vec![];
        for var in self.forall.type_variables() {
            let param =
                universals
                    .get(var.as_str())
                    .ok_or_else(|| TypeError::UndefinedQuantifier {
                        quantifier: var.clone(),
                        in_type: body.clone(),
                    })?;

            boofer.push(param.clone());
        }

        Ok(TypeScheme::new(boofer.as_slice(), body))
    }

    fn synthesize_coproduct_type(&self, universals: &mut HashMap<String, TypeParameter>) -> Type {
        Type::Coproduct(CoproductType::new(
            self.constructors
                .iter()
                .map(|Constructor { name, signature }| {
                    let tuple_signature = signature
                        .iter()
                        .map(|expr| expr.synthesize_type(universals))
                        .collect();
                    (
                        name.as_str().to_owned(),
                        Type::Product(ProductType::Tuple(tuple_signature)),
                    )
                })
                .collect(),
        ))
    }
}

pub struct CoproductModule<A> {
    pub name: TypeName,
    pub declaring_type: TypeScheme,
    pub constructors: Vec<ValueDeclaration<A>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Struct<A>(pub Vec<StructField<A>>);

impl<A> Struct<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Struct<B> {
        let Self(mut xs) = self;
        Struct(xs.drain(..).map(|x| x.map(f)).collect())
    }
}

// Introduce sub-constructors here
#[derive(Clone, Debug, PartialEq)]
pub enum TypeDeclarator<A> {
    Alias(A, TypeExpression<A>),
    Coproduct(A, Coproduct<A>),
    Struct(A, Struct<A>),
}

impl<A> TypeDeclarator<A>
where
    A: Clone,
{
    fn map<B>(self, f: fn(A) -> B) -> TypeDeclarator<B> {
        match self {
            Self::Alias(a, alias) => TypeDeclarator::Alias(f(a), alias.map(f)),
            Self::Coproduct(a, coproduct) => TypeDeclarator::Coproduct(f(a), coproduct.map(f)),
            Self::Struct(a, record) => TypeDeclarator::Struct(f(a), record.map(f)),
        }
    }

    pub fn synthesize_type(&self) -> Typing<TypeScheme> {
        match self {
            Self::Alias(..) => todo!(),
            Self::Coproduct(_, coproduct) => coproduct.synthesize_type(),
            Self::Struct(..) => todo!(),
        }
    }
}

impl<A> fmt::Display for TypeDeclarator<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alias(_, type_expr) => {
                write!(f, "{type_expr}")
            }
            Self::Coproduct(_, coproduct) => {
                for c in &coproduct.constructors {
                    write!(f, "{c}")?;
                }
                Ok(())
            }
            Self::Struct(_, Struct(fields)) => {
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

#[derive(Clone, Debug, PartialEq)]
pub struct StructField<A> {
    pub name: Identifier,
    pub type_annotation: TypeExpression<A>,
}

impl<A> StructField<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> StructField<B> {
        StructField {
            name: self.name,
            type_annotation: self.type_annotation.map(f),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Constructor<A> {
    pub name: Identifier,
    pub signature: Vec<TypeExpression<A>>,
}

impl<A> Constructor<A>
where
    A: Clone,
{
    fn make_function(&self, annotation: A, ty: TypeName) -> ValueDeclaration<A> {
        let mut type_param_map = HashMap::default();
        let parameters = self
            .signature
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                // I could give this one the real TypeParameter via
                // type_parameters. Then Parameter would hold Type
                // instead of TypeExpression
                Parameter::new_with_type_annotation(
                    Identifier::new(&format!("p{index}")),
                    expr.synthesize_type(&mut type_param_map),
                )
            })
            .collect::<Vec<_>>();

        // It should really annotate it with types to make sure the
        // typer gets it right. But that should not be necessary yet.
        let expression = self.make_injection_lambda_tree(&annotation, ty, parameters.clone());

        ValueDeclaration {
            binder: self.name.clone(),
            // Compute a type signature type expression!
            // I have self.signature which can be joined with -> to make the function, I guess?
            type_signature: None,
            declarator: ValueDeclarator { expression },
        }
    }

    fn make_injection_lambda_tree(
        &self,
        annotation: &A,
        name: TypeName,
        parameters: Vec<Parameter>,
    ) -> Expression<A>
    where
        A: Clone,
    {
        let tuple = Product::Tuple(
            parameters
                .iter()
                .map(|p| Expression::Variable(annotation.clone(), p.name.clone()))
                .collect(),
        );

        let inject = Inject {
            name,
            constructor: self.name.clone(),
            argument: Expression::Product(annotation.clone(), tuple).into(),
        };

        parameters.into_iter().rfold(
            Expression::Inject(annotation.clone(), inject),
            |body, parameter| {
                Expression::Lambda(
                    annotation.clone(),
                    Lambda {
                        parameter,
                        body: body.into(),
                    },
                )
            },
        )
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Constructor<B> {
        Constructor {
            name: self.name,
            signature: self.signature.into_iter().map(|expr| expr.map(f)).collect(),
        }
    }

    // This is the problem. This causes undue instantiations.
    pub fn constructed_type(id: &Identifier, ctx: &TypingContext) -> Option<TypeScheme> {
        fn ultimate_codomain(ty: Type) -> Type {
            match ty {
                Type::Arrow(_, codomain) => ultimate_codomain(*codomain),
                otherwise => otherwise,
            }
        }

        ctx.lookup(&id.clone().into())
            .map(|scheme| scheme.clone().map_body(ultimate_codomain))
    }
}

impl<A> fmt::Display for Constructor<A>
where
    A: fmt::Display,
{
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

// This thing needs positions
// Is this the thing that can be a function type too? Yes.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpression<A> {
    Constant(A, TypeName),
    Parameter(A, TypeName),
    Apply(A, TypeApply<A>),
    Arrow(A, Arrow<A>),
}

impl<A> TypeExpression<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> TypeExpression<B> {
        match self {
            Self::Constant(a, id) => TypeExpression::Constant(f(a), id),
            Self::Parameter(a, id) => TypeExpression::Parameter(f(a), id),
            Self::Apply(a, apply) => TypeExpression::Apply(f(a), apply.map(f)),
            Self::Arrow(a, arrow) => TypeExpression::Arrow(f(a), arrow.map(f)),
        }
    }

    // Make a thing to capture this type_param_map later.
    pub fn synthesize_type(&self, type_param_map: &mut HashMap<String, TypeParameter>) -> Type {
        fn map_expression<A>(
            expr: &TypeExpression<A>,
            type_params: &mut HashMap<String, TypeParameter>,
        ) -> Type {
            match expr {
                TypeExpression::Constant(_, name) => Type::Named(name.to_owned()),
                TypeExpression::Parameter(_, TypeName(param)) => Type::Parameter(
                    *type_params
                        .entry(param.to_owned())
                        .or_insert_with(TypeParameter::fresh),
                ),
                TypeExpression::Apply(_, node) => Type::Apply(
                    map_expression(&node.constructor, type_params).into(),
                    map_expression(&node.argument, type_params).into(),
                ),
                TypeExpression::Arrow(_, arrow) => Type::Arrow(
                    map_expression(&arrow.domain, type_params).into(),
                    map_expression(&arrow.codomain, type_params).into(),
                ),
            }
        }

        map_expression(self, type_param_map)
    }
}

impl<A> fmt::Display for TypeExpression<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(_, id) => write!(f, "{id}"),
            Self::Parameter(_, id) => write!(f, "{id}"),
            Self::Apply(_, apply) => write!(f, "{apply}"),
            Self::Arrow(_, arrow) => write!(f, "{arrow}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeApply<A> {
    pub constructor: Box<TypeExpression<A>>,
    pub argument: Box<TypeExpression<A>>,
}

impl<A> TypeApply<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> TypeApply<B> {
        TypeApply {
            constructor: self.constructor.map(f).into(),
            argument: self.argument.map(f).into(),
        }
    }
}

impl<A> fmt::Display for TypeApply<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            constructor,
            argument,
        } = self;
        write!(f, "{constructor} {argument}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arrow<A> {
    pub domain: Box<TypeExpression<A>>,
    pub codomain: Box<TypeExpression<A>>,
}

impl<A> Arrow<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Arrow<B> {
        Arrow {
            domain: self.domain.map(f).into(),
            codomain: self.codomain.map(f).into(),
        }
    }
}

impl<A> fmt::Display for Arrow<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { domain, codomain } = self;
        write!(f, "{domain} -> {codomain}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueDeclarator<A> {
    pub expression: Expression<A>,
}

impl<A> ValueDeclarator<A>
where
    A: fmt::Debug + Clone,
{
    pub fn dependencies(&self) -> HashSet<&Identifier> {
        self.expression.free_identifiers()
    }

    fn map<B>(self, f: fn(A) -> B) -> ValueDeclarator<B> {
        ValueDeclarator {
            expression: self.expression.map(f),
        }
    }
}

impl<A> fmt::Display for ValueDeclarator<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)
    }
}

// these can be pattern matches too
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: Identifier,
    pub type_annotation: Option<Type>,
}

impl Parameter {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            type_annotation: None,
        }
    }

    pub fn new_with_type_annotation(name: Identifier, ty: Type) -> Self {
        Self {
            name,
            type_annotation: Some(ty),
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
    InvokeBridge(A, Identifier),
    Literal(A, Constant),
    SelfReferential(A, SelfReferential<A>),
    Lambda(A, Lambda<A>),
    Apply(A, Apply<A>),
    Inject(A, Inject<A>),
    Product(A, Product<A>),
    Project(A, Project<A>),
    Binding(A, Binding<A>),
    Sequence(A, Sequence<A>),
    ControlFlow(A, ControlFlow<A>),
    DeconstructInto(A, DeconstructInto<A>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeconstructInto<A> {
    pub scrutinee: Box<Expression<A>>,
    pub match_clauses: Vec<MatchClause<A>>,
}

impl<A> DeconstructInto<A> {
    pub fn map<B>(mut self, f: fn(A) -> B) -> DeconstructInto<B> {
        DeconstructInto {
            scrutinee: self.scrutinee.map(f).into(),
            match_clauses: self
                .match_clauses
                .drain(..)
                .map(|clause| clause.map(f))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchClause<A> {
    pub pattern: Pattern<A>,
    pub consequent: Box<Expression<A>>,
}

impl<A> MatchClause<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> MatchClause<B> {
        MatchClause {
            pattern: self.pattern.map(f),
            consequent: self.consequent.map(f).into(),
        }
    }

    fn find_unbound<'a>(
        &'a self,
        bound: &mut HashSet<&'a Identifier>,
        free: &mut HashSet<&'a Identifier>,
    ) {
        let bindings = self.pattern.bindings();
        bound.extend(bindings);
        self.consequent.find_unbound(bound, free);
        free.extend(self.pattern.free_variables());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<A> {
    // First argument should be A, for god's sake.
    Coproduct(A, ConstructorPattern<A>),
    Tuple(A, TuplePattern<A>),
    Literally(Constant),
    Otherwise(Identifier),
}

impl<A> Pattern<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Pattern<B> {
        match self {
            Self::Coproduct(a, pattern) => Pattern::Coproduct(f(a), pattern.map(f)),
            Self::Tuple(a, pattern) => Pattern::Tuple(f(a), pattern.map(f)),
            Self::Literally(literal) => Pattern::Literally(literal),
            Self::Otherwise(id) => Pattern::Otherwise(id),
        }
    }

    fn bindings<'a>(&'a self) -> Vec<&'a Identifier> {
        match self {
            Self::Coproduct(_, pattern) => pattern
                .argument
                .elements
                .iter()
                .flat_map(|p| p.bindings())
                .collect(),
            Self::Tuple(_, p) => p.elements.iter().flat_map(|p| p.bindings()).collect(),
            Self::Literally(_) => vec![],
            Self::Otherwise(pattern) => vec![pattern],
        }
    }

    fn free_variables(&self) -> HashSet<&Identifier> {
        match self {
            Self::Coproduct(_, pattern) => {
                let mut free = HashSet::from([&pattern.constructor]);
                free.extend(
                    pattern
                        .argument
                        .elements
                        .iter()
                        .flat_map(|p| p.free_variables())
                        .collect::<HashSet<_>>(),
                );
                free
            }
            Self::Tuple(_, pattern) => pattern
                .elements
                .iter()
                .flat_map(|p| p.free_variables())
                .collect(),
            _otherwise => HashSet::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstructorPattern<A> {
    pub constructor: Identifier,
    pub argument: TuplePattern<A>,
}

impl<A> ConstructorPattern<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> ConstructorPattern<B> {
        ConstructorPattern {
            constructor: self.constructor,
            argument: self.argument.map(f),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TuplePattern<A> {
    pub elements: Vec<Pattern<A>>,
}

impl<A> TuplePattern<A> {
    pub fn map<B>(mut self, f: fn(A) -> B) -> TuplePattern<B> {
        TuplePattern {
            elements: self.elements.drain(..).map(|p| p.map(f)).collect(),
        }
    }
}

impl<A> fmt::Display for DeconstructInto<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            scrutinee,
            match_clauses,
        } = self;
        writeln!(f, "match ({scrutinee})")?;
        for clause in match_clauses {
            writeln!(f, "case {} -> {}", clause.pattern, clause.consequent)?;
        }
        Ok(())
    }
}

impl<A> fmt::Display for Pattern<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coproduct(
                _,
                ConstructorPattern {
                    constructor,
                    argument,
                },
            ) => write!(f, "C_{constructor} [{argument}]"),
            Self::Tuple(_, pattern) => write!(f, "T_{pattern}"),
            Self::Literally(literal) => write!(f, "L_{literal}"),
            Self::Otherwise(id) => write!(f, "O_`{id}`"),
        }
    }
}

impl<A> fmt::Display for TuplePattern<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for pattern in &self.elements {
            write!(f, "{pattern}")?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfReferential<A> {
    pub name: Identifier,
    pub parameter: Parameter,
    pub body: Box<Expression<A>>,
}
impl<A> SelfReferential<A> {
    fn map<B>(self, f: fn(A) -> B) -> SelfReferential<B> {
        SelfReferential {
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
        Lambda {
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
        Apply {
            function: self.function.map(f).into(),
            argument: self.argument.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inject<A> {
    pub name: TypeName,
    pub constructor: Identifier,
    pub argument: Box<Expression<A>>,
}
impl<A> Inject<A> {
    fn map<B>(self, f: fn(A) -> B) -> Inject<B> {
        Inject {
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
        Project {
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
        Binding {
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
        Sequence {
            this: self.this.map(f).into(),
            and_then: self.and_then.map(f).into(),
        }
    }
}

impl Expression<ParsingInfo> {
    pub fn position(&self) -> &lexer::SourceLocation {
        &self.parsing_info().position
    }

    pub fn parsing_info(&self) -> &ParsingInfo {
        self.annotation()
    }
}

impl<A> Expression<A> {
    pub fn annotation(&self) -> &A {
        match self {
            Self::Variable(annotation, ..)
            | Self::InvokeBridge(annotation, ..)
            | Self::Literal(annotation, ..)
            | Self::SelfReferential(annotation, ..)
            | Self::Lambda(annotation, ..)
            | Self::Apply(annotation, ..)
            | Self::Inject(annotation, ..)
            | Self::Product(annotation, ..)
            | Self::Project(annotation, ..)
            | Self::Binding(annotation, ..)
            | Self::Sequence(annotation, ..)
            | Self::ControlFlow(annotation, ..)
            | Self::DeconstructInto(annotation, ..) => annotation,
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
            Self::InvokeBridge(_, id) => {
                free.insert(id);
            }
            Self::Lambda(_, Lambda { parameter, body }) => {
                // This is probably not correct
                // I have to remove this after looking in "body
                bound.insert(&parameter.name);
                body.find_unbound(bound, free);
            }
            Self::SelfReferential(
                _,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                bound.insert(&parameter.name);
                bound.insert(name);
                body.find_unbound(bound, free);
            }
            Self::Apply(_, Apply { function, argument }) => {
                function.find_unbound(bound, free);
                argument.find_unbound(bound, free);
            }
            Self::Inject(_, inject) => {
                inject.argument.find_unbound(bound, free);
            }
            Self::Product(_, Product::Tuple(expressions)) => {
                for e in expressions {
                    e.find_unbound(bound, free);
                }
            }
            Self::Product(_, Product::Struct(bindings)) => {
                for (_, e) in bindings {
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
            Self::DeconstructInto(_, deconstruct) => {
                deconstruct.scrutinee.find_unbound(bound, free);
                for clause in &deconstruct.match_clauses {
                    clause.find_unbound(bound, free);
                }
            }
            _otherwise => (),
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Expression<B> {
        match self {
            Self::Variable(x, info) => Expression::<B>::Variable(f(x), info),
            Self::InvokeBridge(x, info) => Expression::<B>::InvokeBridge(f(x), info),
            Self::Literal(x, info) => Expression::<B>::Literal(f(x), info),
            Self::SelfReferential(x, info) => Expression::<B>::SelfReferential(f(x), info.map(f)),
            Self::Lambda(x, info) => Expression::<B>::Lambda(f(x), info.map(f)),
            Self::Apply(x, info) => Expression::<B>::Apply(f(x), info.map(f)),
            Self::Inject(x, info) => Expression::<B>::Inject(f(x), info.map(f)),
            Self::Product(x, info) => Expression::<B>::Product(f(x), info.map(f)),
            Self::Project(x, info) => Expression::<B>::Project(f(x), info.map(f)),
            Self::Binding(x, info) => Expression::<B>::Binding(f(x), info.map(f)),
            Self::Sequence(x, info) => Expression::<B>::Sequence(f(x), info.map(f)),
            Self::ControlFlow(x, info) => Expression::<B>::ControlFlow(f(x), info.map(f)),
            Self::DeconstructInto(x, info) => Expression::DeconstructInto(f(x), info.map(f)),
        }
    }

    pub fn erase_annotation(self) -> Expression<()> {
        self.map(|_| ())
    }
}

impl<A> fmt::Display for Expression<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(_, id) => write!(f, "{id}"),
            Expression::InvokeBridge(_, id) => write!(f, "call {id}"),
            Expression::Literal(_, c) => write!(f, "{c}"),
            Expression::SelfReferential(_, SelfReferential { name, body, .. }) => {
                write!(f, "{name}->[{body}]")
            }
            Expression::Lambda(_, Lambda { parameter, body }) => {
                write!(f, "lambda \\{parameter}. {body}")
            }
            Expression::Apply(_, Apply { function, argument }) => {
                write!(f, "{function} {argument}")
            }
            Expression::Inject(
                _,
                Inject {
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
            Expression::DeconstructInto(_, deconstruct) => writeln!(f, "{deconstruct}"),
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

impl<A> fmt::Display for ControlFlow<A>
where
    A: fmt::Display,
{
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
    Struct(Vec<(Identifier, Expression<A>)>),
}
impl<A> Product<A> {
    fn map<B>(self, f: fn(A) -> B) -> Product<B> {
        match self {
            Self::Tuple(mut expressions) => {
                Product::Tuple(expressions.drain(..).map(|x| x.map(f)).collect())
            }
            Self::Struct(mut bindings) => {
                Product::<B>::Struct(bindings.drain(..).map(|(k, v)| (k, v.map(f))).collect())
            }
        }
    }
}

impl<A> fmt::Display for Product<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuple(expressions) => {
                write!(f, "T(")?;
                for e in expressions {
                    write!(f, "{e},")?;
                }
                write!(f, ")")
            }
            Self::Struct(bindings) => {
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
    use crate::ast::ValueDeclaration;

    use super::{Constant, Declaration, Expression, Identifier, ModuleDeclarator, ValueDeclarator};

    #[test]
    fn cyclic_dependencies() {
        // I should parse text instead of this
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("foo")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("quux")),
                        },
                    },
                ),
            ],
        };

        // Still broken
        assert!(!m.dependency_graph().is_acyclic());
    }

    #[test]
    fn satisfiable() {
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("frobnicator")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Literal((), Constant::Int(1)),
                        },
                    },
                ),
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(dep_mat.is_satisfiable(|_| false));
    }

    #[test]
    fn unsatisfiable() {
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("frobnicator")),
                        },
                    },
                ),
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(!dep_mat.is_satisfiable(|_| false));
    }

    //    #[test]
    fn _top_of_the_day() {
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("bar")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable((), Identifier::new("frobnicator")),
                        },
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Literal((), Constant::Int(1)),
                        },
                    },
                ),
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
