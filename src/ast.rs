use std::{
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    path::Display,
};

use crate::{
    interpreter::{DependencyGraph, Value},
    lexer::{self, Literal, SourcePosition},
    parser::ParsingInfo,
    typer::{CoproductType, ProductType, Type, TypeError, TypeParameter, TypeScheme, Typing},
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
    A: Clone,
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
    A: Clone,
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
    pub declarator: ValueDeclarator<A>,
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
    pub fn position(&self) -> &SourcePosition {
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
    A: Clone,
{
    pub fn map<B>(self, f: fn(A) -> B) -> Declaration<B> {
        match self {
            Self::Value(a, ValueDeclaration { binder, declarator }) => Declaration::Value(
                f(a),
                ValueDeclaration {
                    binder,
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
pub struct Forall(pub Vec<TypeName>);

impl Forall {
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
    pub forall: Forall,
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
        Ok(CoproductModule {
            name: self_name.clone(),
            declaring_type: self.synthesize_type()?,
            constructors: self
                .constructors
                .iter()
                .map(|constructor| {
                    constructor.make_function(annotation.clone(), self_name.clone(), vec![])
                })
                .collect(),
        })
    }

    fn synthesize_type(&self) -> Typing<TypeScheme> {
        let mut universals = HashMap::default();
        let coproduct_type = self.synthesize_coproduct_type(&mut universals);
        let mut superfluous = self
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
                quantifier: superfluous.drain(..).next().expect("safe"),
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
    // Think about whether or not this can consume self.
    fn make_function(
        &self,
        annotation: A,
        ty: TypeName,
        _type_parameters: Vec<TypeName>, // this is the forall clause, really
    ) -> ValueDeclaration<A> {
        let parameters = self
            .signature
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                Parameter::new_with_type_annotation(
                    Identifier::new(&format!("p{index}")),
                    expr.clone(),
                )
            })
            .collect::<Vec<_>>();

        let node = self.make_inject_node(&annotation, ty, parameters.clone());

        // Hold off on this for a while
        //        if parameters.is_empty() {
        //            parameters.push(Parameter::new_with_type_annotation(
        //                Identifier::new(&format!("p0")),
        //                TypeExpression::Constant(TypeName::new("builtin::Unit")),
        //            ));
        //        }

        // Think about how many annotations this contains. Are they all needed
        // because they are all on different positions?
        let function = FunctionDeclarator {
            parameters: parameters.clone(),
            return_type_annotation: None, //todo!(),
            body: Expression::Inject(annotation.clone(), node),
        };

        ValueDeclaration {
            binder: self.name.clone(),
            declarator: ValueDeclarator::Function(function),
        }
    }

    fn make_inject_node(
        &self,
        annotation: &A,
        name: TypeName,
        mut parameters: Vec<Parameter<A>>,
    ) -> Inject<A>
    where
        A: Clone,
    {
        // Is this where it goes wrong? The second element
        // in the Cons is a List[a] which it is currently
        // in the process of typing.
        let tuple = Product::Tuple(
            parameters
                .drain(..)
                .map(|p| Expression::Variable(annotation.clone(), p.name))
                .collect(),
        );

        Inject {
            name,
            constructor: self.name.clone(),
            argument: Expression::Product(annotation.clone(), tuple).into(),
        }
    }

    pub fn map<B>(mut self, f: fn(A) -> B) -> Constructor<B> {
        Constructor {
            name: self.name,
            signature: self.signature.drain(..).map(|expr| expr.map(f)).collect(),
        }
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
#[derive(Clone, Debug, PartialEq)]
pub enum TypeExpression<A> {
    Constant(TypeName),
    Parameter(TypeName),
    Apply(TypeApply<A>, PhantomData<A>),
}

impl<A> TypeExpression<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> TypeExpression<B> {
        match self {
            Self::Constant(id) => TypeExpression::Constant(id),
            Self::Parameter(id) => TypeExpression::Parameter(id),
            Self::Apply(apply, ..) => TypeExpression::Apply(apply.map(f), PhantomData::default()),
        }
    }

    // Make a thing to capture this type_param_map later.
    pub fn synthesize_type(&self, type_param_map: &mut HashMap<String, TypeParameter>) -> Type {
        fn map_expression<A>(
            expr: &TypeExpression<A>,
            type_params: &mut HashMap<String, TypeParameter>,
        ) -> Type {
            match expr {
                TypeExpression::Constant(name) => Type::Named(name.to_owned()),
                TypeExpression::Parameter(TypeName(param)) => Type::Parameter(
                    *type_params
                        .entry(param.to_owned())
                        .or_insert_with(TypeParameter::fresh),
                ),
                TypeExpression::Apply(node, ..) => Type::Apply(
                    map_expression(&node.constructor, type_params).into(),
                    map_expression(&node.argument, type_params).into(),
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
            Self::Constant(id) => write!(f, "{id}"),
            Self::Parameter(id) => write!(f, "{id}"),
            Self::Apply(apply, ..) => write!(f, "{apply}"),
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
            Self::Constant(constant) => ValueDeclarator::Constant(constant.map(f)),
            Self::Function(function) => ValueDeclarator::Function(function.map(f)),
        }
    }
}

impl<A> fmt::Display for ValueDeclarator<A>
where
    A: fmt::Display,
{
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
        ConstantDeclarator {
            initializer: self.initializer.map(f),
            type_annotation: self.type_annotation,
        }
    }
}

impl<A> fmt::Display for ConstantDeclarator<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            initializer,
            type_annotation,
        } = self;
        write!(f, "{initializer}")?;
        if let Some(ty) = type_annotation {
            write!(f, "[{ty}]")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDeclarator<A> {
    pub parameters: Vec<Parameter<A>>,
    pub return_type_annotation: Option<TypeName>, // TypeExpression instead
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

    fn map<B>(mut self, f: fn(A) -> B) -> FunctionDeclarator<B> {
        FunctionDeclarator {
            parameters: self.parameters.drain(..).map(|p| p.map(f)).collect(),
            return_type_annotation: self.return_type_annotation,
            body: self.body.map(f),
        }
    }
}

impl<A> fmt::Display for FunctionDeclarator<A>
where
    A: fmt::Display,
{
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
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter<A> {
    pub name: Identifier,
    pub type_annotation: Option<TypeExpression<A>>,
}

impl<A> Parameter<A> {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            type_annotation: None,
        }
    }

    pub fn new_with_type_annotation(name: Identifier, ty: TypeExpression<A>) -> Self {
        Self {
            name,
            type_annotation: Some(ty),
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Parameter<B> {
        Parameter {
            name: self.name,
            type_annotation: self.type_annotation.map(|t| t.map(f)),
        }
    }
}

impl<A> fmt::Display for Parameter<A>
where
    A: fmt::Display,
{
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<A> {
    // First argument should be A, for god's sake.
    Coproduct(ConstructorPattern<A>, PhantomData<A>),
    Tuple(TuplePattern<A>, PhantomData<A>),
    Literally(Constant),
    Otherwise(Identifier),
}

impl<A> Pattern<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Pattern<B> {
        match self {
            Self::Coproduct(pattern, _) => {
                Pattern::Coproduct(pattern.map(f), PhantomData::default())
            }
            Self::Tuple(pattern, _) => Pattern::Tuple(pattern.map(f), PhantomData::default()),
            Self::Literally(literal) => Pattern::Literally(literal),
            Self::Otherwise(id) => Pattern::Otherwise(id),
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
                ConstructorPattern {
                    constructor,
                    argument,
                },
                _,
            ) => write!(f, "{constructor} ({argument})"),
            Self::Tuple(pattern, _) => write!(f, "{pattern}"),
            Self::Literally(literal) => write!(f, "{literal}"),
            Self::Otherwise(id) => write!(f, "{id}"),
        }
    }
}

impl<A> fmt::Display for TuplePattern<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for pattern in &self.elements {
            write!(f, "{pattern}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfReferential<A> {
    pub name: Identifier,
    pub parameter: Parameter<A>,
    pub body: Box<Expression<A>>,
}
impl<A> SelfReferential<A> {
    fn map<B>(self, f: fn(A) -> B) -> SelfReferential<B> {
        SelfReferential {
            name: self.name,
            parameter: self.parameter.map(f),
            body: self.body.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lambda<A> {
    pub parameter: Parameter<A>,
    pub body: Box<Expression<A>>,
}
impl<A> Lambda<A> {
    fn map<B>(self, f: fn(A) -> B) -> Lambda<B> {
        Lambda {
            parameter: self.parameter.map(f),
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
            Self::Apply(_, Apply { function, argument }) => {
                function.find_unbound(bound, free);
                argument.find_unbound(bound, free);
            }
            Self::Inject(_, Inject { argument, .. }) => argument.find_unbound(bound, free),
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

    use super::{
        Constant, ConstantDeclarator, Declaration, Expression, Identifier, ModuleDeclarator,
        ValueDeclarator,
    };

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
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("foo")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("quux")),
                            type_annotation: None,
                        }),
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
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("frobnicator")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Literal((), Constant::Int(1)),
                            type_annotation: None,
                        }),
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
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("frobnicator")),
                            type_annotation: None,
                        }),
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
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("bar")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Variable((), Identifier::new("frobnicator")),
                            type_annotation: None,
                        }),
                    },
                ),
                Declaration::Value(
                    (),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        declarator: ValueDeclarator::Constant(ConstantDeclarator {
                            initializer: Expression::Literal((), Constant::Int(1)),
                            type_annotation: None,
                        }),
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
