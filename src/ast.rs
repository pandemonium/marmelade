use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fmt,
    hash::Hash,
    mem,
};

use crate::{
    interpreter::DependencyGraph,
    lexer::{self, SourceLocation},
    parser::ParsingInfo,
    typer::{
        CoproductType, EmptyAnnotation, Parsed, ProductType, TupleType, Type, TypeError,
        TypeParameter, TypeScheme, Typing, TypingContext,
    },
};

mod names;

#[derive(Debug)]
pub struct CompilationUnit<A> {
    pub annotation: A,
    pub main: ModuleDeclarator<A>,
}

impl<A> CompilationUnit<A>
where
    A: fmt::Display + fmt::Debug + Copy + Parsed,
{
    pub fn map<B>(self, f: fn(A) -> B) -> CompilationUnit<B> {
        let Self { annotation, main } = self;
        CompilationUnit {
            annotation: f(annotation),
            main: main.map(f),
        }
    }
}

impl<A> fmt::Display for CompilationUnit<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { annotation, main } = self;
        writeln!(f, "Compilation unit {annotation}, main: {main}")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModuleDeclarator<A> {
    pub name: Identifier,
    pub declarations: Vec<Declaration<A>>,
}

impl<A> ModuleDeclarator<A>
where
    A: fmt::Display + fmt::Debug + Copy + Parsed,
{
    pub fn with_scoped_names(self) -> Self {
        let name = self.name.clone();
        self.prefixed_with(name)
    }

    pub fn prefixed_with(self, name: Identifier) -> Self {
        Self {
            name: self.name.prefixed_with(name.clone()),
            declarations: self
                .declarations
                .into_iter()
                .map(|decl| decl.prefixed_with(name.clone()))
                .collect(),
        }
    }

    pub fn find_value_declaration<'a>(
        &'a self,
        id: &'a Identifier,
    ) -> Option<&'a ValueDeclaration<A>> {
        self.declarations.iter().find_map(|decl| {
            if let Declaration::Value(_, value) = decl {
                (&value.binder == id).then_some(value)
            } else {
                None
            }
        })
    }

    pub fn dependency_graph(&self) -> DependencyGraph {
        DependencyGraph::from_declarations(&self.declarations)
    }

    fn map<B>(self, f: fn(A) -> B) -> ModuleDeclarator<B> {
        ModuleDeclarator {
            name: self.name,
            declarations: self.declarations.into_iter().map(|d| d.map(f)).collect(),
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

// Maybe this thing should carry a SourceLocation
// I also would like this to contain the identifier from the parse
// although that might be impossible since some names are static.
// Or is it?
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

    pub fn mangle(&self, prefix: &str) -> Self {
        match self {
            Self::Atom(id) => Self::Atom(format!("{prefix}${id}")),
            Self::Select(parent, id) => {
                Self::Select((*parent.clone()).into(), format!("{prefix}${id}"))
            }
        }
    }

    pub fn head(&self) -> &Self {
        match self {
            Self::Atom(..) => self,
            Self::Select(prefix, _) => prefix.head(),
        }
    }

    pub fn prefixed_with(&self, prefix: Identifier) -> Self {
        fn splice_prefix(id: Identifier, prefix: Identifier) -> Identifier {
            match id {
                Identifier::Atom(suffix) => Identifier::Select(prefix.into(), suffix),
                Identifier::Select(base, x) => {
                    Identifier::Select(splice_prefix(*base, prefix).into(), x)
                }
            }
        }

        splice_prefix(self.clone(), prefix)
    }

    pub fn suffix_with(&self, suffix: &str) -> Self {
        Self::Select(self.clone().into(), suffix.to_owned())
    }

    pub fn try_from_components(path: &[&str]) -> Option<Self> {
        if let [head, tail @ ..] = path {
            Some(
                tail.iter()
                    .fold(Self::Atom((*head).to_owned()), |prefix, &x| {
                        Self::Select(prefix.into(), x.to_owned())
                    }),
            )
        } else {
            None
        }
    }

    pub fn components(&self) -> Vec<&str> {
        match self {
            Self::Atom(name) => vec![name],
            Self::Select(parent, name) => {
                let mut path = parent.components();
                path.push(name);
                path
            }
        }
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash, Ord)]
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

impl<A> ValueDeclaration<A>
where
    A: Copy + Parsed + fmt::Debug + fmt::Display,
{
    pub fn map<B>(self, f: fn(A) -> B) -> ValueDeclaration<B> {
        ValueDeclaration {
            binder: self.binder,
            type_signature: self.type_signature.map(|signature| signature.map(f)),
            declarator: self.declarator.map(f),
        }
    }

    pub fn map_expression<F>(self, f: F) -> Self
    where
        F: FnOnce(Expression<A>) -> Expression<A>,
    {
        Self {
            binder: self.binder,
            type_signature: self.type_signature,
            declarator: self.declarator.map_expression(f),
        }
    }

    pub fn dependencies(&self) -> HashSet<&Identifier> {
        let mut deps = HashSet::default();
        if let Some(signature) = &self.type_signature {
            deps.extend(signature.dependencies());
        }
        deps.extend(self.declarator.dependencies());
        deps
    }

    pub fn prefixed_with(self, name: Identifier) -> Self {
        Self {
            binder: self.binder.prefixed_with(name),
            type_signature: self.type_signature,
            declarator: self.declarator,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeSignature<A> {
    pub quantifiers: Option<UniversalQuantifiers>,
    pub body: TypeExpression<A>,
}

impl<A> TypeSignature<A>
where
    A: Clone + fmt::Debug + fmt::Display + Parsed,
{
    pub fn new(body: TypeExpression<A>) -> Self {
        Self {
            quantifiers: None,
            body,
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> TypeSignature<B> {
        TypeSignature {
            quantifiers: self.quantifiers,
            body: self.body.map(f),
        }
    }

    // Surely this resolves named types?
    pub fn synthesize_type(&self, ctx: &TypingContext) -> Typing<TypeScheme> {
        let type_parameters = self
            .quantifiers
            .as_ref()
            .map(|forall| forall.fresh_type_parameters())
            .unwrap_or_default();

        Ok(TypeScheme {
            quantifiers: type_parameters.values().cloned().collect(),
            body: self.body.synthesize_type(&type_parameters, ctx)?,
        })
    }

    fn dependencies(&self) -> HashSet<&Identifier> {
        //        self.body.dependencies()
        HashSet::default()
    }
}

impl<A> fmt::Display for TypeSignature<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { quantifiers, body } = self;

        if let Some(quantifiers) = quantifiers {
            write!(f, "{quantifiers}")?;
        }

        write!(f, "{body}")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeDeclaration<A> {
    pub binder: Identifier,
    pub declarator: TypeDeclarator<A>,
}

impl<A> TypeDeclaration<A> {
    pub fn prefixed_with(self, name: Identifier) -> Self {
        Self {
            binder: self.binder.prefixed_with(name),
            declarator: self.declarator,
        }
    }
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
    // with aliasing
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
    A: fmt::Display + fmt::Debug + Copy + Parsed,
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
                    binder: binding,
                    declarator,
                },
            ) => Declaration::Type(
                f(a),
                TypeDeclaration {
                    binder: binding,
                    declarator: declarator.map(f),
                },
            ),
            Self::Module(a, declarator) => Declaration::Module(f(a), declarator.map(f)),
            Self::ImportModule(a, ImportModule { exported_symbols }) => {
                Declaration::ImportModule(f(a), ImportModule { exported_symbols })
            }
        }
    }

    pub fn prefixed_with(self, name: Identifier) -> Self {
        match self {
            Self::Value(a, decl) => Self::Value(a, decl.prefixed_with(name)),
            Self::Type(a, decl) => Self::Type(a, decl.prefixed_with(name)),
            Self::Module(a, module) => Self::Module(a, module.prefixed_with(name)),
            otherwise => otherwise,
        }
    }

    pub fn binder(&self) -> Option<&Identifier> {
        match self {
            Self::Value(_, decl) => Some(&decl.binder),
            Self::Type(_, decl) => Some(&decl.binder),
            Self::Module(_, decl) => Some(&decl.name),
            Self::ImportModule(..) => None,
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
                    binder: binding,
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
pub struct UniversalQuantifiers(pub Vec<TypeName>);

impl UniversalQuantifiers {
    pub fn add(self, quantifier: TypeName) -> Self {
        let Self(mut quantifiers) = self;
        quantifiers.push(quantifier);
        Self(quantifiers)
    }

    pub fn parameters(&self) -> &[TypeName] {
        &self.0
    }

    pub fn fresh_type_parameters(&self) -> HashMap<TypeName, TypeParameter> {
        self.parameters()
            .iter()
            .map(|ty_var| (ty_var.clone(), TypeParameter::fresh()))
            .collect::<HashMap<TypeName, TypeParameter>>()
    }
}

impl fmt::Display for UniversalQuantifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(quantifiers) = self;

        if !quantifiers.is_empty() {
            write!(f, "forall ")?;

            let mut i = quantifiers.iter();

            if let Some(q) = i.next() {
                write!(f, "{q}")?;
            }

            for q in i {
                write!(f, " {q}")?;
            }

            write!(f, ". ")?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Coproduct<A> {
    pub forall: UniversalQuantifiers,
    pub constructors: Vec<Constructor<A>>,
    pub associated_module: Option<ModuleDeclarator<A>>,
}

impl<A> Coproduct<A>
where
    A: Copy + fmt::Debug + fmt::Display + Parsed,
{
    pub fn map<B>(self, f: fn(A) -> B) -> Coproduct<B> {
        let Self {
            forall,
            constructors,
            associated_module,
        } = self;
        Coproduct {
            forall,
            constructors: constructors.into_iter().map(|c| c.map(f)).collect(),
            associated_module: associated_module.map(|m| m.map(f)),
        }
    }

    // This should store the things in the associated module instead
    // Start with the struct
    pub fn make_implementation_module(
        &self,
        annotation: &A,
        self_name: TypeName,
        ctx: &TypingContext,
    ) -> Typing<CoproductModule<A>> {
        let declared_type = self.synthesize_type(ctx)?;
        println!("make_implementation_module: {declared_type}");

        let forall = self.forall.fresh_type_parameters();

        Ok(CoproductModule {
            name: self_name.clone(),
            declared_type,
            constructors: self
                .constructors
                .iter()
                .map(|constructor| {
                    constructor.make_function(annotation, self_name.clone(), &forall, ctx)
                })
                .collect::<Typing<_>>()?,
        })
    }

    fn synthesize_type(&self, ctx: &TypingContext) -> Typing<TypeScheme> {
        // This maps names to type parameters
        let universals = self.forall.fresh_type_parameters();
        let coproduct_type = self.synthesize_coproduct_type(&universals, ctx)?;
        self.make_type_scheme(universals, coproduct_type)
    }

    fn make_type_scheme(
        &self,
        universals: HashMap<TypeName, TypeParameter>,
        body: Type,
    ) -> Typing<TypeScheme> {
        let mut boofer = vec![];
        for var in self.forall.parameters() {
            let param = universals
                .get(var)
                .ok_or_else(|| TypeError::UndefinedQuantifier {
                    quantifier: var.clone(),
                    in_type: body.clone(),
                })?;

            boofer.push(*param);
        }

        Ok(TypeScheme::new(boofer.as_slice(), body))
    }

    fn synthesize_coproduct_type(
        &self,
        universals: &HashMap<TypeName, TypeParameter>,
        ctx: &TypingContext,
    ) -> Typing<Type> {
        Ok(Type::Coproduct(CoproductType::new(
            self.constructors
                .iter()
                .map(|Constructor { name, signature }| {
                    let tuple_signature = signature
                        .iter()
                        .map(|expr| expr.synthesize_type(universals, ctx))
                        .collect::<Typing<_>>();

                    tuple_signature.map(|signature| {
                        (
                            name.as_str().to_owned(),
                            Type::Product(ProductType::Tuple(TupleType(signature))),
                        )
                    })
                })
                .collect::<Typing<_>>()?,
        )))
    }
}

pub struct CoproductModule<A> {
    pub name: TypeName,
    pub declared_type: TypeScheme,
    pub constructors: Vec<ValueDeclaration<A>>,
}

// Rename into Record?
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Struct<A> {
    pub forall: UniversalQuantifiers,
    pub fields: Vec<StructField<A>>,
    pub associated_module: Option<ModuleDeclarator<A>>,
}

impl<A> Struct<A>
where
    A: fmt::Debug + fmt::Display + Copy + Parsed,
{
    pub fn map<B>(self, f: fn(A) -> B) -> Struct<B> {
        let Self {
            forall,
            fields,
            associated_module,
        } = self;
        Struct {
            forall,
            fields: fields.into_iter().map(|x| x.map(f)).collect(),
            associated_module: associated_module.map(|m| m.map(f)),
        }
    }

    pub fn synthesize_type(&self, ctx: &TypingContext) -> Typing<TypeScheme> {
        let universals = self.forall.fresh_type_parameters();
        let record_type = self.synthesize_struct_type(&universals, ctx)?;
        self.make_type_scheme(universals, record_type)
    }

    fn synthesize_struct_type(
        &self,
        universals: &HashMap<TypeName, TypeParameter>,
        ctx: &TypingContext,
    ) -> Result<Type, TypeError> {
        Ok(Type::Product(ProductType::Struct(
            self.fields
                .iter()
                .map(|f| {
                    f.type_annotation
                        .synthesize_type(universals, ctx)
                        .map(|field_type| (f.name.clone(), field_type))
                })
                .collect::<Typing<_>>()?,
        )))
    }

    fn make_type_scheme(
        &self,
        universals: HashMap<TypeName, TypeParameter>,
        body: Type,
    ) -> Typing<TypeScheme> {
        let mut boofer = vec![];
        for var in self.forall.parameters() {
            let param = universals
                .get(var)
                .ok_or_else(|| TypeError::UndefinedQuantifier {
                    quantifier: var.clone(),
                    in_type: body.clone(),
                })?;

            boofer.push(*param);
        }

        Ok(TypeScheme::new(boofer.as_slice(), body))
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
    A: Copy + Parsed + fmt::Debug + fmt::Display,
{
    fn map<B>(self, f: fn(A) -> B) -> TypeDeclarator<B> {
        match self {
            Self::Alias(a, alias) => TypeDeclarator::Alias(f(a), alias.map(f)),
            Self::Coproduct(a, coproduct) => TypeDeclarator::Coproduct(f(a), coproduct.map(f)),
            Self::Struct(a, record) => TypeDeclarator::Struct(f(a), record.map(f)),
        }
    }

    pub fn synthesize_type(&self, ctx: &TypingContext) -> Typing<TypeScheme> {
        match self {
            Self::Alias(..) => todo!(),
            Self::Coproduct(_, coproduct) => coproduct.synthesize_type(ctx),
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
            Self::Struct(_, Struct { forall, fields, .. }) => {
                writeln!(f, "{forall}struct {{")?;
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

impl<A> StructField<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    fn make_function(
        &self,
        annotation: &A,
        ty: TypeName,
        type_param_map: &HashMap<TypeName, TypeParameter>,
        ctx: &TypingContext,
    ) -> Typing<ValueDeclaration<A>> {
        let parameters = self
            .signature
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                // I could give this one the real TypeParameter via
                // type_parameters. Then Parameter would hold Type
                // instead of TypeExpression
                expr.synthesize_type(type_param_map, ctx).map(|ty| {
                    Parameter::new_with_type_annotation(Identifier::new(&format!("p{index}")), ty)
                })
            })
            .collect::<Typing<Vec<_>>>()?;

        // It should really annotate it with types to make sure the
        // typer gets it right. But that should not be necessary yet.
        let expression = self.make_injection_lambda_tree(annotation, ty, parameters.clone());

        Ok(ValueDeclaration {
            binder: self.name.clone(),
            // Compute a type signature type expression!
            // I have self.signature which can be joined with -> to make the function, I guess?
            type_signature: None,
            declarator: ValueDeclarator { expression },
        })
    }

    fn make_injection_lambda_tree(
        &self,
        annotation: &A,
        name: TypeName,
        parameters: Vec<Parameter>,
    ) -> Expression<A>
    where
        A: fmt::Display + Clone + Parsed,
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
    Constructor(A, Identifier),
    Parameter(A, Identifier),
    Apply(A, TypeApply<A>),
    Arrow(A, Arrow<A>),
}

impl<A> TypeExpression<A>
where
    A: Clone + Parsed + fmt::Debug + fmt::Display,
{
    pub fn map<B>(self, f: fn(A) -> B) -> TypeExpression<B> {
        match self {
            Self::Constructor(a, id) => TypeExpression::Constructor(f(a), id),
            Self::Parameter(a, id) => TypeExpression::Parameter(f(a), id),
            Self::Apply(a, apply) => TypeExpression::Apply(f(a), apply.map(f)),
            Self::Arrow(a, arrow) => TypeExpression::Arrow(f(a), arrow.map(f)),
        }
    }

    // Make a thing to capture this type_param_map later.
    // Why isn't this a TypeScheme?
    pub fn synthesize_type(
        &self,
        type_params: &HashMap<TypeName, TypeParameter>,
        ctx: &TypingContext,
    ) -> Typing<Type> {
        match self {
            Self::Constructor(_, name) => {
                let name = TypeName::new(&name.as_str());
                let scheme = ctx
                    .lookup_scheme(&name.clone().into())
                    .cloned()
                    .unwrap_or_else(|| {
                        // If the type scheme is defined, then instantiate it. Otherwise default to a named
                        // reference. This is necessary when declaring recursive types.
                        TypeScheme::from_constant(Type::Named(TypeName::new(name.as_str())))
                    });

                // A little kludgy
                if !scheme.is_type_constructor() {
                    scheme.instantiate(ctx)
                } else {
                    Ok(Type::Named(TypeName::new(name.as_str())))
                }
            }
            Self::Parameter(_, param) => {
                let type_name = TypeName::new(&param.as_str());

                type_params
                    .get(&type_name)
                    .cloned()
                    .map(Type::Parameter)
                    .ok_or_else(|| TypeError::UndefinedQuantifierInTypeExpression {
                        quantifier: type_name,
                        in_expression: self.clone().map(|a| *a.info()),
                    })
            }
            Self::Apply(_, node) => Ok(Type::Apply(
                node.constructor.synthesize_type(type_params, ctx)?.into(),
                node.argument.synthesize_type(type_params, ctx)?.into(),
            )),
            Self::Arrow(_, arrow) => Ok(Type::Arrow(
                arrow.domain.synthesize_type(type_params, ctx)?.into(),
                arrow.codomain.synthesize_type(type_params, ctx)?.into(),
            )),
        }
    }
}

impl<A> fmt::Display for TypeExpression<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constructor(_, id) => write!(f, "{id}"),
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

impl<A> TypeApply<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Arrow<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn dependencies(&self) -> HashSet<&Identifier> {
        self.expression.free_identifiers()
    }

    fn map<B>(self, f: fn(A) -> B) -> ValueDeclarator<B> {
        ValueDeclarator {
            expression: self.expression.map(f),
        }
    }

    pub fn map_expression<F>(self, f: F) -> Self
    where
        F: FnOnce(Expression<A>) -> Expression<A>,
    {
        Self {
            expression: f(self.expression),
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
    TypeAscription(A, TypeAscription<A>),
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
pub struct TypeAscription<A> {
    // Should this be a TypeSignature instead?
    pub type_signature: TypeSignature<A>,
    pub underlying: Box<Expression<A>>,
}

impl<A> TypeAscription<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn map<B>(self, f: fn(A) -> B) -> TypeAscription<B> {
        TypeAscription {
            type_signature: self.type_signature.map(f),
            underlying: self.underlying.map(f).into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeconstructInto<A> {
    pub scrutinee: Box<Expression<A>>,
    pub match_clauses: Vec<MatchClause<A>>,
}

impl<A> DeconstructInto<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn map<B>(self, f: fn(A) -> B) -> DeconstructInto<B> {
        DeconstructInto {
            scrutinee: self.scrutinee.map(f).into(),
            match_clauses: self
                .match_clauses
                .into_iter()
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

impl<A> MatchClause<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

        // Why am I doing this?
        free.extend(self.pattern.free_variables());
    }
}

pub struct PatternMatrix {
    domain: DomainExpression,
    matched_space: DomainExpression,
}

impl PatternMatrix {
    pub fn from_scrutinee(scrutinee: Type, _ctx: &TypingContext) -> Typing<Self> {
        Ok(Self {
            domain: DomainExpression::from_type(scrutinee),
            matched_space: DomainExpression::default(),
        })
    }

    pub fn integrate(&mut self, pattern: DomainExpression) {
        self.matched_space.join(pattern);
    }

    pub fn is_useful(&self, pattern: &DomainExpression) -> bool {
        !pattern.is_covered_by(&self.matched_space)
    }

    pub fn is_exhaustive(&self) -> bool {
        self.residual() == DomainExpression::Nothing
    }

    pub fn residual(&self) -> DomainExpression {
        self.domain.eliminate(&self.matched_space)
    }
}

impl fmt::Display for PatternMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            domain,
            matched_space,
        } = self;
        write!(f, "domain: {domain} and matched space: {matched_space}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DomainExpression {
    Nothing,
    Whole(Type),
    Join(BTreeSet<DomainExpression>),
    Subtraction {
        lhs: Box<DomainExpression>,
        rhs: Box<DomainExpression>,
    },

    // These are different from the rest
    Literal(Constant),
    Coproduct(Vec<(Identifier, Vec<DomainExpression>)>),
    Tuple(Vec<DomainExpression>),
    Struct(Vec<(Identifier, DomainExpression)>),
}

impl fmt::Display for DomainExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nothing => write!(f, "()"),
            Self::Whole(ty) => write!(f, "dom({ty})"),
            Self::Join(set) => {
                let mut i = set.iter();
                if let Some(s) = i.next() {
                    write!(f, "{{ {s}")?;
                }
                for s in i {
                    write!(f, "| {s}")?;
                }
                write!(f, " }}")
            }
            Self::Subtraction { lhs, rhs } => {
                write!(f, "{lhs} \\ ")?;
                write!(f, "({rhs})")
            }
            Self::Literal(constant) => write!(f, "{constant}"),
            Self::Coproduct(constructors) => {
                let mut i = constructors.iter();
                if let Some((constructor, arguments)) = i.next() {
                    write!(
                        f,
                        "{constructor} {}",
                        arguments
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(" ")
                    )?;
                }

                for (constructor, arguments) in i {
                    write!(
                        f,
                        "{constructor} {}",
                        arguments
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(" ")
                    )?;
                }

                Ok(())
            }
            Self::Tuple(elements) => {
                let mut i = elements.iter();
                if let Some(element) = i.next() {
                    write!(f, "{}", element)?;
                }
                for element in i {
                    write!(f, "{}", element)?;
                }
                Ok(())
            }
            Self::Struct(fields) => {
                let mut i = fields.iter();
                if let Some((identifier, expr)) = i.next() {
                    write!(f, "{identifier}: {expr}")?;
                }
                for (identifier, expr) in i {
                    write!(f, "{identifier}: {expr}")?;
                }
                Ok(())
            }
        }
    }
}

impl DomainExpression {
    pub fn from_pattern(pattern: &Pattern<ParsingInfo>, ctx: &TypingContext) -> Typing<Self> {
        Ok(match pattern {
            Pattern::Coproduct(_, coproduct) => Self::Coproduct(vec![(
                coproduct.constructor.clone(),
                coproduct
                    .argument
                    .elements
                    .iter()
                    .map(|p| Self::from_pattern(p, ctx))
                    .collect::<Typing<_>>()?,
            )]),

            Pattern::Tuple(_, product) => Self::Tuple(
                product
                    .elements
                    .iter()
                    .map(|p| Self::from_pattern(p, ctx))
                    .collect::<Typing<_>>()?,
            ),

            Pattern::Struct(_, record) => Self::Struct(
                record
                    .fields
                    .iter()
                    .map(|(field, p)| Self::from_pattern(p, ctx).map(|expr| (field.clone(), expr)))
                    .collect::<Typing<_>>()?,
            ),

            Pattern::Literally(_, constant) => Self::Literal(constant.clone()),

            Pattern::Otherwise(..) => {
                let pattern = pattern.synthesize_type(ctx)?;
                Self::Whole(pattern.inferred_type)
            }
        })
    }

    // The constructor function ought to expand the type,
    // but not the internal/ recursive one
    fn from_type(domain: Type) -> Self {
        match domain {
            Type::Product(product) => Self::from_product(product),
            Type::Coproduct(coproduct) => Self::from_coproduct(coproduct),
            otherwise => Self::Whole(otherwise),
        }
    }

    fn from_coproduct(coproduct: CoproductType) -> Self {
        Self::Coproduct(
            coproduct
                .into_iter()
                .map(|(constructor, signature)| {
                    if let product @ Type::Product(..) = signature {
                        if let Self::Tuple(elements) = Self::from_type(product) {
                            (Identifier::new(&constructor), elements)
                        } else {
                            panic!("Constructor signatures are expected to be tuples")
                        }
                    } else {
                        // internal error
                        panic!("Constructor signatures are expected to be tuples")
                    }
                })
                .collect(),
        )
    }

    fn from_product(product: ProductType) -> Self {
        match product {
            ProductType::Tuple(tuple) => {
                let TupleType(elements) = tuple.unspine();
                Self::Tuple(elements.into_iter().map(Self::from_type).collect())
            }
            ProductType::Struct(fields) => Self::Struct(
                fields
                    .into_iter()
                    .map(|(field, ty)| (field, Self::from_type(ty)))
                    .collect(),
            ),
        }
    }

    fn join(&mut self, rhs: Self) {
        *self = match (mem::take(self), rhs) {
            (Self::Nothing, rhs) => rhs,

            (Self::Whole(t), _) | (_, Self::Whole(t)) => Self::Whole(t),

            (lhs @ Self::Literal(..), rhs @ Self::Literal(..)) => {
                Self::Join(BTreeSet::from([lhs, rhs]))
            }

            (Self::Tuple(mut lhs), Self::Tuple(rhs)) => {
                inner_join(&mut lhs, rhs);
                Self::Tuple(lhs)
            }

            (Self::Struct(mut lhs), Self::Struct(rhs)) => Self::Struct({
                let mut rhs = rhs.into_iter().collect::<HashMap<_, _>>();

                for (field, lhs) in lhs.iter_mut() {
                    lhs.join(rhs.remove(field).expect("bad pattern"));
                }

                lhs
            }),

            (Self::Coproduct(mut lhs), Self::Coproduct(rhs)) => Self::Coproduct({
                let mut rhs = rhs.into_iter().collect::<HashMap<_, _>>();

                for (constructor, lhs) in lhs.iter_mut() {
                    if let Some(rhs) = rhs.remove(constructor) {
                        inner_join(lhs, rhs);
                    }
                }

                lhs.extend(rhs);
                lhs
            }),

            (Self::Join(mut lhs), Self::Join(rhs)) => Self::Join({
                lhs.extend(rhs);
                lhs
            }),

            (Self::Join(mut lhs), rhs) => Self::Join({
                lhs.insert(rhs);
                lhs
            }),

            (lhs, rhs) => panic!("join: {lhs:?} and {rhs:?} must not be joined"),
        };
    }

    fn eliminate(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (_lhs, Self::Whole(..)) => Self::Nothing,

            (Self::Whole(..), rhs) => Self::Subtraction {
                lhs: self.clone().into(),
                rhs: rhs.clone().into(),
            },

            (Self::Coproduct(lhs), Self::Coproduct(rhs)) => self.eliminate_constructors(lhs, rhs),

            (Self::Tuple(lhs), Self::Tuple(rhs)) => {
                let elements = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| lhs.eliminate(rhs))
                    .collect::<Vec<_>>();

                if elements.iter().all(|e| e.is_nothing()) {
                    Self::Nothing
                } else {
                    Self::Tuple(elements)
                }
            }

            (Self::Struct(lhs), Self::Struct(rhs)) => {
                let mut rhs = rhs.iter().cloned().collect::<HashMap<_, _>>();

                let fields = lhs
                    .iter()
                    .map(|(field, lhs)| {
                        (
                            field.clone(),
                            lhs.eliminate(&rhs.remove(field).expect("bad pattern")),
                        )
                    })
                    .collect::<Vec<_>>();

                if fields.iter().all(|(_, e)| e.is_nothing()) {
                    Self::Nothing
                } else {
                    Self::Struct(fields)
                }
            }

            (lhs, rhs) => panic!("Absurd combination {lhs:?} {rhs:?}"),
        }
    }

    // [x | constructors(C) \ constructors(D), elements(C) \ elements(D), c in C \ d in D, x != Nothing]
    fn eliminate_constructors(
        &self,
        lhs: &[(Identifier, Vec<DomainExpression>)],
        rhs: &[(Identifier, Vec<DomainExpression>)],
    ) -> DomainExpression {
        let rhs = rhs.iter().cloned().collect::<HashMap<_, _>>();
        let constructors = lhs
            .iter()
            .filter_map(|(constructor, lhs)| {
                if let Some(rhs) = rhs.get(constructor) {
                    let parameters = lhs
                        .iter()
                        .zip(rhs.iter())
                        .map(|(lhs, rhs)| lhs.eliminate(rhs))
                        .collect::<Vec<_>>();
                    parameters
                        .iter()
                        .any(|de| !de.is_nothing())
                        .then_some((constructor.clone(), parameters))
                } else {
                    Some((constructor.clone(), lhs.clone()))
                }
            })
            .collect::<Vec<(Identifier, Vec<DomainExpression>)>>();

        if constructors.is_empty() {
            Self::Nothing
        } else {
            Self::Coproduct(constructors)
        }
    }

    pub fn is_nothing(&self) -> bool {
        self == &DomainExpression::Nothing
    }

    pub fn is_saturated(&self) -> bool {
        match self {
            Self::Whole(..) => true,
            Self::Tuple(elements) => elements.iter().all(|e| e.is_saturated()),
            Self::Struct(fields) => fields.iter().all(|(_, f)| f.is_saturated()),
            _otherwise => false,
        }
    }

    fn is_covered_by(&self, matched_space: &Self) -> bool {
        match (self, matched_space) {
            (Self::Nothing, ..) | (.., Self::Whole(..)) => true,

            (Self::Literal(lhs), Self::Literal(rhs)) => lhs == rhs,

            (lhs, Self::Join(set)) => set.contains(lhs),

            (Self::Coproduct(lhs), Self::Coproduct(rhs)) => lhs.iter().all(|(id, lhs)| {
                rhs.iter().any(|(id1, rhs)| {
                    (id == id1)
                        && lhs
                            .iter()
                            .zip(rhs.iter())
                            .all(|(lhs, rhs)| lhs.is_covered_by(rhs))
                })
            }),

            (Self::Tuple(lhs), Self::Tuple(rhs)) => lhs
                .iter()
                .zip(rhs.iter())
                .all(|(lhs, rhs)| lhs.is_covered_by(rhs)),

            (Self::Struct(lhs), Self::Struct(rhs)) => {
                let mut rhs = rhs
                    .iter()
                    .map(|(field, rhs)| (field, rhs))
                    .collect::<HashMap<_, _>>();

                lhs.iter().all(|(field, lhs)| {
                    lhs.is_covered_by(rhs.remove(field).expect("Internal error"))
                })
            }

            (_lhs, rhs) => rhs.is_saturated(),
        }
    }
}

fn inner_join(lhs: &mut [DomainExpression], rhs: Vec<DomainExpression>) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        lhs.join(rhs);
    }
}

impl Default for DomainExpression {
    fn default() -> Self {
        Self::Nothing
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<A> {
    // First argument should be A, for god's sake.
    Coproduct(A, ConstructorPattern<A>),
    Tuple(A, TuplePattern<A>),
    Struct(A, StructPattern<A>),
    Literally(A, Constant),
    Otherwise(A, Identifier),
}

impl<A> Pattern<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> Pattern<B> {
        match self {
            Self::Coproduct(a, pattern) => Pattern::Coproduct(f(a), pattern.map(f)),
            Self::Tuple(a, pattern) => Pattern::Tuple(f(a), pattern.map(f)),
            Self::Struct(a, pattern) => Pattern::Struct(f(a), pattern.map(f)),
            Self::Literally(a, literal) => Pattern::Literally(f(a), literal),
            Self::Otherwise(a, id) => Pattern::Otherwise(f(a), id),
        }
    }

    fn bindings(&self) -> Vec<&Identifier> {
        match self {
            Self::Coproduct(_, pattern) => pattern
                .argument
                .elements
                .iter()
                .flat_map(|p| p.bindings())
                .collect(),
            Self::Tuple(_, p) => p.elements.iter().flat_map(|p| p.bindings()).collect(),
            Self::Struct(_, p) => p
                .fields
                .iter()
                .flat_map(|(field, p)| {
                    let mut pattern_bindings = p.bindings();
                    pattern_bindings.push(field);
                    pattern_bindings
                })
                .collect(),
            Self::Literally(..) => vec![],
            Self::Otherwise(_, pattern) => vec![pattern],
        }
    }

    // Why is this necessary?
    // This thing only collects coproduct constructor names
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

            Self::Struct(_, pattern) => pattern
                .fields
                .iter()
                .flat_map(|(_field, p)| p.free_variables())
                .collect(),

            _otherwise => HashSet::default(),
        }
    }

    pub fn annotation(&self) -> &A {
        match self {
            Pattern::Coproduct(annotation, ..) => annotation,
            Pattern::Tuple(annotation, ..) => annotation,
            Pattern::Struct(annotation, ..) => annotation,
            Pattern::Literally(annotation, ..) => annotation,
            Pattern::Otherwise(annotation, ..) => annotation,
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
    pub fn map<B>(self, f: fn(A) -> B) -> TuplePattern<B> {
        TuplePattern {
            elements: self.elements.into_iter().map(|p| p.map(f)).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructPattern<A> {
    pub fields: Vec<(Identifier, Pattern<A>)>,
}

impl<A> StructPattern<A> {
    pub fn map<B>(self, f: fn(A) -> B) -> StructPattern<B> {
        StructPattern {
            fields: self
                .fields
                .into_iter()
                .map(|(field, p)| (field, p.map(f)))
                .collect(),
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
            ) => write!(f, "{constructor} {argument}"),
            Self::Tuple(_, pattern) => write!(f, "( {pattern} )"),
            Self::Struct(_, pattern) => write!(f, "{{ {pattern} }}"),
            Self::Literally(_, literal) => write!(f, "{literal}"),
            Self::Otherwise(_, id) => write!(f, "{id}"),
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

impl<A> fmt::Display for StructPattern<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (field, pattern) in &self.fields {
            write!(f, "{field}: {pattern}")?;
        }
        write!(f, "}}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfReferential<A> {
    pub name: Identifier,
    pub parameter: Parameter,
    pub body: Box<Expression<A>>,
}

impl<A> SelfReferential<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Lambda<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Apply<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Inject<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Project<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Binding<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

impl<A> Sequence<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    fn map<B>(self, f: fn(A) -> B) -> Sequence<B> {
        Sequence {
            this: self.this.map(f).into(),
            and_then: self.and_then.map(f).into(),
        }
    }
}

impl Expression<ParsingInfo> {
    pub fn position(&self) -> &lexer::SourceLocation {
        &self.parsing_info().location()
    }

    pub fn parsing_info(&self) -> &ParsingInfo {
        self.annotation()
    }
}

impl<A> Expression<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn annotation(&self) -> &A {
        match self {
            Self::TypeAscription(annotation, ..)
            | Self::Variable(annotation, ..)
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

    pub fn free_identifiers(&self) -> HashSet<&Identifier> {
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

            Self::TypeAscription(_, ascription) => {
                ascription.underlying.find_unbound(bound, free);
            }

            _otherwise => (),
        }
    }

    pub fn map<B>(self, f: fn(A) -> B) -> Expression<B> {
        match self {
            Self::TypeAscription(x, info) => Expression::<B>::TypeAscription(f(x), info.map(f)),
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

    pub fn erase_annotation(self) -> Expression<EmptyAnnotation> {
        self.map(|_| EmptyAnnotation)
    }
}

impl<A> Expression<A> where A: Clone {}

impl<A> fmt::Display for Expression<A>
where
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::TypeAscription(_, ta) => {
                write!(f, "{}::{}", ta.underlying, ta.type_signature)
            }
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

impl<A> ControlFlow<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Unit,
}

impl Eq for Constant {}

impl Ord for Constant {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("error")
    }
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

impl<A> Product<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    fn map<B>(self, f: fn(A) -> B) -> Product<B> {
        match self {
            Self::Tuple(expressions) => {
                Product::Tuple(expressions.into_iter().map(|x| x.map(f)).collect())
            }
            Self::Struct(bindings) => {
                Product::<B>::Struct(bindings.into_iter().map(|(k, v)| (k, v.map(f))).collect())
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
                write!(f, "(")?;
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
    use crate::{ast::ValueDeclaration, parser::ParsingInfo};

    use super::{Constant, Declaration, Expression, Identifier, ModuleDeclarator, ValueDeclarator};

    #[test]
    fn cyclic_dependencies() {
        // I should parse text instead of this
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("foo"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("quux"),
                            ),
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
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("frobnicator"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Literal(
                                ParsingInfo::default(),
                                Constant::Int(1),
                            ),
                        },
                    },
                ),
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(dep_mat.is_satisfiable());
    }

    #[test]
    fn unsatisfiable() {
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("frobnicator"),
                            ),
                        },
                    },
                ),
            ],
        };

        let dep_mat = m.dependency_graph();
        assert!(dep_mat.is_acyclic());
        assert!(!dep_mat.is_satisfiable());
    }

    //    #[test]
    fn _top_of_the_day() {
        let m = ModuleDeclarator {
            name: Identifier::new(""),
            declarations: vec![
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("foo"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("quux"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("bar"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("bar"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Variable(
                                ParsingInfo::default(),
                                Identifier::new("frobnicator"),
                            ),
                        },
                    },
                ),
                Declaration::Value(
                    ParsingInfo::default(),
                    ValueDeclaration {
                        binder: Identifier::new("frobnicator"),
                        type_signature: None,
                        declarator: ValueDeclarator {
                            expression: Expression::Literal(
                                ParsingInfo::default(),
                                Constant::Int(1),
                            ),
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
