#![allow(dead_code)]

use marmelade::{
    ast::{
        Apply, Binding, Constant, Constructor, ControlFlow, Coproduct, Declaration, Expression,
        Identifier, Sequence, TypeApply, TypeDeclarator, TypeExpression, TypeName,
        UniversalQuantifiers,
    },
    context::Linkage,
    interpreter::Value,
    lexer::LexicalAnalyzer,
    parser::{self, ParsingInfo},
    stdlib,
};

use Expression as E;

pub fn decl_fixture(source: &str, rhs: Declaration<ParsingInfo>) {
    let mut lexer = LexicalAnalyzer::default();
    let lhs = parser::parse_declaration_phrase(lexer.tokenize(&into_unicode_text(source))).unwrap();

    //dbg!(&lhs);
    //    if let Declaration::Type(
    //        _,
    //        TypeDeclaration {
    //            declarator: TypeDeclarator::Coproduct(_, ctor),
    //            ..
    //        },
    //    ) = &lhs
    //    {
    //        for ctor in &ctor.constructors {
    //            for s in &ctor.signature {
    //                println!("decl_fixture: {:?}", s.deconstruct_apply_tree());
    //            }
    //        }
    //    }

    assert_eq!(lhs.map(|_| ParsingInfo::default()), rhs)
}

pub fn expr_fixture(source: &str, rhs: E<ParsingInfo>) {
    let mut lexer = LexicalAnalyzer::default();
    let lhs = parser::parse_expression_phrase(lexer.tokenize(&into_unicode_text(source))).unwrap();

    assert_eq!(lhs.erase_annotation(), rhs.erase_annotation())
}

pub fn eval_fixture<A>(source: &str, rhs: A)
where
    A: Into<Value>,
{
    let source_text = into_unicode_text(source);
    let mut compilation = Linkage::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let return_value = compilation.typecheck_and_interpret();

    assert_eq!(
        return_value
            .map_err(|e| format!("{e}"))
            .inspect(|x| println!("{x}"))
            .unwrap(),
        //            .try_into_base_type()
        //            .unwrap(),
        rhs.into() //.try_into_base_type().unwrap()
    )
}

pub fn int(i: i64) -> E<ParsingInfo> {
    E::Literal(ParsingInfo::default(), Constant::Int(i))
}

pub fn text(s: &str) -> E<ParsingInfo> {
    E::Literal(ParsingInfo::default(), Constant::Text(s.to_owned()))
}

pub fn float(f: f64) -> E<ParsingInfo> {
    E::Literal(ParsingInfo::default(), Constant::Float(f))
}

pub fn let_in(binder: &str, bound: E<ParsingInfo>, body: E<ParsingInfo>) -> E<ParsingInfo> {
    E::Binding(
        ParsingInfo::default(),
        Binding {
            binder: ident(binder),
            bound: bound.into(),
            body: body.into(),
        },
    )
}

pub fn if_else(
    predicate: E<ParsingInfo>,
    consequent: E<ParsingInfo>,
    alternate: E<ParsingInfo>,
) -> E<ParsingInfo> {
    E::ControlFlow(
        ParsingInfo::default(),
        ControlFlow::If {
            predicate: predicate.into(),
            consequent: consequent.into(),
            alternate: alternate.into(),
        },
    )
}

pub fn seq(this: E<ParsingInfo>, and_then: E<ParsingInfo>) -> E<ParsingInfo> {
    E::Sequence(
        ParsingInfo::default(),
        Sequence {
            this: this.into(),
            and_then: and_then.into(),
        },
    )
}

pub fn apply(f: E<ParsingInfo>, x: E<ParsingInfo>) -> E<ParsingInfo> {
    E::Apply(
        ParsingInfo::default(),
        Apply {
            function: f.into(),
            argument: x.into(),
        },
    )
}

pub fn var(id: &str) -> E<ParsingInfo> {
    E::Variable(ParsingInfo::default(), ident(id))
}

pub fn ident(id: &str) -> Identifier {
    Identifier::new(id)
}

pub fn tyname(id: &str) -> TypeName {
    TypeName::new(id)
}

pub fn typar(id: &str) -> TypeExpression<ParsingInfo> {
    TypeExpression::Parameter(ParsingInfo::default(), ident(id))
}

pub fn tyref(id: &str) -> TypeExpression<ParsingInfo> {
    TypeExpression::Constructor(ParsingInfo::default(), ident(id))
}

pub fn constructor(id: &str, te: Vec<TypeExpression<ParsingInfo>>) -> Constructor<ParsingInfo> {
    Constructor {
        name: ident(id),
        signature: te,
    }
}

pub fn tyapp(
    f: TypeExpression<ParsingInfo>,
    a: TypeExpression<ParsingInfo>,
) -> TypeExpression<ParsingInfo> {
    TypeExpression::Apply(
        ParsingInfo::default(),
        TypeApply {
            constructor: f.into(),
            argument: a.into(),
        },
    )
}

pub fn coproduct(
    forall: UniversalQuantifiers,
    constructors: Vec<Constructor<ParsingInfo>>,
) -> TypeDeclarator<ParsingInfo> {
    TypeDeclarator::Coproduct(
        ParsingInfo::default(),
        Coproduct {
            forall,
            constructors,
            associated_module: None,
        },
    )
}

pub fn into_unicode_text(source: &str) -> Vec<char> {
    source
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .collect()
}
