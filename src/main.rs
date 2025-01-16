use std::{collections::HashMap, result};

use uuid::Uuid;

#[derive(Clone, Debug)]
struct TypingContext {
    types: HashMap<Name, Type>,
    symbols: HashMap<Name, Type>,
}

impl TypingContext {
    fn check(&mut self, expected: Type, e: &Expression) -> Option<TypingError> {
        match e.infer_type(self) {
            Ok(t) if t == expected => None,
            Ok(t) => Some(TypingError::Expected { found: t, expected }),
            Err(err) => Some(err),
        }
    }

    fn symbol_type(&self, name: &Name) -> Option<&Type> {
        self.symbols.get(name)
    }

    fn annotate_symbol(&mut self, name: &Name, ty: Type) -> Option<Type> {
        self.symbols.insert(name.clone(), ty)
    }

    fn type_for_name(&self, name: &Name) -> Option<&Type> {
        self.types.get(name)
    }

    fn define_type(&mut self, name: &Name, ty: Type) -> Option<Type> {
        self.types.insert(name.clone(), ty)
    }

    fn assume_type_in<F>(&mut self, name: &Name, ty: Type, infer: F) -> Typing
    where
        F: FnOnce(&mut Self) -> Typing,
    {
        let former = self.annotate_symbol(name, ty);
        let tpe = infer(self)?;
        if let Some(former) = former {
            self.annotate_symbol(name, former);
        }
        Ok(tpe)
    }
}

impl Default for TypingContext {
    fn default() -> Self {
        let mut ctx = Self {
            types: HashMap::default(),
            symbols: HashMap::default(),
        };

        ctx.define_type(&TrivialType::Text.name(), Type::Trivial(TrivialType::Text));
        ctx.define_type(
            &TrivialType::Boolean.name(),
            Type::Trivial(TrivialType::Boolean),
        );
        ctx.define_type(
            &TrivialType::FloatingPoint.name(),
            Type::Trivial(TrivialType::FloatingPoint),
        );
        ctx.define_type(
            &TrivialType::Integer.name(),
            Type::Trivial(TrivialType::Integer),
        );

        ctx
    }
}

#[derive(Debug)]
enum TypingError {
    UnknownType(Name),
    Expected { found: Type, expected: Type },
}

type Typing = result::Result<Type, TypingError>;

#[derive(Debug, Default)]
struct Typer {
    context: TypingContext,
}

impl Typer {
    fn check(&mut self, program: &CompilationUnit) {}
}

struct CompilationUnit {
    main: Expression,
}

#[derive(Debug)]
enum Statement {
    // This is a toplevel Let
    //   Requires type annotations for abstractions
    //   So perhaps call these Define instead and
    //   include the abstraction in it
    // Add a Let-In Expression
    //   Optional annotations
    Let(Name, Expression),
    Expression(Expression),
}

#[derive(Debug)]
enum Expression {
    Constant(Literal),
    Variable(Name),
    Apply {
        abstraction: Box<Expression>,
        term: Box<Expression>,
    },
    ControlFlow(ControlFlow),
    // This forces the typer to be mutable
    AndThen(Vec<Statement>, Box<Expression>),
}

impl Expression {
    fn infer_type(&self, ctx: &mut TypingContext) -> Typing {
        match self {
            Self::Constant(literal) => literal.infer_type(ctx),
            Self::Variable(name) => ctx
                .symbol_type(name)
                .cloned()
                .ok_or_else(|| TypingError::UnknownType(name.clone())),

            Self::Apply { abstraction, term } => {
                let term = term.infer_type(ctx)?;
                let found = abstraction.infer_type(ctx)?;

                if let Type::Compound(CompoundType::Function { domain, codomain }) = found {
                    match *domain {
                        Type::Undecided(parameter) => {
                            ctx.assume_type_in(&parameter, term, |ctx| self.infer_type(ctx))
                        }
                        expected if expected == term => Ok(*codomain),
                        expected => Err(TypingError::Expected {
                            found: term,
                            expected,
                        }),
                    }
                } else {
                    Err(TypingError::Expected {
                        found,
                        expected: Type::Compound(CompoundType::Function {
                            domain: term.into(),
                            codomain: Type::Undecided(Name::fresh()).into(),
                        }),
                    })
                }
            }
            Self::ControlFlow(control_flow) => control_flow.infer_type(ctx),
            Self::AndThen(..) => todo!(),
        }
    }
}

#[derive(Debug)]
enum ControlFlow {
    If {
        predicate: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Box<Expression>,
    },
}

impl ControlFlow {
    fn infer_type(&self, ctx: &mut TypingContext) -> Typing {
        match self {
            ControlFlow::If {
                predicate,
                consequent,
                alternate,
            } => {
                if let Some(error) = ctx.check(Type::Trivial(TrivialType::Boolean), predicate) {
                    Err(error)
                } else {
                    let consequent = consequent.infer_type(ctx)?;
                    if let Some(error) = ctx.check(consequent.clone(), alternate) {
                        Err(error)
                    } else {
                        Ok(consequent)
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
enum Literal {
    Scalar(Constant),
    Tuple(Vec<Expression>),
    Record(Vec<(Name, Expression)>),
    // \x -> x + 1 =>
    //
    // Abstract {
    //   parameter: "x",
    //   parameter_type: None,
    //   closure: Box::new(
    //     Expression::Apply {
    //       function: Expression::Apply {
    //         symbol: Expression::Variable("builtin::math::+"),
    //         term: Expressoin::Variable("x")
    //       },
    //       term: Expression::Literal(Constant::Integer(1))
    //     }
    //   )
    // }
    //
    // Why is this a Literal?
    Abstract {
        parameter: Name,
        parameter_type: Option<Name>,
        // return_type: Option<Name>,
        closure: Box<Expression>,
    },
}

impl Literal {
    fn infer_type(&self, ctx: &mut TypingContext) -> Typing {
        match self {
            Self::Scalar(constant) => Ok(constant.infer_type()),
            Self::Tuple(elements) => Ok(Type::Compound(CompoundType::Product(ProductType::Tuple(
                elements
                    .into_iter()
                    .map(|e| e.infer_type(ctx))
                    .collect::<Result<_, _>>()?,
            )))),
            Self::Record(elements) => {
                Ok(Type::Compound(CompoundType::Product(ProductType::Record(
                    elements
                        .iter()
                        .map(|(name, e)| e.infer_type(ctx).map(|t| (name.clone(), t.clone())))
                        .collect::<Result<_, _>>()?,
                ))))
            }
            Self::Abstract {
                parameter,
                parameter_type,
                closure,
            } => {
                let domain = match parameter_type {
                    Some(type_name) => ctx
                        .type_for_name(type_name)
                        .cloned()
                        .ok_or_else(|| TypingError::UnknownType(type_name.clone()))?,
                    None => ctx
                        .symbol_type(parameter)
                        .cloned()
                        .unwrap_or_else(|| Type::Undecided(parameter.clone())),
                };

                let codomain =
                    ctx.assume_type_in(parameter, domain.clone(), |ctx| closure.infer_type(ctx))?;

                Ok(Type::Compound(CompoundType::Function {
                    domain: Box::new(domain),
                    codomain: Box::new(codomain),
                }))
            }
        }
    }
}

#[derive(Debug)]
enum Constant {
    Integer(isize),
    FloatingPoint(f64),
    Text(String),
    Boolean(bool),
}

impl Constant {
    fn infer_type(&self) -> Type {
        let ty = match self {
            Self::Integer(..) => TrivialType::Integer,
            Self::FloatingPoint(..) => TrivialType::FloatingPoint,
            Self::Text(..) => TrivialType::Text,
            Self::Boolean(..) => TrivialType::Boolean,
        };
        Type::Trivial(ty)
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct Name(String);

impl Name {
    fn fresh() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Type {
    Trivial(TrivialType),
    Compound(CompoundType),
    Undecided(Name),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TrivialType {
    Integer,
    FloatingPoint,
    Text,
    Boolean,
}

impl TrivialType {
    fn name(&self) -> Name {
        match self {
            Self::Integer => Name("int".to_owned()),
            Self::FloatingPoint => Name("float".to_owned()),
            Self::Text => Name("text".to_owned()),
            Self::Boolean => Name("boolean".to_owned()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CompoundType {
    Function {
        domain: Box<Type>,
        codomain: Box<Type>,
    },
    Sum(Vec<Type>),
    Product(ProductType),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProductType {
    Tuple(Vec<Type>),
    Record(Vec<(Name, Type)>),
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_scalar(c: Constant) -> Expression {
        Expression::Constant(Literal::Scalar(c))
    }

    #[test]
    fn constants() {
        let mut gamma = TypingContext::default();

        assert_eq!(
            Type::Trivial(TrivialType::Text),
            mk_scalar(Constant::Text("".to_owned()))
                .infer_type(&mut gamma)
                .unwrap()
        );

        assert_eq!(
            Type::Trivial(TrivialType::Boolean),
            mk_scalar(Constant::Boolean(true))
                .infer_type(&mut gamma)
                .unwrap()
        );

        assert_eq!(
            Type::Trivial(TrivialType::FloatingPoint),
            mk_scalar(Constant::FloatingPoint(1.0))
                .infer_type(&mut gamma)
                .unwrap()
        );

        assert_eq!(
            Type::Trivial(TrivialType::Integer),
            mk_scalar(Constant::Integer(1))
                .infer_type(&mut gamma)
                .unwrap()
        );
    }

    fn mk_id_fun(param: &str, tpe: TrivialType) -> Expression {
        Expression::Constant(Literal::Abstract {
            parameter: Name(param.to_owned()),
            parameter_type: Some(tpe.name()),
            closure: Box::new(Expression::Variable(Name(param.to_owned()))),
        })
    }

    #[test]
    fn simple_functions() {
        let mut gamma = TypingContext::default();

        let abs = mk_id_fun("x", TrivialType::Integer);

        let app = Expression::Apply {
            abstraction: Box::new(abs),
            term: Box::new(mk_scalar(Constant::Integer(1))),
        };

        assert_eq!(
            Type::Trivial(TrivialType::Integer),
            app.infer_type(&mut gamma).unwrap()
        )
    }
}
