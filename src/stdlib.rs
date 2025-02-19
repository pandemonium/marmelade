use crate::{
    bridge,
    context::CompilationContext,
    interpreter::{Environment, Interpretation},
    lexer::Operator,
    types::{TrivialType, Type, TypeParameter},
};

pub fn import(context: &mut CompilationContext) -> Interpretation<()> {
    //    import_std_file(env)?;
    import_operators(context)?;

    Ok(())
}

fn _import_std_file(_env: &mut Environment) -> Interpretation<()> {
    todo!()
}

fn make_binary_boolean_operator() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Function(
            tp.clone(),
            Type::Function(tp, Type::Trivial(TrivialType::Bool).into()).into(),
        )
        .into(),
    )
}

fn make_binary_operator() -> Type {
    let ty = TypeParameter::fresh();
    let tp = Box::new(Type::Parameter(ty.clone()));
    Type::Forall(
        ty.clone(),
        Type::Function(tp.clone(), Type::Function(tp.clone(), tp).into()).into(),
    )
}

mod operator {
    use crate::interpreter::Scalar;

    pub fn equals(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Bool(p), Scalar::Bool(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Text(p), Scalar::Text(q)) => Some(Scalar::Bool(p == q)),
            _otherwise => None,
        }
    }

    pub fn plus(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p + q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p + q)),
            _otherwise => None,
        }
    }

    pub fn minus(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p - q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p - q)),
            _otherwise => None,
        }
    }

    pub fn times(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p * q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p * q)),
            _otherwise => None,
        }
    }

    pub fn divides(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p / q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p / q)),
            _otherwise => None,
        }
    }

    pub fn modulo(p: Scalar, q: Scalar) -> Option<Scalar> {
        match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p % q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p % q)),
            _otherwise => None,
        }
    }
}

fn import_operators(env: &mut CompilationContext) -> Interpretation<()> {
    use operator::*;

    bridge::define(
        Operator::Equals.id(),
        bridge::PartialRawLambda2 {
            apply: equals,
            signature: make_binary_boolean_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Plus.id(),
        bridge::PartialRawLambda2 {
            apply: plus,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Minus.id(),
        bridge::PartialRawLambda2 {
            apply: minus,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Times.id(),
        bridge::PartialRawLambda2 {
            apply: times,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Divides.id(),
        bridge::PartialRawLambda2 {
            apply: divides,
            signature: make_binary_operator(),
        },
        env,
    )?;

    bridge::define(
        Operator::Modulo.id(),
        bridge::PartialRawLambda2 {
            apply: modulo,
            signature: make_binary_operator(),
        },
        env,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Expression as E, Identifier},
        context::CompilationContext,
        interpreter::{RuntimeError, Scalar},
        stdlib,
    };

    #[test]
    fn plus_i64() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn plus_f64() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Float(1.5)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Float(2.3)).into(),
        };

        assert_eq!(
            Scalar::Float(1.5 + 2.3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn plus_wrong_types() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Float(1.5)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            RuntimeError::InapplicableLamda2,
            e.reduce(&mut context.interpreter_environment).unwrap_err()
        );
    }

    #[test]
    fn minus() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("-")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(-1),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn times() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("*")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(2),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn divides() {
        let mut context = CompilationContext::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply {
            function: E::Apply {
                function: E::Variable(Identifier::new("/")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            }
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        };

        assert_eq!(
            Scalar::Int(0),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }
}
