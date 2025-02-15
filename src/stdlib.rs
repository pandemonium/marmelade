use crate::{
    bridge,
    interpreter::{Environment, Interpretation, Scalar},
    lexer::Operator,
};

pub fn import(env: &mut Environment) -> Interpretation<()> {
    //    import_std_file(env)?;
    import_operator(env)?;

    Ok(())
}

fn import_std_file(env: &mut Environment) -> Interpretation<()> {
    todo!()
}

fn import_operator(env: &mut Environment) -> Interpretation<()> {
    bridge::define(
        Operator::Equals.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Bool(p), Scalar::Bool(q)) => Some(Scalar::Bool(p == q)),
            (Scalar::Text(p), Scalar::Text(q)) => Some(Scalar::Bool(p == q)),
            _otherwise => None,
        }),
        env,
    )?;
    bridge::define(
        Operator::Plus.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p + q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p + q)),
            _otherwise => None,
        }),
        env,
    )?;
    bridge::define(
        Operator::Minus.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p - q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p - q)),
            _otherwise => None,
        }),
        env,
    )?;
    bridge::define(
        Operator::Times.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p * q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p * q)),
            _otherwise => None,
        }),
        env,
    )?;
    bridge::define(
        Operator::Divides.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p / q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p / q)),
            _otherwise => None,
        }),
        env,
    )?;
    bridge::define(
        Operator::Modulo.id(),
        bridge::PartialRawLambda2(|p, q| match (p, q) {
            (Scalar::Int(p), Scalar::Int(q)) => Some(Scalar::Int(p % q)),
            (Scalar::Float(p), Scalar::Float(q)) => Some(Scalar::Float(p % q)),
            _otherwise => None,
        }),
        env,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Expression as E, Identifier},
        interpreter::{Environment, RuntimeError, Scalar},
        stdlib,
    };

    #[test]
    fn plus_i64() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn plus_f64() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn plus_wrong_types() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap_err()
        );
    }

    #[test]
    fn minus() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn times() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }

    #[test]
    fn divides() {
        let mut env = Environment::default();
        stdlib::import(&mut env).unwrap();

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
            e.reduce(&mut env).unwrap().try_into_scalar().unwrap()
        );
    }
}
