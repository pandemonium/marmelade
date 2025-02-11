use crate::{
    bridge::{self, Lambda2},
    interpreter::{Environment, Interpretation},
    lexer::Operator,
};

pub fn import(env: &mut Environment) -> Interpretation<()> {
    // Could I polymorph this even more by traiting up the underlying operators?
    // A: Add, A: Mul, etc?
    //    OperatorBridge::<i64, _>::new(Operator::Plus, |p, q| p + q).define(env)?;
    //    OperatorBridge::<i64, _>::new(Operator::Minus, |p, q| p - q).define(env)?;
    //    OperatorBridge::<i64, _>::new(Operator::Times, |p, q| p * q).define(env)?;
    //    OperatorBridge::<i64, _>::new(Operator::Divides, |p, q| p / q).define(env)?;

    // See if this can be golfed.
    bridge::define(
        Operator::Plus.id(),
        bridge::Lambda2(|p: i64, q: i64| p + q),
        env,
    )?;
    bridge::define(
        Operator::Minus.id(),
        bridge::Lambda2(|p: i64, q: i64| p - q),
        env,
    )?;
    bridge::define(
        Operator::Times.id(),
        bridge::Lambda2(|p: i64, q: i64| p * q),
        env,
    )?;
    bridge::define(
        Operator::Divides.id(),
        bridge::Lambda2(|p: i64, q: i64| p / q),
        env,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Constant, Expression as E, Identifier},
        interpreter::{Environment, Scalar},
        stdlib,
    };

    #[test]
    fn plus() {
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
