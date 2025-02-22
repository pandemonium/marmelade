use crate::{
    context::CompileState,
    interpreter::{Environment, Interpretation},
};

mod operators;

pub fn import(context: &mut CompileState) -> Interpretation<()> {
    //    import_std_file(env)?;
    operators::import(context)?;
    stdio::import(context)?;
    conversions::import(context)?;

    Ok(())
}

mod stdio {
    use crate::{ast::Identifier, bridge, context::CompileState, interpreter::Interpretation};

    pub fn print_endline(text: String) {
        println!("{text}");
    }

    pub fn print(text: String) {
        print!("{text}");
    }

    pub fn import(context: &mut CompileState) -> Interpretation<()> {
        bridge::define(
            Identifier::new("print_endline"),
            bridge::Lambda1(print_endline),
            context,
        )?;

        bridge::define(Identifier::new("print"), bridge::Lambda1(print), context)?;

        Ok(())
    }
}

mod conversions {
    use crate::{
        ast::Identifier,
        bridge,
        context::CompileState,
        interpreter::{Interpretation, Value},
    };

    pub fn show(value: Value) -> String {
        format!("{value}")
    }

    pub fn import(context: &mut CompileState) -> Interpretation<()> {
        bridge::define(Identifier::new("show"), bridge::RawLambda1(show), context)?;

        Ok(())
    }
}

fn _import_std_file(_env: &mut Environment) -> Interpretation<()> {
    todo!()
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{Apply, Constant, Expression as E, Identifier},
        context::CompileState,
        interpreter::{Base, RuntimeError},
        stdlib,
    };

    #[test]
    fn plus_i64() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        });

        assert_eq!(
            Base::Int(3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn plus_f64() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Float(1.5)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Float(2.3)).into(),
        });

        assert_eq!(
            Base::Float(1.5 + 2.3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn plus_wrong_types() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("+")).into(),
                argument: E::Literal(Constant::Float(1.5)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        });

        assert_eq!(
            RuntimeError::InapplicableLamda2,
            e.reduce(&mut context.interpreter_environment).unwrap_err()
        );
    }

    #[test]
    fn minus() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("-")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        });

        assert_eq!(
            Base::Int(-1),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn times() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("*")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        });

        assert_eq!(
            Base::Int(2),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }

    #[test]
    fn divides() {
        let mut context = CompileState::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(Apply {
            function: E::Apply(Apply {
                function: E::Variable(Identifier::new("/")).into(),
                argument: E::Literal(Constant::Int(1)).into(),
            })
            .into(),
            argument: E::Literal(Constant::Int(2)).into(),
        });

        assert_eq!(
            Base::Int(0),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_scalar()
                .unwrap()
        );
    }
}
