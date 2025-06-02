use crate::{
    context::Linkage,
    interpreter::{Environment, Interpretation},
};

mod operators;

pub fn import(linkage: &mut Linkage) -> Interpretation<()> {
    //    import_std_file(env)?;
    types::import(linkage)?;
    operators::import(linkage)?;
    stdio::import(linkage)?;
    conversions::import(linkage)?;

    Ok(())
}

mod types {
    use crate::{
        context::Linkage,
        interpreter::Interpretation,
        typer::{self, Type, BASE_TYPES},
    };

    pub fn import(linkage: &mut Linkage) -> Interpretation<()> {
        for base_type in BASE_TYPES {
            linkage.bind_type(
                base_type.type_name().into(),
                typer::TypeScheme {
                    quantifiers: vec![],
                    body: Type::Constant(base_type.clone()),
                },
            );
        }

        Ok(())
    }
}

mod stdio {
    use crate::{ast::Identifier, bridge, context::Linkage, interpreter::Interpretation};

    pub fn print_endline(text: String) {
        println!("> {text}");
    }

    pub fn print(text: String) {
        print!("{text}");
    }

    pub fn import(context: &mut Linkage) -> Interpretation<()> {
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
        context::Linkage,
        interpreter::{Interpretation, Value},
    };

    pub fn show(value: Value) -> String {
        format!("{value}")
    }

    pub fn import(context: &mut Linkage) -> Interpretation<()> {
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
        context::Linkage,
        interpreter::{Base, RuntimeError, Value},
        stdlib,
        typer::EmptyAnnotation,
    };

    #[test]
    fn plus_i64() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("+")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Int(1)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Int(2)).into(),
            },
        );

        assert_eq!(
            Base::Int(3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }

    #[test]
    fn plus_f64() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("+")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Float(1.5)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Float(2.3)).into(),
            },
        );

        assert_eq!(
            Base::Float(1.5 + 2.3),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }

    #[test]
    fn plus_wrong_types() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("+")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Float(1.5)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Int(2)).into(),
            },
        );

        assert_eq!(
            Box::new(RuntimeError::InapplicableLamda2 {
                fst: Value::Base(Base::Float(1.5)),
                snd: Value::Base(Base::Int(2))
            }),
            e.reduce(&mut context.interpreter_environment).unwrap_err()
        );
    }

    #[test]
    fn minus() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("-")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Int(1)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Int(2)).into(),
            },
        );

        assert_eq!(
            Base::Int(-1),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }

    #[test]
    fn times() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("*")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Int(1)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Int(2)).into(),
            },
        );

        assert_eq!(
            Base::Int(2),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }

    #[test]
    fn divides() {
        let mut context = Linkage::default();
        stdlib::import(&mut context).unwrap();

        let e = E::Apply(
            EmptyAnnotation,
            Apply {
                function: E::Apply(
                    EmptyAnnotation,
                    Apply {
                        function: E::Variable(EmptyAnnotation, Identifier::new("/")).into(),
                        argument: E::Literal(EmptyAnnotation, Constant::Int(1)).into(),
                    },
                )
                .into(),
                argument: E::Literal(EmptyAnnotation, Constant::Int(2)).into(),
            },
        );

        assert_eq!(
            Base::Int(0),
            e.reduce(&mut context.interpreter_environment)
                .unwrap()
                .try_into_base_type()
                .unwrap()
        );
    }
}
