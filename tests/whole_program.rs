use marmelade::{
    ast::{Apply, Expression, Identifier, Lambda, Parameter},
    context::CompileState,
    interpreter::{Base, Closure, Environment, Interpreter, Value},
    lexer::LexicalAnalyzer,
    parser, stdlib,
};

fn into_input(source: &str) -> Vec<char> {
    source
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .collect()
}

#[test]
fn main1() {
    let mut lexer = LexicalAnalyzer::default();
    let source_text = into_input(
        r#"|create_window = fun x y ->
           |    let q = 1+2*x*8-1 + y in
           |       q
           |
           |create =
           |   create_window 20
           |create_window2 = fun x y ->1+2*x*8-1 + y
           |
           |main = create 7 * create 7 / create 7
           |"#,
    );
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = CompileState::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let return_value = Interpreter::new(compilation.interpreter_environment)
        .load_and_run(compilation.typing_context, program.map(|_| ()))
        .unwrap();

    assert_eq!(Base::Int(327), return_value.try_into_scalar().unwrap());
}

fn _make_fix_value(env: Environment) -> Value {
    Value::Closure(Closure {
        parameter: Identifier::new("f"),
        capture: env.clone(),
        body: Expression::Apply(
            (),
            Apply {
                function: Box::new(Expression::Lambda(
                    (),
                    Lambda {
                        parameter: Parameter::new(Identifier::new("x")),
                        body: Box::new(Expression::Apply(
                            (),
                            Apply {
                                function: Box::new(Expression::Variable((), Identifier::new("f"))),
                                argument: Box::new(Expression::Lambda(
                                    (),
                                    Lambda {
                                        parameter: Parameter::new(Identifier::new("y")),
                                        body: Box::new(Expression::Apply(
                                            (),
                                            Apply {
                                                function: Box::new(Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )),
                                                argument: Box::new(Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )),
                                            },
                                        )),
                                    },
                                )),
                            },
                        )),
                    },
                )),
                argument: Box::new(Expression::Lambda(
                    (),
                    Lambda {
                        parameter: Parameter::new(Identifier::new("x")),
                        body: Box::new(Expression::Apply(
                            (),
                            Apply {
                                function: Box::new(Expression::Variable((), Identifier::new("f"))),
                                argument: Box::new(Expression::Lambda(
                                    (),
                                    Lambda {
                                        parameter: Parameter::new(Identifier::new("y")),
                                        body: Box::new(Expression::Apply(
                                            (),
                                            Apply {
                                                function: Box::new(Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )),
                                                argument: Box::new(Expression::Variable(
                                                    (),
                                                    Identifier::new("x"),
                                                )),
                                            },
                                        )),
                                    },
                                )),
                            },
                        )),
                    },
                )),
            },
        ),
    })
}

#[test]
fn factorial20() {
    let mut lexer = LexicalAnalyzer::default();
    let source_text = into_input(
        r#"|factorial = fun x ->
           |  if x == 0 then
           |      1
           |  else
           |      let xx =
           |          x - 1
           |      in
           |          x * factorial xx
           |main = factorial 20
           |"#,
    );
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = CompileState::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let program_environment = compilation.interpreter_environment.into_parent();

    let return_value = Interpreter::new(program_environment)
        .load_and_run(compilation.typing_context, program.map(|_| ()));
    assert_eq!(
        Base::Int(2432902008176640000),
        return_value.unwrap().try_into_scalar().unwrap()
    );
}

#[test]
fn fibonacci23() {
    let mut lexer = LexicalAnalyzer::default();
    let source_text = into_input(
        r#"|fibonacci = fun x ->
           |  if 0 == x then
           |    0
           |  else
           |
           |    if 1 == x
           |      then print_endline "hi"; 1
           |      else
           |        let a = x - 1 in
           |        let b =
           |          x - 2
           |        in fibonacci a + fibonacci b
           |main = fibonacci 5
           |"#,
    );
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = CompileState::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    //    HMm, so it will correctly find type errors in a lot
    //    of places, but not if I put == "1"

    // This program must not type-check successfully because
    // the type of the apply does not type-check

    // Check that main is either a function that takes args
    // or a value. Both returning the Int type.
    let return_value = compilation.typecheck_and_interpret(program);

    assert_eq!(
        //        Base::Int(75025),
        Base::Int(5),
        return_value.unwrap().try_into_scalar().unwrap()
    );
}
