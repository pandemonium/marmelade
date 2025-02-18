use std::rc::Rc;

use marmelade::{
    ast::{Expression, Identifier, Parameter},
    interpreter::{Closure, Environment, Interpreter, Scalar, Value},
    lexer::{LexicalAnalyzer, Token},
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
    let program = parser::parse_compilation_unit(lexer.tokenize(&into_input(
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
    )))
    .unwrap();

    let mut prelude = Environment::default();
    stdlib::import(&mut prelude).unwrap();

    let return_value = Interpreter::new(prelude).load_and_run(program).unwrap();

    assert_eq!(Scalar::Int(327), return_value.try_into_scalar().unwrap());
}

fn make_fix_value(env: Environment) -> Value {
    Value::Closure(Closure {
        parameter: Identifier::new("f"),
        capture: env.clone(),
        body: Expression::Apply {
            function: Box::new(Expression::Lambda {
                parameter: Parameter::new(Identifier::new("x")),
                body: Box::new(Expression::Apply {
                    function: Box::new(Expression::Variable(Identifier::new("f"))),
                    argument: Box::new(Expression::Lambda {
                        parameter: Parameter::new(Identifier::new("y")),
                        body: Box::new(Expression::Apply {
                            function: Box::new(Expression::Variable(Identifier::new("x"))),
                            argument: Box::new(Expression::Variable(Identifier::new("x"))),
                        }),
                    }),
                }),
            }),
            argument: Box::new(Expression::Lambda {
                parameter: Parameter::new(Identifier::new("x")),
                body: Box::new(Expression::Apply {
                    function: Box::new(Expression::Variable(Identifier::new("f"))),
                    argument: Box::new(Expression::Lambda {
                        parameter: Parameter::new(Identifier::new("y")),
                        body: Box::new(Expression::Apply {
                            function: Box::new(Expression::Variable(Identifier::new("x"))),
                            argument: Box::new(Expression::Variable(Identifier::new("x"))),
                        }),
                    }),
                }),
            }),
        },
    })
}

#[test]
fn factorial22() {
    let mut lexer = LexicalAnalyzer::default();
    // Dependency resolution probably gets stuck now that there is a cycle.
    let program = parser::parse_compilation_unit(lexer.tokenize(&into_input(
        r#"|factorial = fun x ->
           |  if x == 0 then 1 else let xx = x - 1 in x * factorial xx
           |
           |fibonacci = fun x ->
           |  if x == 0 then 1 else if x == 1 then 1 else let a = x - 1 in let b = x - 2 in fibonacci a + fibonacci b
           |
           |
           |main = fibonacci 25
           |"#,
    )))
    .unwrap();

    println!("{program}");

    let mut prelude = Environment::default();
    stdlib::import(&mut prelude).unwrap();

    let program_environment = Environment::make_child(Rc::new(prelude));

    let return_value = Interpreter::new(program_environment).load_and_run(program);
    assert_eq!(
        Scalar::Int(120),
        return_value.unwrap().try_into_scalar().unwrap()
    );
}
