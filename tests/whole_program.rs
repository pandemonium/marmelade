use marmelade::{
    interpreter::{Environment, Interpreter, Scalar},
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

    // Crashes in the parser
    lexer.tokenize(&into_input(
        r#"|create_window = fun x->1+2*x*8-1
           |
           |
           |main = create_window 20
           |"#,
    ));

    let program = parser::parse_compilation_unit(lexer.tokens()).unwrap();

    let mut prelude = Environment::default();
    stdlib::import(&mut prelude).unwrap();

    let return_value = Interpreter::new(prelude).load_and_run(program).unwrap();

    assert_eq!(Scalar::Int(320), return_value.try_into_scalar().unwrap());
}
