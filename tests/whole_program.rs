use marmelade::{
    interpreter::{Environment, Interpreter},
    lexer::{Layout, LexicalAnalyzer, TokenType},
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

    lexer.tokenize(&into_input(
        r#"|create_window =
           |    fun x ->
           |        1 + x
           |
           |
           |main = create_window 1
           |"#,
    ));

    let program = parser::parse_compilation_unit(lexer.tokens()).unwrap();

    let mut prelude = Environment::default();
    stdlib::define(&mut prelude).unwrap();

    let return_value = Interpreter::new(prelude).load_and_run(program).unwrap();

    println!("{return_value:?}");
}
