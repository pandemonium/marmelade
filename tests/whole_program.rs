use marmelade::{
    interpreter::{Environment, Interpreter, Scalar},
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

    // The parser does not deal very well, but partial application works
    // This does create a dependency cycle though. It has seen create twice
    // probably, but this is actually fine
    lexer.tokenize(&into_input(
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
    ));
    //    lexer.tokenize(&into_input(
    //        r#"|create = create_window 20
    //           |
    //           |main = create 7 * create 7 / create 7
    //           |"#,
    //    ));

    let input = lexer.tokens();

    //    for Token(tt, loc) in input {
    //        println!("{} {}", loc, tt);
    //    }

    let program = parser::parse_compilation_unit(input).unwrap();

    let mut prelude = Environment::default();
    stdlib::import(&mut prelude).unwrap();

    let return_value = Interpreter::new(prelude).load_and_run(program).unwrap();

    assert_eq!(Scalar::Int(327), return_value.try_into_scalar().unwrap());
}
