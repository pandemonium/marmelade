use marmelade::{
    lexer::{Layout, LexicalAnalyzer, TokenType},
    parser,
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
           |print_endline = fun s ->
           |    __print s
           |
           |main = fun _ -> create_window 1
           |"#,
    ));

    for t in lexer.tokens() {
        print!("{} ", t.token_type());
        if let TokenType::Layout(tt) = t.token_type() {
            match tt {
                Layout::Indent => print!("({})", t.location()),
                Layout::Dedent => print!("({})", t.location()),
                Layout::Newline => print!("({})", t.location()),
            }
        }
    }

    println!("{}", lexer.untokenize());

    let program = parser::parse_compilation_unit(lexer.tokens()).unwrap();
    println!("{program:?}");
}
