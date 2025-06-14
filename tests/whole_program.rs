use marmelade::{
    context::Linkage,
    interpreter::{Base, Interpreter},
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
        r#"|create_window = lambda x y.
           |    let q = 1+2*x*8-1 + y in
           |       q
           |
           |create =
           |   create_window 20
           |create_window2 = lambda x y.1+2*x*8-1 + y
           |
           |main = create 7 * create 7 / create 7
           |"#,
    );
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = Linkage::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let return_value = Interpreter::new(compilation.interpreter_environment)
        .load_and_run(compilation.typing_context, program)
        .unwrap();

    assert_eq!(Base::Int(327), return_value.try_into_base_type().unwrap());
}

#[test]
fn factorial20() {
    let mut lexer = LexicalAnalyzer::default();
    let source_text = into_input(
        r#"|factorial = lambda x.
           |  if x = 0 then
           |      1
           |  else
           |      x * factorial (x - 1)
           |main = factorial 20
           |"#,
    );
    let program = parser::parse_compilation_unit(lexer.tokenize(&source_text)).unwrap();

    let mut compilation = Linkage::new(&source_text);
    stdlib::import(&mut compilation).unwrap();

    let program_environment = compilation.interpreter_environment.into_parent();

    let return_value =
        Interpreter::new(program_environment).load_and_run(compilation.typing_context, program);

    assert_eq!(
        Base::Int(2432902008176640000),
        return_value.unwrap().try_into_base_type().unwrap()
    );
}

#[test]
fn fibonacci23() {
    let source_text = into_input(
        r#"|fibonacci = lambda x.
           |  if 0 = x then
           |    0
           |  else
           |    if 1 = x
           |      then 1
           |      else fibonacci (x - 1) + fibonacci (x - 2)
           |Perhaps ::= forall a. This a | Nope
           |List ::= forall a. Cons a (List a) | Nil
           |main = fibonacci 18
           |"#,
    );

    let mut linkage = Linkage::new(&source_text);
    stdlib::import(&mut linkage).unwrap();

    let return_value = linkage.typecheck_and_interpret();

    // Thia has lost several magnitudes of performance and I do not know why.
    assert_eq!(
        Base::Int(2584),
        //        Base::Int(5),
        return_value.unwrap().try_into_base_type().unwrap()
    );
}
