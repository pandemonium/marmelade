use crate::{
    interpreter::{Environment, Interpreter, Loaded, Value},
    lexer::LexicalAnalyzer,
    parser,
    typer::TypingContext,
};

#[derive(Debug, Clone, Default)]
pub struct Linkage<'a> {
    pub main_source_text: &'a [char],
    pub typing_context: TypingContext,
    pub interpreter_environment: Environment,
}

impl<'a> Linkage<'a> {
    pub fn new(source_text: &'a [char]) -> Self {
        Self {
            main_source_text: source_text,
            ..Self::default()
        }
    }

    pub fn typecheck_and_interpret(self) -> Loaded<Value> {
        let mut lexer = LexicalAnalyzer::default();
        let program = parser::parse_compilation_unit(lexer.tokenize(&self.main_source_text))?;

        Interpreter::new(self.interpreter_environment.into_parent())
            .load_and_run(self.typing_context, program)
    }
}
