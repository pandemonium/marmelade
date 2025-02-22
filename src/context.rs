use crate::{
    ast::CompilationUnit,
    interpreter::{Environment, Interpreter, Loaded, Value},
    parser::ParsingInfo,
    types::TypingContext,
};

#[derive(Debug, Clone, Default)]
pub struct CompileState<'a> {
    pub source_text: &'a [char],
    pub typing_context: TypingContext,
    pub interpreter_environment: Environment,
}

impl<'a> CompileState<'a> {
    pub fn new(source_text: &'a [char]) -> Self {
        Self {
            source_text,
            ..Self::default()
        }
    }

    pub fn typecheck_and_interpret(self, program: CompilationUnit<ParsingInfo>) -> Loaded<Value> {
        Interpreter::new(self.interpreter_environment.into_parent())
            .load_and_run(self.typing_context, program.map(|_| ()))
    }
}
