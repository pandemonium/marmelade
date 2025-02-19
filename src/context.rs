use crate::{interpreter::Environment, types::TypingContext};

#[derive(Debug, Clone, Default)]
pub struct CompilationContext {
    pub typing_context: TypingContext,
    pub interpreter_environment: Environment,
}
