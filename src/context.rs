use crate::{interpreter::Environment, types::TypingContext};

#[derive(Debug, Clone, Default)]
pub struct InterpretationContext {
    pub typing_context: TypingContext,
    pub interpreter_environment: Environment,
}
