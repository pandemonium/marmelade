use crate::ast::{CompilationUnit, Expression};

pub fn parse(source: &str) -> CompilationUnit {
    todo!()
}

pub struct Parser<'a> {
    tokens: &'a [char],
}

pub type Parse = Result<Expression, ParseError>;

pub enum ParseError {}

impl<'a> Parser<'a> {
    fn parse_prefix(&mut self) -> Parse {
        todo!()
    }
}
