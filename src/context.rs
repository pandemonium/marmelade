use std::marker::PhantomData;

use crate::{
    ast::{CompilationUnit, Identifier},
    interpreter::{DependencyGraph, Environment, Interpreter, Resolved, Value},
    lexer::LexicalAnalyzer,
    parser::{self, ParsingInfo},
    typer::{Binding, TypeScheme, TypingContext},
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

    pub fn bind_value(&mut self, binder: Identifier, bound: Value) {
        self.interpreter_environment.insert_binding(binder, bound);
    }

    pub fn bind_type(&mut self, binder: Binding, scheme: TypeScheme) {
        self.typing_context.bind(binder, scheme);
    }

    pub fn typecheck_and_interpret(self) -> Resolved<Value> {
        let mut lexer = LexicalAnalyzer::default();
        let input = lexer.tokenize(self.main_source_text);
        let program = parser::parse_compilation_unit(input)?.scope_library_modules();

        Interpreter::new(self.interpreter_environment.into_parent())
            .load_and_run(self.typing_context, program)
    }

    //    pub fn frobnicate(self) -> Resolved<Value> {
    //        let mut lexer = LexicalAnalyzer::default();
    //        let input = lexer.tokenize(self.main_source_text);
    //        let program = parser::parse_compilation_unit(input)?.scope_library_modules();
    //
    //
    //
    //        todo!()
    //    }
}

pub struct Start;

pub struct Compilation<'a, Stage> {
    program: &'a CompilationUnit<ParsingInfo>,
    dependency_graph: DependencyGraph<'a>,
    environment: Environment,
    _stage: PhantomData<Stage>,
}

impl<'a, A> Compilation<'a, A> {
    pub fn new(
        program: &'a CompilationUnit<ParsingInfo>,
        environment: Environment,
    ) -> Compilation<'a, Start> {
        Compilation {
            program,
            dependency_graph: program.dependency_graph(),
            environment,
            _stage: PhantomData::default(),
        }
    }
}
