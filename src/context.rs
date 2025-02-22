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
        //        if let CompilationUnit::Implicit(module) = &program {
        //            if let Declaration::Value { declarator, .. } = module
        //                .find_value_declaration(&Identifier::new("fibonacci"))
        //                .unwrap()
        //            {
        //                if let ValueDeclarator::Function(function) = declarator {
        //                    let x = self
        //                        .typing_context
        //                        .infer_type(
        //                            &function
        //                                .clone()
        //                                .into_lambda_tree(Identifier::new("fibonacci")),
        //                        )
        //                        .unwrap();
        //                    println!("Type of fibonacci: {:?}", x.inferred_type);
        //                }
        //            }
        //        }
        //
        Interpreter::new(Environment::make_child(self.interpreter_environment))
            .load_and_run(program.map(|_| ()))
    }
}
