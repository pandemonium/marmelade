use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use super::{Environment, LoadError, Loaded};
use crate::{
    ast::{
        Declaration, Identifier, ImportModule, ModuleDeclarator, ValueDeclaration, ValueDeclarator,
    },
    typer::{Parsed, TypeError, TypeInference, TypeScheme, Typing, TypingContext},
};

pub struct ModuleLoader<'a, A> {
    module: &'a ModuleDeclarator<A>,
    dependency_graph: DependencyGraph<'a>,
    //    type_dependencies: DependencyGraph<'a>,
    initialized: Environment,
}

impl<'a, A> ModuleLoader<'a, A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn try_loading(module: &'a ModuleDeclarator<A>, prelude: Environment) -> Loaded<Self> {
        let resolver = |id: &Identifier| prelude.is_defined(id);

        let dependency_graph = module.dependency_graph();

        if !dependency_graph.is_wellformed(resolver) {
            if !dependency_graph.is_acyclic() {
                Err(LoadError::DependencyCycle)
            } else if !dependency_graph.is_satisfiable(resolver) {
                Err(LoadError::UnsatisfiedDependencies)
            } else {
                unreachable!()
            }
        } else {
            Ok(Self {
                module,
                dependency_graph,
                initialized: prelude,
            })
        }
    }

    pub fn type_check(self, mut typing_context: TypingContext) -> Loaded<Self> {
        for id in self.dependency_graph.compute_resolution_order().drain(..) {
            if self.module.find_value_declaration(id).is_some() {
                println!("type_check: `{id}` ...");
                let declaration_type = self
                    .infer_declaration_type(id, &typing_context)?
                    .inferred_type;

                let scheme = TypeScheme {
                    quantifiers: declaration_type.free_variables().iter().cloned().collect(),
                    body: declaration_type,
                };

                println!("type_check: `{id}` `{scheme}`");

                typing_context.bind(id.clone().into(), scheme);
            }
        }

        Ok(self)
    }

    pub fn initialize(mut self) -> Loaded<Environment> {
        for id in self.dependency_graph.compute_resolution_order().drain(..) {
            self.initialize_declaration(id)?
        }

        Ok(self.initialized)
    }

    fn initialize_declaration(&mut self, id: &Identifier) -> Loaded<()> {
        if let Some(Declaration::Value(_, ValueDeclaration { declarator, .. })) =
            self.module.find_value_declaration(id)
        {
            self.initialize_binding(id, declarator)
        } else if self.initialized.is_defined(id) {
            Ok(())
        } else {
            panic!("Unable to resolve declaration: `{id}` - not implemented")
        }
    }

    fn initialize_binding(
        &mut self,
        id: &Identifier,
        declarator: &ValueDeclarator<A>,
    ) -> Loaded<()> {
        // That this has to clone the Expressions is not ideal

        let expression = match declarator.clone() {
            ValueDeclarator::Constant(constant) => constant.initializer,
            ValueDeclarator::Function(function) => function.into_lambda_tree(id.clone()),
        };

        println!("initialize_binding: {id}");

        let env = &mut self.initialized;
        let value = expression.reduce(env)?;
        env.insert_binding(id.clone(), value);

        Ok(())
    }

    fn infer_declaration_type(
        &self,
        id: &Identifier,
        typing_context: &TypingContext,
    ) -> Loaded<TypeInference> {
        let declaration = self
            .module
            .find_value_declaration(id)
            .ok_or_else(|| TypeError::UndefinedSymbol(id.clone()))?;

        match declaration {
            Declaration::Value(_, declaration) => Ok(self.infer_declarator_type(
                &declaration.binder,
                &declaration.declarator,
                typing_context,
            )?),
            _otherwise => todo!(),
        }
    }

    // Change into ParsingInfo
    // A TypeInference contains substitutions. Can I use these?
    fn infer_declarator_type(
        &self,
        binder: &Identifier,
        declarator: &ValueDeclarator<A>,
        typing_context: &TypingContext,
    ) -> Typing
    where
        A: fmt::Display + Parsed,
    {
        let expression = match declarator.clone() {
            ValueDeclarator::Constant(constant) => constant.initializer,
            ValueDeclarator::Function(function) => {
                // The fact that this has to always happen means that it ought
                // not be duplicated.
                function.into_lambda_tree(binder.clone())
            }
        };

        println!("infer_declarator_type: {}", expression);

        typing_context.infer_type(&expression)
    }
}

#[derive(Debug)]
pub struct DependencyGraph<'a> {
    dependencies: HashMap<&'a Identifier, Vec<&'a Identifier>>,
}

impl<'a> DependencyGraph<'a> {
    pub fn from_declarations<A>(decls: &'a [Declaration<A>]) -> Self
    where
        A: Clone,
    {
        let mut outbound = HashMap::with_capacity(decls.len());

        for decl in decls {
            match decl {
                Declaration::Value(_, value) => {
                    outbound.insert(
                        &value.binder,
                        value.declarator.dependencies().drain().collect(),
                    );
                }
                Declaration::ImportModule(
                    _,
                    ImportModule {
                        exported_symbols, ..
                    },
                ) => {
                    for dep in exported_symbols {
                        outbound.entry(dep).or_default();
                    }
                }
                _otherwise => (),
            }
        }

        Self {
            dependencies: outbound,
        }
    }

    // Think about whether or not this consumes self.
    pub fn compute_resolution_order(&self) -> Vec<&'a Identifier> {
        let mut boofer = Vec::with_capacity(self.dependencies.len());

        let mut graph = self
            .dependencies
            .iter()
            .map(|(&node, edges)| {
                (
                    node,
                    edges
                        .iter()
                        .filter(|&&edge| edge != node)
                        .copied()
                        .collect::<HashSet<_>>(),
                )
            })
            .collect::<Vec<_>>();

        // Look in to doing away with this go-between structure and make the lookups
        // directly in graph instead
        let mut in_degrees = graph
            .iter()
            .map(|(node, edges)| (*node, edges.len()))
            .collect::<HashMap<_, _>>();

        let mut queue = in_degrees
            .iter()
            .filter_map(|(&node, in_degree)| (*in_degree == 0).then_some(node))
            .collect::<VecDeque<_>>();

        while let Some(independent) = queue.pop_front() {
            boofer.push(independent);

            for (node, edges) in graph.iter_mut() {
                if edges.remove(independent) {
                    if let Some(count) = in_degrees.get_mut(node) {
                        *count -= 1;
                        if *count == 0 {
                            queue.push_back(node);
                        }
                    }
                }
            }

            in_degrees.remove(independent);
        }

        boofer
    }

    pub fn is_wellformed<F>(&'a self, is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        /*self.is_acyclic() &&*/
        self.is_satisfiable(is_external)
    }

    pub fn is_satisfiable<F>(&'a self, mut is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        self.dependencies.values().all(|deps| {
            deps.iter().all(|dep| {
                let retval = self.dependencies.contains_key(dep) || is_external(dep);

                if !retval {
                    println!("is_satisfiable: `{dep}` not found");
                    println!("is_satisfiable: {:?}", self.dependencies)
                }

                retval
            })
        })
    }

    pub fn is_acyclic(&self) -> bool {
        let mut visited = HashSet::new();

        self.dependencies
            .keys()
            .all(|child| !self.is_cyclic(child, &mut visited, &mut HashSet::new()))
    }

    fn is_cyclic(
        &self,
        node: &'a Identifier,
        seen: &mut HashSet<&'a Identifier>,
        path: &mut HashSet<&'a Identifier>,
    ) -> bool {
        if path.contains(node) {
            path.len() > 1
        //            true
        } else if seen.contains(node) {
            false
        } else {
            path.insert(node);
            seen.insert(node);

            let has_cycle = self
                .dependencies(node)
                .unwrap_or_default()
                .iter()
                .any(|&child| self.is_cyclic(child, seen, path));

            path.remove(node);

            has_cycle
        }
    }

    pub fn nodes(&self) -> Vec<&'a Identifier> {
        self.dependencies.keys().cloned().collect()
    }

    pub fn find<F>(&'a self, mut p: F) -> Option<&'a &'a Identifier>
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies.keys().find(|id| p(id))
    }

    pub fn satisfies<F>(&'a self, mut p: F) -> bool
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies.keys().all(|id| p(id))
    }

    pub fn dependencies(&self, d: &'a Identifier) -> Option<&[&'a Identifier]> {
        self.dependencies.get(d).map(Vec::as_slice)
    }
}
