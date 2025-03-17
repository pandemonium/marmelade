use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use super::{Environment, LoadError, Loaded};
use crate::{
    ast::{
        Declaration, Identifier, ImportModule, ModuleDeclarator, ValueDeclaration, ValueDeclarator,
    },
    typer::{self, Parsed, TypingContext},
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
        for id in self.dependency_graph.compute_resolution_order() {
            if let Some(declaration) = self.module.find_value_declaration(id) {
                println!("type_check: `{id}` ...");

                let declaration_type =
                    typer::infer_declaration_type(declaration, &typing_context)?.inferred_type;

                let scheme = declaration_type.generalize(&typing_context);

                println!("type_check: `{id}` `{scheme}`");

                typing_context.bind(id.clone().into(), scheme);
            }
        }

        Ok(self)
    }

    pub fn initialize(mut self) -> Loaded<Environment> {
        for id in self.dependency_graph.compute_resolution_order() {
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

        let env = &mut self.initialized;
        let value = declarator.expression.clone().reduce(env)?;
        env.insert_binding(id.clone(), value);

        Ok(())
    }
}

#[derive(Debug)]
pub struct DependencyGraph<'a> {
    //    dependencies: HashMap<&'a Identifier, Vec<&'a Identifier>>,
    dependencies: Vec<(&'a Identifier, Vec<&'a Identifier>)>,
}

impl<'a> DependencyGraph<'a> {
    pub fn from_declarations<A>(decls: &'a [Declaration<A>]) -> Self
    where
        A: fmt::Debug + Clone,
    {
        let mut outbound = Vec::with_capacity(decls.len());

        for decl in decls {
            match decl {
                Declaration::Value(_, value) => {
                    outbound.push((
                        &value.binder,
                        value.declarator.dependencies().into_iter().collect(),
                    ));
                }
                Declaration::ImportModule(
                    _,
                    ImportModule {
                        exported_symbols, ..
                    },
                ) => {
                    for dep in exported_symbols {
                        //                        outbound.entry(dep).or_default();
                        outbound.push((dep, vec![]));
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
        let mut boofer: Vec<&'a Identifier> = Vec::with_capacity(self.dependencies.len());

        let mut graph = self
            .dependencies
            .iter()
            .map(|(node, edges)| {
                (
                    *node,
                    edges
                        .iter()
                        .filter(|&&edge| edge != *node)
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
            .filter_map(|(node, in_degree)| (*in_degree == 0).then_some(*node))
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
        self.is_acyclic() && self.is_satisfiable(is_external)
    }

    pub fn is_satisfiable<F>(&'a self, mut is_external: F) -> bool
    where
        F: FnMut(&Identifier) -> bool,
    {
        self.dependencies.iter().all(|(_, deps)| {
            deps.iter().all(|&dep| {
                let retval =
                    self.dependencies.iter().any(|(key, _)| *key == dep) || is_external(dep);

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
            .iter()
            .all(|(child, _)| !self.is_cyclic(child, &mut visited, &mut HashSet::new()))
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

            if has_cycle {
                println!("is_cyclic: {node}");
            }

            has_cycle
        }
    }

    pub fn nodes(&self) -> Vec<&'a Identifier> {
        self.dependencies
            .iter()
            .map(|(key, _)| key)
            .cloned()
            .collect()
    }

    pub fn find<F>(&'a self, mut p: F) -> Option<&'a &'a Identifier>
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies
            .iter()
            .map(|(key, _)| key)
            .find(|id| p(id))
    }

    pub fn satisfies<F>(&'a self, mut p: F) -> bool
    where
        F: FnMut(&'a Identifier) -> bool,
    {
        self.dependencies.iter().map(|(key, _)| key).all(|id| p(id))
    }

    pub fn dependencies(&self, d: &'a Identifier) -> Option<&[&'a Identifier]> {
        self.dependencies
            .iter()
            .find_map(|(key, deps)| (*key == d).then_some(deps.as_slice()))
    }
}
