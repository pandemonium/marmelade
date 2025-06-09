use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use super::{Environment, ResolutionError, Resolved};
use crate::{
    ast::{
        Declaration, DeconstructInto, Expression, Identifier, ImportModule, ModuleDeclarator,
        ValueDeclaration, ValueDeclarator, Variable,
    },
    interpreter::{Base, Value},
    typer::{Parsed, TypeChecker},
};

// Untyped instead.
// It ought to take a CompilationUnit and also be called something else.
// How about Compiler?
pub struct ModuleResolver<'a, A> {
    module: &'a ModuleDeclarator<A>,
    dependency_graph: DependencyGraph<'a>,
    resolved: Environment,
}

impl<'a, A> ModuleResolver<'a, A>
where
    A: fmt::Debug + fmt::Display + Copy + Parsed,
{
    pub fn initialize(module: &'a ModuleDeclarator<A>, prelude: Environment) -> Resolved<Self> {
        let dependency_graph = module.dependency_graph();

        if !dependency_graph.is_acyclic() {
            Err(ResolutionError::DependencyCycle)?;
        }

        dependency_graph.check_satisfiable()?;

        Ok(Self {
            module,
            dependency_graph,
            resolved: prelude,
        })
    }

    pub fn type_check(self, mut type_checker: TypeChecker) -> Resolved<Self> {
        for id in self.dependency_graph.compute_resolution_order() {
            if let Some(declaration) = self.module.find_value_declaration(id).cloned() {
                println!("type_check: `{id}` ...");
                // Lose the map call here. The resolver must only work with ParsingInfo
                // from the start.
                type_checker.check_declaration(&declaration.map(|a| *a.info()))?;
            }
        }
        Ok(self)
    }

    pub fn into_environment(mut self) -> Resolved<Environment> {
        for id in self.dependency_graph.compute_resolution_order() {
            self.resolve_declaration(id)?;
        }
        Ok(self.resolved)
    }

    fn resolve_declaration(&mut self, id: &Identifier) -> Resolved<()> {
        if let Some(ValueDeclaration { declarator, .. }) = self.module.find_value_declaration(id) {
            self.resolve_binding(id, declarator)
        } else if self.resolved.is_defined(id) {
            Ok(())
        } else {
            panic!("Unable to resolve declaration: `{id}` - not implemented")
        }
    }

    fn resolve_binding(
        &mut self,
        id: &Identifier,
        declarator: &ValueDeclarator<A>,
    ) -> Resolved<()> {
        // That this has to clone the Expressions is not ideal
        let env = &mut self.resolved;

        // Re-write the expression here?
        let value = declarator
            .expression
            .clone()
            .de_bruijnify(&mut env.clone())
            .reduce(env)?;

        env.insert_binding(id.clone(), value);

        Ok(())
    }
}

impl<A> Expression<A>
where
    A: fmt::Debug + fmt::Display + Clone + Parsed,
{
    pub fn de_bruijnify(self, env: &mut Environment) -> Self {
        match self {
            Self::TypeAscription(annotation, ascription) => Self::TypeAscription(
                annotation,
                ascription.map_expression(|e| e.de_bruijnify(env)),
            ),
            Self::Variable(annotation, Variable::Identifier(binder)) => Self::Variable(
                annotation,
                Variable::Index({
                    let index = env
                        .de_bruijn_index(&binder)
                        .expect(&format!("{binder} to exist"));

                    println!("de_bruijnify: {binder} is at #{index}");

                    index
                }),
            ),
            Self::InvokeBridge(annotation, Variable::Identifier(binder)) => Self::InvokeBridge(
                annotation,
                Variable::Index(
                    env.de_bruijn_index(&binder)
                        .expect(&format!("{binder} to exist")),
                ),
            ),
            Self::Interpolation(annotation, interpolate) => Self::Interpolation(
                annotation,
                interpolate.map_expression(|e| e.de_bruijnify(env)),
            ),
            Self::SelfReferential(annotation, self_referential) => {
                //
                Self::SelfReferential(annotation, {
                    let parameter_name = self_referential.parameter.name.clone();
                    env.insert_binding(parameter_name.clone(), Value::Base(Base::Unit));
                    let v = self_referential.map_expression(|e| e.de_bruijnify(env));
                    env.remove_binding(&parameter_name);
                    v
                })
            }
            Self::Lambda(annotation, lambda) => Self::Lambda(annotation, {
                let parameter_name = lambda.parameter.name.clone();
                env.insert_binding(parameter_name.clone(), Value::Base(Base::Unit));
                let v = lambda.map_expression(|e| e.de_bruijnify(env));
                env.remove_binding(&parameter_name);
                v
            }),
            Self::Apply(annotation, apply) => {
                Self::Apply(annotation, apply.map_expression(|e| e.de_bruijnify(env)))
            }
            Self::Inject(annotation, inject) => {
                Self::Inject(annotation, inject.map_expression(|e| e.de_bruijnify(env)))
            }
            Self::Product(annotation, product) => {
                Self::Product(annotation, product.map_expression(|e| e.de_bruijnify(env)))
            }
            Self::Project(annotation, project) => {
                Self::Project(annotation, project.map_expression(|e| e.de_bruijnify(env)))
            }
            Self::Binding(annotation, binding) => Self::Binding(annotation, {
                let binder = binding.binder.clone();
                env.insert_binding(binder.clone(), Value::Base(Base::Unit));
                let v = binding.map_expression(|e| e.de_bruijnify(env));
                env.remove_binding(&binder);
                v
            }),
            Self::Sequence(annotation, sequence) => {
                Self::Sequence(annotation, sequence.map_expression(|e| e.de_bruijnify(env)))
            }
            Self::ControlFlow(annotation, control_flow) => Self::ControlFlow(
                annotation,
                control_flow.map_expression(|e| e.de_bruijnify(env)),
            ),
            Self::DeconstructInto(annotation, deconstruct_into) => {
                Self::DeconstructInto(annotation, {
                    DeconstructInto {
                        scrutinee: deconstruct_into.scrutinee.de_bruijnify(env).into(),
                        match_clauses: deconstruct_into
                            .match_clauses
                            .into_iter()
                            .map(|clause| {
                                let pattern = clause.pattern.clone();
                                let bindings = pattern.bindings();
                                for binder in &bindings {
                                    println!("de_bruijnify: insert {binder}");
                                    env.insert_binding((*binder).clone(), Value::Base(Base::Unit));
                                }
                                let v = clause.map_expression(|e| e.de_bruijnify(env));
                                for binder in bindings.iter().rev() {
                                    println!("de_bruijnify: remove {binder}");
                                    env.remove_binding(binder);
                                }
                                v
                            })
                            .collect(),
                    }
                })
            }
            otherwise => otherwise,
        }
    }
}

#[derive(Debug)]
pub struct DependencyGraph<'a> {
    dependencies: Vec<(&'a Identifier, Vec<&'a Identifier>)>,
}

impl<'a> DependencyGraph<'a> {
    pub fn from_declarations<A>(decls: &'a [Declaration<A>]) -> Self
    where
        A: fmt::Display + fmt::Debug + Copy + Parsed,
    {
        let mut outbound = Vec::with_capacity(decls.len());

        for decl in decls {
            match decl {
                Declaration::Value(_, value) => {
                    outbound.push((&value.binder, value.dependencies().into_iter().collect()));
                }
                Declaration::ImportModule(
                    _,
                    ImportModule {
                        exported_symbols, ..
                    },
                ) => {
                    for dep in exported_symbols {
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

            for (node, edges) in &mut graph {
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

    pub fn is_wellformed(&'a self) -> bool {
        self.is_acyclic() && self.is_satisfiable()
    }

    pub fn is_satisfiable(&'a self) -> bool {
        self.dependencies.iter().all(|(_, deps)| {
            deps.iter()
                .all(|&dep| self.dependencies.iter().any(|(key, _)| *key == dep))
        })
    }

    pub fn check_satisfiable(&self) -> Resolved<()> {
        // union all snd, compare to fst
        for (root, deps) in &self.dependencies {
            for dep in deps {
                if !self.dependencies.iter().any(|(root1, _)| root1 == dep) {
                    // I need SourceLocation here
                    // Identifier should contain it, really. How hard would that be?
                    // Identifier::new has to pass in a SourceLocation (or ParsingInfo.)
                    Err(ResolutionError::UnsatisfiedDependency {
                        root: (*root).clone(),
                        dependency: (*dep).clone(),
                    })?;
                }
            }
        }

        Ok(())
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
            .copied()
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

impl fmt::Display for DependencyGraph<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (dep, deps) in &self.dependencies {
            write!(f, "{dep}")?;

            if !deps.is_empty() {
                write!(f, ": \t\t[{}", &deps[0])?;
                for dep in &deps[1..] {
                    write!(f, ", {dep}")?;
                }
                writeln!(f, "]")?;
            } else {
                writeln!(f, "\t\t (nothing)")?;
            }
        }

        Ok(())
    }
}
