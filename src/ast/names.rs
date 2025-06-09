use std::{collections::HashSet, fmt};

use crate::{
    ast::{
        Apply, Binding, ControlFlow, DeconstructInto, Expression, Identifier, Inject, Lambda,
        MatchClause, Product, Project, SelfReferential, Sequence, Variable,
    },
    typer::Parsed,
};

use super::{ModuleNames, ProductIndex, TypeAscription};

// bound variables on the form a.b.c are projections
// unbound variables are module selections, which are implemented as projections
// The bridge-code must not use prefix_woth, they should "mangle" the names instead
//   they can be bridge$xxx
// does it need the ModuleMap?
// Either way: they are all going to be Project
//   but it comes down to the base prefix.
//
// Given A.b.c, unless A is bound, pick A.b if in ModuleMap
//   but what happens it is not in the ModuleMap? UndefinedSymbol?
//
// Any identifier path has several prefixes
// What if about:
//   A.b.c.d and A.b is bound. Double projections?
//   Compute the shortest bound prefix, project remainder!
//
// I have to be able to check whether or not a symbol exists.
//
// But hmm, how could there be unbound identifiers?
//   function calls or plain variables defined "elsewhere"
//
// Forgot about tuple indices.
//   A.b.0
//   Can I get help from the parser here? It might be that any .0 suffixes
//     are already part of an enclosing Project created by the parser

impl<A> Expression<A>
where
    A: Clone + fmt::Debug + fmt::Display + Parsed,
{
    fn into_projection_tree(annotation: &A, name: &Identifier) -> Self {
        Self::make_projection_tree(
            annotation,
            Self::Variable(
                annotation.clone(),
                Variable::Identifier(name.head().clone()),
            ),
            &name.components()[1..],
        )
    }

    fn make_projection_tree(annotation: &A, head: Self, tail: &[&str]) -> Self {
        tail.iter().fold(head, |prefix, field| {
            Self::Project(
                annotation.clone(),
                Project {
                    base: prefix.into(),
                    index: ProductIndex::Struct(Identifier::new(field)),
                },
            )
        })
    }

    // This and the other rewrite-function, surely a common traverser can be extracted out of it
    pub fn rewrite_identifier_paths(self, bound: &mut HashSet<Identifier>) -> Self {
        match self {
            Self::Variable(annotation, Variable::Identifier(name)) => {
                if bound.contains(name.head()) {
                    Self::into_projection_tree(&annotation, &name)
                } else {
                    Self::Variable(annotation, Variable::Identifier(name))
                }
            }
            Self::Lambda(annotation, Lambda { parameter, body }) => {
                bound.insert(parameter.name.clone());
                Self::Lambda(
                    annotation,
                    Lambda {
                        parameter,
                        body: body.rewrite_identifier_paths(bound).into(),
                    },
                )
            }
            Self::SelfReferential(
                annotation,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                bound.insert(parameter.name.clone());
                bound.insert(name.clone());
                Self::SelfReferential(
                    annotation,
                    SelfReferential {
                        name,
                        parameter,
                        body: body.rewrite_identifier_paths(bound).into(),
                    },
                )
            }
            Self::Apply(annotation, Apply { function, argument }) => Self::Apply(
                annotation,
                Apply {
                    function: function.rewrite_identifier_paths(bound).into(),
                    argument: argument.rewrite_identifier_paths(bound).into(),
                },
            ),
            Self::Inject(
                annotation,
                Inject {
                    name,
                    constructor,
                    argument,
                },
            ) => Self::Inject(
                annotation,
                Inject {
                    name,
                    constructor,
                    argument: argument.rewrite_identifier_paths(bound).into(),
                },
            ),
            Self::Product(annotation, Product::Tuple(expressions)) => Self::Product(
                annotation,
                Product::Tuple(
                    expressions
                        .into_iter()
                        .map(|expr| expr.rewrite_identifier_paths(bound))
                        .collect(),
                ),
            ),
            Self::Product(annotation, Product::Struct(bindings)) => Self::Product(
                annotation,
                Product::Struct(
                    bindings
                        .into_iter()
                        .map(|(identifier, expr)| {
                            (identifier, expr.rewrite_identifier_paths(bound))
                        })
                        .collect(),
                ),
            ),
            Self::Project(annotation, Project { base, index }) => Self::Project(
                annotation,
                Project {
                    base: base.rewrite_identifier_paths(bound).into(),
                    index,
                },
            ),
            Self::Binding(
                annotation,
                Binding {
                    binder,
                    bound: bound_expr,
                    body,
                },
            ) => {
                bound.insert(binder.clone());
                Self::Binding(
                    annotation,
                    Binding {
                        binder,
                        bound: bound_expr.rewrite_identifier_paths(bound).into(),
                        body: body.rewrite_identifier_paths(bound).into(),
                    },
                )
            }
            Self::Sequence(annotation, Sequence { this, and_then }) => Self::Sequence(
                annotation,
                Sequence {
                    this: this.rewrite_identifier_paths(bound).into(),
                    and_then: and_then.rewrite_identifier_paths(bound).into(),
                },
            ),
            Self::ControlFlow(
                annotation,
                ControlFlow::If {
                    predicate,
                    consequent,
                    alternate,
                },
            ) => Self::ControlFlow(
                annotation,
                ControlFlow::If {
                    predicate: predicate.rewrite_identifier_paths(bound).into(),
                    consequent: consequent.rewrite_identifier_paths(bound).into(),
                    alternate: alternate.rewrite_identifier_paths(bound).into(),
                },
            ),
            Self::DeconstructInto(
                annotation,
                DeconstructInto {
                    scrutinee,
                    match_clauses,
                },
            ) => Self::DeconstructInto(
                annotation,
                DeconstructInto {
                    scrutinee: scrutinee.rewrite_identifier_paths(bound).into(),
                    match_clauses: match_clauses
                        .into_iter()
                        .map(|clause| {
                            bound.extend(clause.pattern.bindings().into_iter().cloned());
                            MatchClause {
                                pattern: clause.pattern,
                                consequent: clause
                                    .consequent
                                    .rewrite_identifier_paths(bound)
                                    .into(),
                            }
                        })
                        .collect(),
                },
            ),
            Self::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying,
                },
            ) => Self::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying: underlying.rewrite_identifier_paths(bound).into(),
                },
            ),
            otherwise => otherwise,
        }
    }

    pub fn rewrite_local_access(
        self,
        bound: &mut HashSet<Identifier>,
        module: &ModuleNames,
    ) -> Self {
        match self {
            Self::Variable(annotation, Variable::Identifier(name)) => {
                if !bound.contains(&name) && module.defines(&name) {
                    Self::Variable(
                        annotation,
                        Variable::Identifier(name.prefixed_with(module.name().clone())),
                    )
                } else {
                    Self::Variable(annotation, Variable::Identifier(name))
                }
            }
            Self::Lambda(annotation, Lambda { parameter, body }) => {
                bound.insert(parameter.name.clone());
                Self::Lambda(
                    annotation,
                    Lambda {
                        parameter,
                        body: body.rewrite_local_access(bound, module).into(),
                    },
                )
            }
            Self::Interpolation(annotation, interpolation) => Self::Interpolation(
                annotation,
                interpolation.rewrite_local_access(bound, module),
            ),
            Self::SelfReferential(
                annotation,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                bound.insert(parameter.name.clone());
                bound.insert(name.clone());
                Self::SelfReferential(
                    annotation,
                    SelfReferential {
                        name,
                        parameter,
                        body: body.rewrite_local_access(bound, module).into(),
                    },
                )
            }
            Self::Apply(annotation, Apply { function, argument }) => Self::Apply(
                annotation,
                Apply {
                    function: function.rewrite_local_access(bound, module).into(),
                    argument: argument.rewrite_local_access(bound, module).into(),
                },
            ),
            Self::Inject(
                annotation,
                Inject {
                    name,
                    constructor,
                    argument,
                },
            ) => Self::Inject(
                annotation,
                Inject {
                    name,
                    constructor,
                    argument: argument.rewrite_local_access(bound, module).into(),
                },
            ),
            Self::Product(annotation, Product::Tuple(expressions)) => Self::Product(
                annotation,
                Product::Tuple(
                    expressions
                        .into_iter()
                        .map(|expr| expr.rewrite_local_access(bound, module))
                        .collect(),
                ),
            ),
            Self::Product(annotation, Product::Struct(bindings)) => Self::Product(
                annotation,
                Product::Struct(
                    bindings
                        .into_iter()
                        .map(|(identifier, expr)| {
                            (identifier, expr.rewrite_local_access(bound, module))
                        })
                        .collect(),
                ),
            ),
            Self::Project(annotation, Project { base, index }) => Self::Project(
                annotation,
                Project {
                    base: base.rewrite_local_access(bound, module).into(),
                    index,
                },
            ),
            Self::Binding(
                annotation,
                Binding {
                    binder,
                    bound: bound_expr,
                    body,
                },
            ) => {
                bound.insert(binder.clone());
                Self::Binding(
                    annotation,
                    Binding {
                        binder,
                        bound: bound_expr.rewrite_local_access(bound, module).into(),
                        body: body.rewrite_local_access(bound, module).into(),
                    },
                )
            }
            Self::Sequence(annotation, Sequence { this, and_then }) => Self::Sequence(
                annotation,
                Sequence {
                    this: this.rewrite_local_access(bound, module).into(),
                    and_then: and_then.rewrite_local_access(bound, module).into(),
                },
            ),
            Self::ControlFlow(
                annotation,
                ControlFlow::If {
                    predicate,
                    consequent,
                    alternate,
                },
            ) => Self::ControlFlow(
                annotation,
                ControlFlow::If {
                    predicate: predicate.rewrite_local_access(bound, module).into(),
                    consequent: consequent.rewrite_local_access(bound, module).into(),
                    alternate: alternate.rewrite_local_access(bound, module).into(),
                },
            ),
            Self::DeconstructInto(
                annotation,
                DeconstructInto {
                    scrutinee,
                    match_clauses,
                },
            ) => Self::DeconstructInto(
                annotation,
                DeconstructInto {
                    scrutinee: scrutinee.rewrite_local_access(bound, module).into(),
                    match_clauses: match_clauses
                        .into_iter()
                        .map(|clause| {
                            bound.extend(clause.pattern.bindings().into_iter().cloned());
                            MatchClause {
                                pattern: clause.pattern,
                                consequent: clause
                                    .consequent
                                    .rewrite_local_access(bound, module)
                                    .into(),
                            }
                        })
                        .collect(),
                },
            ),
            Self::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying,
                },
            ) => Self::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying: underlying.rewrite_local_access(bound, module).into(),
                },
            ),
            otherwise => otherwise,
        }
    }

    pub fn resolve_names(self) -> Self {
        self.rewrite_identifier_paths(&mut HashSet::default())
    }

    pub fn resolve_module_local_names(self, module: &ModuleNames) -> Self {
        // What should this product? Identifier::Select because the next rewrite-step
        // sorts it?
        self.rewrite_local_access(&mut HashSet::default(), module)
    }
}
