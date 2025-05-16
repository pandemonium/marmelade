use std::{collections::HashSet, fmt};

use crate::ast::{
    Apply, Binding, ControlFlow, DeconstructInto, Expression, Identifier, Inject, Lambda,
    MatchClause, Product, Project, SelfReferential, Sequence,
};

use super::{ProductIndex, TypeAscription};

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
    A: Copy + fmt::Debug,
{
    fn into_projection_tree(annotation: &A, name: Identifier) -> Expression<A> {
        let tree = Self::make_projection_tree(
            annotation,
            Expression::Variable(*annotation, name.head().clone()),
            &name.components()[1..],
        );

        tree
    }

    fn make_projection_tree(annotation: &A, head: Expression<A>, tail: &[&str]) -> Expression<A> {
        tail.iter().fold(head, |prefix, field| {
            Expression::Project(
                *annotation,
                Project {
                    base: prefix.into(),
                    index: ProductIndex::Struct(Identifier::new(field)),
                },
            )
        })
    }

    fn into_module_path(
        annotation: &A,
        name: Identifier,
        module_map: &HashSet<Identifier>,
    ) -> Expression<A> {
        let path = name.components();
        let module_base_expr = (1..path.len())
            // This is not super efficient -- fix at some point
            .find_map(|prefix_length| {
                Identifier::try_from_components(&path[0..prefix_length]).and_then(|prefix| {
                    module_map
                        .contains(&prefix)
                        .then_some((prefix, &path[prefix_length..]))
                })
            });

        if let Some((base, index)) = module_base_expr {
            Self::make_projection_tree(annotation, Expression::Variable(*annotation, base), index)
        } else {
            Expression::Variable(*annotation, name)
        }
    }

    // Needs to take into account _where_ the identifier is.
    fn rewrite_identifiers(
        self,
        bound: &mut HashSet<Identifier>,
        module_map: &HashSet<Identifier>,
    ) -> Expression<A> {
        match self {
            Expression::Variable(annotation, name) => {
                if bound.contains(name.head()) {
                    Self::into_projection_tree(&annotation, name)
                } else {
                    Self::into_module_path(&annotation, name, module_map)
                }
            }
            Expression::Lambda(annotation, Lambda { parameter, body }) => {
                bound.insert(parameter.name.clone());
                let body = body.rewrite_identifiers(bound, module_map);
                Expression::Lambda(
                    annotation,
                    Lambda {
                        parameter,
                        body: body.into(),
                    },
                )
            }
            Expression::SelfReferential(
                annotation,
                SelfReferential {
                    name,
                    parameter,
                    body,
                },
            ) => {
                bound.insert(parameter.name.clone());
                bound.insert(name.clone());
                let body = body.rewrite_identifiers(bound, module_map);
                Expression::SelfReferential(
                    annotation,
                    SelfReferential {
                        name,
                        parameter,
                        body: body.into(),
                    },
                )
            }
            Expression::Apply(annotation, Apply { function, argument }) => {
                let function = function.rewrite_identifiers(bound, module_map);
                let argument = argument.rewrite_identifiers(bound, module_map);
                Expression::Apply(
                    annotation,
                    Apply {
                        function: function.into(),
                        argument: argument.into(),
                    },
                )
            }
            Expression::Inject(
                annotation,
                Inject {
                    name,
                    constructor,
                    argument,
                },
            ) => {
                let argument = argument.rewrite_identifiers(bound, module_map);
                Expression::Inject(
                    annotation,
                    Inject {
                        name,
                        constructor,
                        argument: argument.into(),
                    },
                )
            }
            Expression::Product(annotation, Product::Tuple(expressions)) => Expression::Product(
                annotation,
                Product::Tuple(
                    expressions
                        .into_iter()
                        .map(|expr| expr.rewrite_identifiers(bound, module_map))
                        .collect(),
                ),
            ),
            Expression::Product(annotation, Product::Struct(bindings)) => Expression::Product(
                annotation,
                Product::Struct(
                    bindings
                        .into_iter()
                        .map(|(identifier, expr)| {
                            (identifier, expr.rewrite_identifiers(bound, module_map))
                        })
                        .collect(),
                ),
            ),
            Expression::Project(annotation, Project { base, index }) => {
                let base = base.rewrite_identifiers(bound, module_map);
                Expression::Project(
                    annotation,
                    Project {
                        base: base.into(),
                        index,
                    },
                )
            }
            Expression::Binding(
                annotation,
                Binding {
                    binder,
                    bound: bound_expr,
                    body,
                },
            ) => {
                let bound_expr = bound_expr.rewrite_identifiers(bound, module_map);
                bound.insert(binder.clone());
                let body = body.rewrite_identifiers(bound, module_map);

                Expression::Binding(
                    annotation,
                    Binding {
                        binder,
                        bound: bound_expr.into(),
                        body: body.into(),
                    },
                )
            }
            Expression::Sequence(annotation, Sequence { this, and_then }) => {
                let this = this.rewrite_identifiers(bound, module_map);
                let and_then = and_then.rewrite_identifiers(bound, module_map);
                Expression::Sequence(
                    annotation,
                    Sequence {
                        this: this.into(),
                        and_then: and_then.into(),
                    },
                )
            }
            Expression::ControlFlow(
                annotation,
                ControlFlow::If {
                    predicate,
                    consequent,
                    alternate,
                },
            ) => {
                let predicate = predicate.rewrite_identifiers(bound, module_map);
                let consequent = consequent.rewrite_identifiers(bound, module_map);
                let alternate = alternate.rewrite_identifiers(bound, module_map);

                Expression::ControlFlow(
                    annotation,
                    ControlFlow::If {
                        predicate: predicate.into(),
                        consequent: consequent.into(),
                        alternate: alternate.into(),
                    },
                )
            }
            Expression::DeconstructInto(
                annotation,
                DeconstructInto {
                    scrutinee,
                    match_clauses,
                },
            ) => {
                let scrutinee = scrutinee.rewrite_identifiers(bound, module_map);
                Expression::DeconstructInto(
                    annotation,
                    DeconstructInto {
                        scrutinee: scrutinee.into(),
                        match_clauses: match_clauses
                            .into_iter()
                            .map(|clause| {
                                bound.extend(
                                    clause
                                        .pattern
                                        .bindings()
                                        .into_iter()
                                        .cloned()
                                        .collect::<Vec<_>>(),
                                );
                                let consequent =
                                    clause.consequent.rewrite_identifiers(bound, module_map);

                                MatchClause {
                                    pattern: clause.pattern,
                                    consequent: consequent.into(),
                                }
                            })
                            .collect(),
                    },
                )
            }
            Expression::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying,
                },
            ) => Expression::TypeAscription(
                annotation,
                TypeAscription {
                    type_signature,
                    underlying: underlying.rewrite_identifiers(bound, module_map).into(),
                },
            ),
            otherwise => otherwise,
        }
    }

    pub fn resolve_names<'a>(self, module_map: &HashSet<Identifier>) -> Self {
        self.rewrite_identifiers(&mut HashSet::default(), module_map)
    }
}
