# todo
[ ] better tracing than println! I want to see where the trace happens
[X] remove PhantomData - it is incorrectly used
[ ] improve error reporting, it should print where it goes bad
[ ] can the error report show the source code?
[ ] structs
  [X] parser
  [X] synthesize type
  [X] make_type_scheme - move from Coproduct?
  [ ] make_implementation_module for smart constructors and smart projectors
  [X] projections
    [ ] tuple projections seem off. Runtime representation is flat, type represetation is nested (right.)
[ ] modules
