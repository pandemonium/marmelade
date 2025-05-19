# todo
[ ] type_check seems to run twice and the second time without type signatures
[ ] better tracing than println! I want to see where the trace happens
[ ] improve error reporting, it should print where it goes bad
[ ] can the error report show the source code?
[ ] string interpolation
[ ] modules
  [ ] associated modules
    [X] for structs
    [X] for coproducts
    [ ] for type aliases
  [ ] plain old

# Parser
[ ] Struct make_implementation_module for smart constructors and smart projectors

# Namer
[X] Delete the module struct stuff
[X] Prepend declarations with the module path of its declaring module
[X] Prepend free variables with name of module declaring the declaration
