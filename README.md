# Toy functional language
# Syntax playground:

# Co-products
    Perhaps ::= forall a. This a | Nil
    Perhaps ::= This a | Nil
    Perhaps ::=
      This a
      Nil
    Perhaps ::= forall a.
      This a
      Nil

    ## Implies that the Eval type is Eval a e, and not e a.
    Eval ::= forall a e. Fault e | Return a

    ## Declares Eval e a
    Eval ::= Fault e | Return a

    Type_Error ::=
        Undefined_Symbol       Text
        Unification_Impossible Type Type

    Type ::=
        Parameter typer::TypeVariable
        Trivial   TrivialType
        Function  Type Type


    List ::= forall a. Cons a (List a) | Nil
    List ::= Cons a (List a) | Nil
    List ::=
      Cons a (List a)
      Nil

    Typing ::= alias Result Type Type_Error
    Loaded ::= forall a. alias Result a Load_error

    ## Records
    Gui_Window ::=
        { width  : u32
          height : u32 }

    Gui_Window ::= { width : u32; height : u32 }

    let <symbol> <=> [<Indent>] <expr> ( ( <;> or <newline> <expr>)* ) [<Dedent>]
    in [<Indent>] <expr> ( ( <;> or <newline> <expr>)* )

With the caveat that: If there is an Indent following <=>, then the Dedent is optionally present. If there isn't one, then there must be no Dedent before in

    let x =
        call_some_stuff 1 2 3
    in
        render_the_thing x y z

    Option module
        T a coproduct The of a | Nil

        make x = The x

        map f self =
            deconstruct self into
            The x -> f x
            Nil   -> Nil

        bind f =
            function The x -> f x | Nil -> Nil


    width =
        let window =
            { width  427
        height 100 }
            in window.width

    height =
        let window = 427, 100
        in window.1

    print_it =
        std::io::println "Hello, world #1"

    print_hello :: (_ : unit) -> unit =
        use std::io::println
        println "Hello, world #2"

    print_hello ()

    identity :: a -> a =
        fun x -> x

    (* is this enough? *)
    identity =
        fun x -> x

    (* Multiple parameters? *)
    mk_pair = lambda x y. x, y

    (* Tuple argument *)
    fst = lambda (x, y) -> x

    (* Tuple argument *)
    snd = lambda (x, )y -> y
