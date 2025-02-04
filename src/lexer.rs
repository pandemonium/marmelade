#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Location {
    pub row: u32,
    pub column: u32,
}

impl Location {
    pub fn new(row: u32, column: u32) -> Self {
        Self { row, column }
    }

    fn move_right(&mut self, delta: u32) {
        self.column += delta;
    }

    fn new_line(&mut self) {
        self.row += 1;
        self.column = 1;
    }

    pub fn is_left_of(&self, indentation_level: u32) -> bool {
        self.column < indentation_level
    }

    pub fn is_right_of(&self, indentation_level: u32) -> bool {
        self.column > indentation_level
    }

    pub fn is_below(&self, rhs: &Self) -> bool {
        self.row > rhs.row
    }

    pub fn is_same_block(&self, rhs: &Self) -> bool {
        self.column == rhs.column && self.is_below(rhs)
    }
}

impl Default for Location {
    fn default() -> Self {
        Self { row: 1, column: 1 }
    }
}

// Flatten this?
// What does this hierarchy buy me?
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Equals,      // =
    TypeAssign,  // ::=
    TypeAscribe, // ::
    Arrow,       // ->
    Comma,       // ,
    LeftParen,   // (
    RightParen,  // )
    Underscore,  // _
    Pipe,        // |
    DoubleQuote, // "
    SingleQuote, // '
    Semicolon,   // ;

    Identifier(String),

    Keyword(Keyword),
    Literal(Literal),
    Operator(Operator),

    Layout(Layout),
    End,
}

impl TokenType {
    fn decode_identifier(id: String) -> Self {
        match Keyword::try_from_identifier(&id) {
            Some(keyword) => Self::Keyword(keyword),
            None => id
                .parse::<bool>()
                .map(|x| Self::Literal(Literal::Bool(x)))
                .unwrap_or_else(|_| Self::Identifier(id)),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Divides,
    Modulo,
    Juxtaposition,
}

impl Operator {
    pub fn precedence(&self) -> usize {
        match self {
            Self::Plus | Self::Minus => 1,
            Self::Times | Self::Divides | Self::Modulo => 2,
            Self::Juxtaposition => 69,
        }
    }

    pub fn function_identifier(&self) -> String {
        // These mappings are highly dubious
        match self {
            Self::Plus => "+".to_owned(),
            Self::Minus => "-".to_owned(),
            Self::Times => "*".to_owned(),
            Self::Divides => "/".to_owned(),
            Self::Modulo => "%".to_owned(),
            Self::Juxtaposition => "$".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    Indent,
    Dedent,
    Newline,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Keyword {
    Let,
    In,
    If,
    Struct,
    Coproduct,
    Alias,
    Module,
    Use,
    Fun,
}

impl Keyword {
    fn try_from_identifier(id: &str) -> Option<Keyword> {
        match id {
            "let" => Some(Keyword::Let),
            "in" => Some(Keyword::In),
            "if" => Some(Keyword::If),
            "struct" => Some(Keyword::Struct),
            "coproduct" => Some(Keyword::Coproduct),
            "alias" => Some(Keyword::Alias),
            "module" => Some(Keyword::Module),
            "use" => Some(Keyword::Use),
            "fun" => Some(Keyword::Fun),
            _otherwise => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Integer(i64),
    Text(String),
    Bool(bool),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Token(pub TokenType, pub Location);

impl Token {
    pub fn token_type(&self) -> &TokenType {
        &self.0
    }

    pub fn location(&self) -> &Location {
        &self.1
    }
}

#[derive(Debug)]
pub struct LexicalAnalyzer {
    location: Location,
    indentation_level: u32,
    output: Vec<Token>,
}

impl LexicalAnalyzer {
    pub fn untokenize(&self, _input: &[Token]) -> String {
        todo!()
    }

    pub fn tokenize(&mut self, input: &[char]) -> &[Token] {
        let mut input = input;
        loop {
            input = match input {
                remains @ [c, ..] if c.is_whitespace() => self.scan_whitespace(remains),
                prefix @ [c, ..] if is_identifier_prefix(*c) => self.scan_identifier(prefix),
                prefix @ [c, ..] if is_number_prefix(*c) => self.scan_number(prefix),
                ['"', remains @ ..] => self.scan_text_literal(remains),

                [':', ':', '=', remains @ ..] => self.emit(3, TokenType::TypeAssign, remains),
                [':', ':', remains @ ..] => self.emit(2, TokenType::TypeAscribe, remains),
                ['-', '>', remains @ ..] => self.emit(2, TokenType::Arrow, remains),

                ['=', remains @ ..] => self.emit(1, TokenType::Equals, remains),
                [',', remains @ ..] => self.emit(1, TokenType::Comma, remains),
                ['(', remains @ ..] => self.emit(1, TokenType::LeftParen, remains),
                [')', remains @ ..] => self.emit(1, TokenType::RightParen, remains),
                ['_', remains @ ..] => self.emit(1, TokenType::Underscore, remains),
                ['|', remains @ ..] => self.emit(1, TokenType::Pipe, remains),
                [';', remains @ ..] => self.emit(1, TokenType::Semicolon, remains),

                ['+', remains @ ..] => self.emit_operator(1, Operator::Plus, remains),
                ['-', remains @ ..] => self.emit_operator(1, Operator::Minus, remains),
                ['*', remains @ ..] => self.emit_operator(1, Operator::Times, remains),
                ['/', remains @ ..] => self.emit_operator(1, Operator::Divides, remains),
                ['%', remains @ ..] => self.emit_operator(1, Operator::Modulo, remains),

                [c, ..] => panic!("{c}"),

                [] => {
                    self.emit(0, TokenType::End, &[]);
                    break &self.output;
                }
            };
        }
    }

    pub fn into_token_type_stream(&self) -> Vec<TokenType> {
        self.output.iter().map(|t| t.token_type().clone()).collect()
    }

    fn scan_identifier<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = input[1..]
            .iter()
            .position(|c| !is_identifier_continuation(*c))
        {
            (&input[..(end + 1)], &input[(end + 1)..])
        } else {
            (input, &input[..0])
        };

        let identifier = prefix.into_iter().collect::<String>();
        self.emit(
            identifier.len() as u32,
            TokenType::decode_identifier(identifier),
            remains,
        )
    }

    fn scan_text_literal<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = input.iter().position(|&c| c == '"') {
            (&input[..end], &input[end + 1..])
        } else {
            (input, &input[..0])
        };

        let image = prefix.into_iter().collect::<String>();
        self.emit(
            image.len() as u32,
            TokenType::Literal(Literal::Text(image)),
            remains,
        )
    }

    fn emit_operator<'a>(&mut self, length: u32, op: Operator, remains: &'a [char]) -> &'a [char] {
        self.emit(length, TokenType::Operator(op), remains)
    }

    fn emit<'a>(&mut self, length: u32, token_type: TokenType, remains: &'a [char]) -> &'a [char] {
        self.output.push(Token(token_type, self.location.clone()));
        self.location.move_right(length);
        remains
    }

    fn scan_whitespace<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (whitespace_prefix, remains) =
            if let Some(end) = input.iter().position(|c| !c.is_whitespace()) {
                (&input[..end], &input[end..])
            } else {
                (input, &input[..0])
            };

        let next_location = self.compute_new_location(whitespace_prefix);

        self.update_location(next_location);

        remains
    }

    fn compute_new_location(&mut self, whitespace: &[char]) -> Location {
        let mut next_location = self.location.clone();

        for c in whitespace {
            match c {
                ' ' => next_location.move_right(1),
                '\n' => next_location.new_line(),
                otherwise => {
                    println!("process_whitespace: {}", *otherwise as u32);
                    ()
                }
            }
        }

        next_location
    }

    fn update_location(&mut self, next: Location) {
        if next.is_below(&self.location) {
            if next.is_left_of(self.indentation_level) {
                self.emit_layout(next, Layout::Dedent);
            } else if next.is_right_of(self.indentation_level) {
                self.emit_layout(next, Layout::Indent);
            } else {
                self.emit_layout(next, Layout::Newline);
            }
            self.indentation_level = next.column;
        }

        self.location = next;
    }

    // Which location is the location of an Indent or Dedent?
    fn emit_layout(&mut self, _location: Location, indentation: Layout) {
        self.output
            .push(Token(TokenType::Layout(indentation), self.location));
    }

    fn scan_number<'a>(&mut self, prefix: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = prefix.iter().position(|c| !c.is_ascii_digit()) {
            (&prefix[..end], &prefix[end..])
        } else {
            (prefix, &prefix[..0])
        };

        // This has to be able to fail the tokenization here
        let num = prefix.iter().collect::<String>().parse().unwrap();

        self.emit(
            prefix.len() as u32,
            TokenType::Literal(Literal::Integer(num)),
            remains,
        );

        remains
    }
}

fn is_number_prefix(c: char) -> bool {
    c.is_digit(10) // Improved
}

fn is_identifier_prefix(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_identifier_continuation(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c.is_numeric()
}

impl Default for LexicalAnalyzer {
    fn default() -> Self {
        let location = Location::default();
        Self {
            location,
            indentation_level: location.column,
            output: Vec::default(), // This could actually be something a lot bigger.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TokenType as TT;

    fn into_input(source: &str) -> Vec<char> {
        source
            .lines()
            .filter(|s| !s.trim().is_empty())
            .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .collect()
    }

    #[test]
    fn option_decl() {
        let mut lexer = LexicalAnalyzer::default();

        let input = into_input(
            "|Option a ::=
             |    The a | Nil
             |",
        );

        lexer.tokenize(&input);

        // Look at prefix
        assert_eq!(
            &lexer.into_token_type_stream(),
            &[
                TT::Identifier("Option".to_owned()),
                TT::Identifier("a".to_owned()),
                TT::TypeAssign,
                TT::Layout(Layout::Indent),
                TT::Identifier("The".to_owned()),
                TT::Identifier("a".to_owned()),
                TT::Pipe,
                TT::Identifier("Nil".to_owned()),
                TT::Layout(Layout::Dedent),
                TT::End
            ]
        );
    }

    #[test]
    fn let_in() {
        let mut lexer = LexicalAnalyzer::default();

        let input = into_input("|let x = 10 in x + 1");

        lexer.tokenize(&input);

        // Look at prefix
        assert_eq!(
            &lexer.into_token_type_stream(),
            &[
                TT::Keyword(Keyword::Let),
                TT::Identifier("x".to_owned()),
                TT::Equals,
                TT::Literal(Literal::Integer(10)),
                TT::Keyword(Keyword::In),
                TT::Identifier("x".to_owned()),
                TT::Operator(Operator::Plus),
                TT::Literal(Literal::Integer(1)),
                TT::End
            ]
        );
    }

    #[test]
    fn let_in_more() {
        let mut lexer = LexicalAnalyzer::default();

        let input = into_input(
            "|let x = f 1 2
             |in 3 * f 4 + 5",
        );

        lexer.tokenize(&input);

        // Look at prefix
        assert_eq!(
            &lexer.into_token_type_stream(),
            &[
                TT::Keyword(Keyword::Let),
                TT::Identifier("x".to_owned()),
                TT::Equals,
                TT::Identifier("f".to_owned()),
                TT::Literal(Literal::Integer(1)),
                TT::Literal(Literal::Integer(2)),
                TT::Layout(Layout::Newline),
                TT::Keyword(Keyword::In),
                TT::Literal(Literal::Integer(3)),
                TT::Operator(Operator::Times),
                TT::Identifier("f".to_owned()),
                TT::Literal(Literal::Integer(4)),
                TT::Operator(Operator::Plus),
                TT::Literal(Literal::Integer(5)),
                TT::End,
            ]
        );
    }

    #[test]
    fn literals() {
        let mut lexer = LexicalAnalyzer::default();

        let input = into_input(r#"12345 "Hi, mom" 5"#);

        lexer.tokenize(&input);

        // Look at prefix
        assert_eq!(
            &lexer.into_token_type_stream(),
            &[
                TT::Literal(Literal::Integer(12345)),
                TT::Literal(Literal::Text("Hi, mom".to_owned())),
                TT::Literal(Literal::Integer(5)),
                TT::End
            ]
        );
    }
}
