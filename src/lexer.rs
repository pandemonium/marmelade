use std::fmt;

use crate::ast::Identifier;

// // For this type to be worthwhile, it would have to still
// // retain the whole slice, otherwise it cannot be used in
// // an error reporting context anyway
// #[derive(Debug, Copy, Clone)]
// pub struct SourceText<'a> {
//     offset: usize,
//     text: &'a [char],
// }
//
// impl<'a> SourceText<'a> {
//     fn split_at(self, position: usize) -> (Self, Self) {
//         let (lhs, rhs) = self.0.split_at(position);
//         (Self(lhs), Self(rhs))
//     }
//
//     fn empty(self) -> Self {
//         Self(&self.0[..0])
//     }
// }
//
// impl<'a> IntoIterator for SourceText<'a> {
//     type Item = &'a char;
//     type IntoIter = Iter<'a, char>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         self.0.iter()
//     }
// }
//
// impl<'a> Deref for SourceText<'a> {
//     type Target = [char];
//
//     fn deref(&self) -> &Self::Target {
//         self.0
//     }
// }

#[derive(Debug)]
pub struct LexicalAnalyzer {
    location: SourcePosition,
    indentation_level: u32,
    output: Vec<Token>,
}

impl LexicalAnalyzer {
    pub fn tokens(&self) -> &[Token] {
        &self.output
    }

    pub fn untokenize(&self) -> String {
        let mut buf = String::with_capacity(self.output.len() * 2);
        let mut indentation = 1;
        let mut spaces = " ".repeat(indentation);
        let mut row = 0;
        for t in &self.output {
            if let TokenType::Layout(layout) = t.token_type() {
                match layout {
                    Layout::Indent => {
                        indentation = t.location().column as usize;
                        spaces = " ".repeat(indentation)
                    }
                    Layout::Dedent => {
                        indentation = t.location().column as usize;
                        spaces = " ".repeat(indentation)
                    }
                    _otherwise => (),
                }
            } else {
                if row != t.location().row {
                    row = t.location().row;
                    buf.push_str(&format!("\n{row:>3}{spaces}"));
                }
                buf.push_str(&format!("{} ", t.token_type()));
            }
        }
        buf
    }

    pub fn tokenize(&mut self, input: &[char]) -> &[Token] {
        let mut input = input;
        loop {
            input = match input {
                ['a', 'n', 'd', remains @ ..] => self.emit_operator(3, Operator::And, remains),
                ['o', 'r', remains @ ..] => self.emit_operator(2, Operator::Or, remains),
                ['x', 'o', 'r', remains @ ..] => self.emit_operator(3, Operator::Xor, remains),
                ['n', 'o', 't', remains @ ..] => self.emit_operator(3, Operator::Not, remains),

                remains @ [c, ..] if c.is_whitespace() => self.scan_whitespace(remains),
                prefix @ [c, ..] if is_identifier_prefix(*c) => self.scan_identifier(prefix),
                prefix @ [c, ..] if is_number_prefix(*c) => self.scan_number(prefix),
                ['"', remains @ ..] => self.scan_text_literal(remains),

                [':', ':', '=', remains @ ..] => self.emit(3, TokenType::TypeAssign, remains),
                [':', ':', remains @ ..] => self.emit(2, TokenType::TypeAscribe, remains),
                ['-', '>', remains @ ..] => self.emit(2, TokenType::Arrow, remains),

                ['=', '=', remains @ ..] => self.emit_operator(2, Operator::Equals, remains),
                ['>', '=', remains @ ..] => self.emit_operator(2, Operator::Gte, remains),
                ['<', '=', remains @ ..] => self.emit_operator(2, Operator::Lte, remains),
                ['>', remains @ ..] => self.emit_operator(1, Operator::Gt, remains),
                ['<', remains @ ..] => self.emit_operator(1, Operator::Lt, remains),

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

    fn compute_new_location(&mut self, whitespace: &[char]) -> SourcePosition {
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

    fn update_location(&mut self, next: SourcePosition) {
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
    fn emit_layout(&mut self, location: SourcePosition, indentation: Layout) {
        self.output
            .push(Token(TokenType::Layout(indentation), location));
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
        let location = SourcePosition::default();
        Self {
            location,
            indentation_level: location.column,
            output: Vec::default(), // This could actually be something a lot bigger.
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SourcePosition {
    pub row: u32,
    pub column: u32,
}

impl SourcePosition {
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
        (self.column == rhs.column && self.is_below(rhs)) || self.row == rhs.row
    }
}

impl Default for SourcePosition {
    fn default() -> Self {
        Self { row: 1, column: 1 }
    }
}

impl fmt::Display for SourcePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { row, column } = self;
        write!(f, "({},{})", row, column)
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

    Equals,
    Gte,
    Lte,
    Gt,
    Lt,

    And,
    Or,
    Xor,

    Not,
}

impl Operator {
    pub fn precedence(&self) -> usize {
        match self {
            Self::Juxtaposition => 69,

            Self::Times | Self::Divides | Self::Modulo => 6,
            Self::Plus | Self::Minus => 5,

            Self::Equals | Self::Gte | Self::Lte | Self::Gt | Self::Lt => 4,

            Self::Not => 3,

            Self::And => 2,
            Self::Xor | Self::Or => 1,
        }
    }

    pub fn id(&self) -> Identifier {
        Identifier::new(&self.function_identifier())
    }

    pub fn function_identifier(&self) -> &str {
        // These mappings are highly dubious
        match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Times => "*",
            Self::Divides => "/",
            Self::Modulo => "%",

            Self::Equals => "==",
            Self::Gte => ">=",
            Self::Lte => "<=",
            Self::Gt => ">",
            Self::Lt => "<",

            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Not => "not",

            Self::Juxtaposition => "$",
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Times => write!(f, "*"),
            Self::Divides => write!(f, "/"),
            Self::Modulo => write!(f, "%"),

            Self::Equals => write!(f, "=="),

            Self::Gte => write!(f, ">="),
            Self::Lte => write!(f, ">="),
            Self::Gt => write!(f, ">"),
            Self::Lt => write!(f, "<"),

            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Not => write!(f, "not"),

            Self::Juxtaposition => write!(f, "$"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    Indent,
    Dedent,
    Newline,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Indent => write!(f, "<Indent>"),
            Self::Dedent => write!(f, "<Dedent>"),
            Self::Newline => write!(f, "<Newline>"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Keyword {
    Let,
    In,
    If,
    Then,
    Else,
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
            "then" => Some(Keyword::Then),
            "else" => Some(Keyword::Else),
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

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Let => write!(f, "Let"),
            Self::In => write!(f, "In"),
            Self::If => write!(f, "If"),
            Self::Then => write!(f, "Then"),
            Self::Else => write!(f, "Else"),
            Self::Struct => write!(f, "Struct"),
            Self::Coproduct => write!(f, "Coproduct"),
            Self::Alias => write!(f, "Alias"),
            Self::Module => write!(f, "Module"),
            Self::Use => write!(f, "Use"),
            Self::Fun => write!(f, "Fun"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Integer(i64),
    Text(String),
    Bool(bool),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(x) => write!(f, "{x}"),
            Self::Text(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{x}"),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Token(pub TokenType, pub SourcePosition);

impl Token {
    pub fn token_type(&self) -> &TokenType {
        &self.0
    }

    pub fn location(&self) -> &SourcePosition {
        &self.1
    }

    pub fn is_layout(&self) -> bool {
        matches!(self, Token(TokenType::Layout(..), ..))
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equals => write!(f, "="),
            Self::TypeAssign => write!(f, "::="),
            Self::TypeAscribe => write!(f, "::"),
            Self::Arrow => write!(f, "->"),
            Self::Comma => write!(f, ","),
            Self::LeftParen => write!(f, "("),
            Self::RightParen => write!(f, ")"),
            Self::Underscore => write!(f, "_"),
            Self::Pipe => write!(f, "|"),
            Self::DoubleQuote => write!(f, "\""),
            Self::SingleQuote => write!(f, "'"),
            Self::Semicolon => write!(f, ";"),
            Self::Identifier(id) => write!(f, "{id}"),
            Self::Keyword(keyword) => write!(f, "{keyword}"),
            Self::Literal(literal) => write!(f, "{literal}"),
            Self::Operator(operator) => write!(f, "{operator}"),
            Self::Layout(layout) => write!(f, "{layout}"),
            Self::End => write!(f, "°"),
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
