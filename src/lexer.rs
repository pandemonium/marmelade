pub struct Location {
    pub row: u16,
    pub column: u16,
}

pub enum TokenType {
    Separator(Separator),
    Identifier,
    Keyword(Keyword),
    Literal(Literal),
}

pub enum Separator {
    Equals,
    TypeAlias,
    TypeAnnotation,
    FieldSeparator,
    LeftParen,
    RightParen,
    Scope,
    Underscore,
}

pub enum Keyword {
    Let,
    In,
    If,
}

pub enum Literal {
    Numeric,
    Textual,
}

pub struct Token {
    location: Location,
    token_type: TokenType,
}

pub fn tokenize(input: &[char], output: &mut Vec<Token>) {}
