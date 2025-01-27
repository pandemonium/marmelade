#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub row: u32,
    pub column: u32,
}

impl Location {
    fn move_right(&mut self, delta: u32) {
        self.column += delta;
    }

    fn new_line(&mut self) {
        self.row += 1;
        self.column = 1;
    }
}

impl Default for Location {
    fn default() -> Self {
        Self { row: 1, column: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Separator(Separator),
    Identifier(String),
    Keyword(Keyword),
    Literal(Literal),
    End,
}

impl TokenType {
    fn identifier_or_keyword(id: String) -> Self {
        Keyword::try_from_identifier(&id)
            .map(Self::Keyword)
            .unwrap_or_else(|| Self::Identifier(id))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Separator {
    Equals,      // =
    TypeAssign,  // ::=
    DoubleColon, // ::
    Comma,       // ,
    LeftParen,   // (
    RightParen,  // )
    Underscore,  // _
    Pipe,        // |
    DoubleQuote, // "
    SingleQuote, // '
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Keyword {
    Let,
    In,
    If,
    Struct,
    Alias,
    Module,
    Use,
}

impl Keyword {
    fn try_from_identifier(id: &str) -> Option<Keyword> {
        match id {
            "let" => Some(Keyword::Let),
            "in" => Some(Keyword::In),
            "if" => Some(Keyword::If),
            "struct" => Some(Keyword::Struct),
            "alias" => Some(Keyword::Alias),
            "module" => Some(Keyword::Module),
            "use" => Some(Keyword::Use),
            _otherwise => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Number,
    Text(String),
}

#[derive(Debug, Eq, PartialEq)]
pub struct Token {
    location: Location,
    token_type: TokenType,
}

#[derive(Debug, Default)]
pub struct LexicalAnalyzer {
    location: Location,
    output: Vec<Token>,
}

impl LexicalAnalyzer {
    pub fn parse(&mut self, input: &[char]) -> &[Token] {
        let mut input = input;
        loop {
            input = match input {
                [c, remains @ ..] if c.is_whitespace() => self.process_whitespace(*c, remains),
                ['=', remains @ ..] => self.emit_separator(1, Separator::Equals, remains),
                [':', ':', '=', remains @ ..] => {
                    self.emit_separator(3, Separator::TypeAssign, remains)
                }
                [':', ':', remains @ ..] => self.emit_separator(2, Separator::DoubleColon, remains),
                [',', remains @ ..] => self.emit_separator(1, Separator::Comma, remains),
                ['(', remains @ ..] => self.emit_separator(1, Separator::LeftParen, remains),
                [')', remains @ ..] => self.emit_separator(1, Separator::RightParen, remains),
                ['_', remains @ ..] => self.emit_separator(1, Separator::Underscore, remains),
                ['|', remains @ ..] => self.emit_separator(1, Separator::Pipe, remains),

                prefix @ [c, ..] if is_identifier_start(*c) => self.process_identifier(prefix),

                ['"', remains @ ..] => self.process_text_literal(remains),

                [c, ..] => panic!("{c}"),

                [] => {
                    self.emit(0, TokenType::End, &[]);
                    break &self.output;
                }
            };
        }
    }

    fn process_identifier<'a>(&mut self, remains: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = remains[1..]
            .iter()
            .position(|c| !is_identifier_continuation(*c))
        {
            let p = &remains[..(end + 1)];
            let q = &remains[(end + 1)..];

            println!(
                "process_identifier: '{}' '{}'",
                p.iter().collect::<String>(),
                q.iter().collect::<String>(),
            );

            (&remains[..(end + 1)], &remains[(end + 1)..])
        } else {
            (remains, &remains[..0])
        };

        let image = prefix.into_iter().collect::<String>();
        self.emit(
            image.len() as u32,
            TokenType::identifier_or_keyword(image),
            remains,
        )
    }

    fn process_text_literal<'a>(&mut self, remains: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = remains.iter().position(|&c| c == 'c') {
            (&remains[..end], &remains[end..])
        } else {
            (remains, &remains[..0])
        };

        let image = prefix.into_iter().collect::<String>();
        self.emit(
            image.len() as u32,
            TokenType::Literal(Literal::Text(image)),
            remains,
        )
    }

    fn emit_separator<'a>(
        &mut self,
        length: u32,
        sep: Separator,
        remains: &'a [char],
    ) -> &'a [char] {
        self.emit(length, TokenType::Separator(sep), remains)
    }

    fn emit<'a>(&mut self, length: u32, token_type: TokenType, remains: &'a [char]) -> &'a [char] {
        self.output.push(Token {
            location: self.location.clone(),
            token_type,
        });
        self.location.move_right(length);
        remains
    }

    fn process_whitespace<'a>(&mut self, c: char, remains: &'a [char]) -> &'a [char] {
        match c {
            ' ' => self.location.move_right(1),
            '\n' => self.location.new_line(),
            otherwise => {
                println!("process_whitespace: {}", otherwise as u32);
                ()
            }
        }
        remains
    }
}

fn is_identifier_start(c: char) -> bool {
    let x = c.is_alphabetic() || c == '_';
    println!("is_identifier_start: {c} is {x}");
    x
}

fn is_identifier_continuation(c: char) -> bool {
    let x = c.is_alphabetic() || c == '_' || c.is_numeric();
    println!("is_identifier_continuation: {c} is {x}");
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use TokenType as TT;

    #[test]
    fn option_decl() {
        let source = "|Option a ::=
                    |    The a | Nil
                    |"
        .lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| line.trim_start().strip_prefix("|").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .collect::<Vec<_>>();

        let mut lexer = LexicalAnalyzer::default();
        lexer.parse(&source);

        assert_eq!(
            &lexer
                .output
                .into_iter()
                .map(|t| t.token_type)
                .collect::<Vec<_>>(),
            &[
                TT::Identifier("Option".to_owned()),
                TT::Identifier("a".to_owned()),
                TT::Separator(Separator::TypeAssign),
                TT::Identifier("The".to_owned()),
                TT::Identifier("a".to_owned()),
                TT::Separator(Separator::Pipe),
                TT::Identifier("Nil".to_owned()),
                TT::End
            ]
        );
    }
}
