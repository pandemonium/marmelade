#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

    fn is_left_of(&self, indentation_level: u32) -> bool {
        self.column < indentation_level
    }

    fn is_right_of(&self, indentation_level: u32) -> bool {
        self.column > indentation_level
    }

    fn is_below(&self, rhs: &Self) -> bool {
        self.row > rhs.row
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
    Indentation(Indentation),
    End,
}

impl TokenType {
    fn identifier_or_keyword(id: String) -> Self {
        Keyword::try_from_identifier(&id)
            .map(Self::Keyword)
            .unwrap_or_else(|| Self::Identifier(id))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Indentation {
    Increase,
    Decrease,
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
    indentation_level: u32,
    output: Vec<Token>,
}

impl LexicalAnalyzer {
    pub fn parse(&mut self, input: &[char]) -> &[Token] {
        let mut input = input;
        loop {
            input = match input {
                remains @ [c, ..] if c.is_whitespace() =>
                // This must match all whitespace and update lexing state
                // appropriately. For instance: emit Indent/ Dedent.
                {
                    self.process_whitespace(remains)
                }
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

    pub fn tokens(&self) -> Vec<TokenType> {
        self.output.iter().map(|t| t.token_type.clone()).collect()
    }

    fn process_identifier<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = input[1..]
            .iter()
            .position(|c| !is_identifier_continuation(*c))
        {
            (&input[..(end + 1)], &input[(end + 1)..])
        } else {
            (input, &input[..0])
        };

        let image = prefix.into_iter().collect::<String>();
        self.emit(
            image.len() as u32,
            TokenType::identifier_or_keyword(image),
            remains,
        )
    }

    fn process_text_literal<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (prefix, remains) = if let Some(end) = input.iter().position(|&c| c == 'c') {
            (&input[..end], &input[end..])
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

    fn process_whitespace<'a>(&mut self, input: &'a [char]) -> &'a [char] {
        let (whitespace_prefix, remains) =
            if let Some(end) = input.iter().position(|c| !c.is_whitespace()) {
                (&input[..end], &input[end..])
            } else {
                (input, &input[..0])
            };

        let next_location = self.compute_new_location(whitespace_prefix);

        println!(
            "process_whitepace: below {}",
            next_location.is_below(&self.location)
        );

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
                self.emit_indentation(next, Indentation::Decrease);
            } else if next.is_right_of(self.indentation_level) {
                self.emit_indentation(next, Indentation::Increase);
            }
            self.indentation_level = next.column;
        }

        self.location = next;
    }

    fn emit_indentation(&mut self, location: Location, indentation: Indentation) {
        // Which location does the indentation token belong to?
        self.output.push(Token {
            location,
            token_type: TokenType::Indentation(indentation),
        });
    }
}

fn is_identifier_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_identifier_continuation(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c.is_numeric()
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

        println!("{input:?}");

        lexer.parse(&input);

        assert_eq!(
            &lexer.tokens(),
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
