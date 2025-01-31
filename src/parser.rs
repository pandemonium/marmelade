use core::panic;

use crate::{
    ast::{Constant, Expression, Identifier},
    lexer::{Keyword, LexicalAnalyzer, Literal, Operator, Token, TokenType},
};

pub type ParseResult<'a> = Result<(Expression, &'a [Token]), ParseError>;

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Token),
    UnexpectedRemains(Vec<Token>),
}

use Keyword::*;
use Token as T;
use TokenType as TT;
//use TT::{Identifier as Id, Keyword as Kw, Separator as Sep};

fn parse<'a>(tokens: &'a [Token]) -> Result<Expression, ParseError> {
    let (expression, remains) = parse_expression(tokens, 0)?;

    if let &[T(TT::End, ..)] = remains {
        Ok(expression)
    } else {
        Err(ParseError::UnexpectedRemains(remains.to_vec()))
    }
}

fn parse_prefix<'a>(tokens: &'a [Token]) -> ParseResult<'a> {
    match &tokens {
        &[T(TT::Keyword(Let), ..), T(TT::Identifier(binding), ..), T(TT::Equals, ..), remains @ ..] => {
            parse_binding(binding, remains)
        }
        &[T(TT::Literal(Literal::Integer(x)), ..), remains @ ..] => {
            Ok((Expression::Literal(Constant::Int(*x)), remains))
        }
        &[T(TT::Identifier(id), ..), remains @ ..] => {
            Ok((Expression::Variable(Identifier::new(&id)), remains))
        }
        otherwise => {
            panic!("{otherwise:?}");
        }
    }
}

fn parse_binding<'a>(binding: &str, prefix: &'a [Token]) -> ParseResult<'a> {
    let (bound, remains) = parse_expression(prefix, 0)?;

    println!("parse_binding: remains {remains:?}");

    match &remains {
        &[T(TT::Keyword(In), ..), remains @ ..] => {
            let (body, remains) = parse_expression(remains, 0)?;

            Ok((
                Expression::Binding {
                    binding: Identifier::new(binding),
                    bound: bound.into(),
                    body: body.into(),
                },
                remains,
            ))
        }
        _otherwise => todo!(),
    }
}

fn parse_expression<'a>(tokens: &'a [Token], precedence: usize) -> ParseResult<'a> {
    let (prefix, remains) = parse_prefix(tokens)?;
    parse_infix(prefix, remains, precedence)
}

fn parse_infix<'a>(lhs: Expression, tokens: &'a [Token], precedence: usize) -> ParseResult<'a> {
    match &tokens {
        &[T(TT::Operator(op), ..), remains @ ..] => {
            let op_precendence = op.precedence();
            if op_precendence > precedence {
                let (rhs, remains) = parse_expression(remains, op_precendence)?;
                let apply_op = make_binary_apply_op(*op, lhs, rhs);
                parse_infix(apply_op, remains, precedence)
            } else {
                Ok((lhs, tokens))
            }
        }
        otherwise => Ok((lhs, tokens)),
    }
}

fn make_binary_apply_op(op: Operator, lhs: Expression, rhs: Expression) -> Expression {
    let apply_lhs = Expression::Apply {
        function: Expression::Variable(operator_symbol(op)).into(),
        argument: lhs.into(),
    };
    Expression::Apply {
        function: apply_lhs.into(),
        argument: rhs.into(),
    }
}

fn operator_symbol(op: Operator) -> Identifier {
    match op {
        Operator::Plus => Identifier::new("builtin::plus"),
        Operator::Minus => Identifier::new("builtin::minus"),
        Operator::Times => Identifier::new("builtin::times"),
        Operator::Divides => Identifier::new("builtin::divides"),
        Operator::Modulo => Identifier::new("builtin::modulo"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn let_in() {
        let mut lexer = LexicalAnalyzer::default();

        let input = into_input("|let x = 10 in x + 20");

        let expr = super::parse(lexer.parse(&input)).unwrap();
        println!("{expr:?}");
    }
}
