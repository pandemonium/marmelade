use std::{env, fs};

use marmelade::{context::Linkage, stdlib};

fn main() {
    println!("Marmelade Compiler v0.69");
    if let Some(file_path) = env::args().collect::<Vec<_>>().get(1) {
        println!("Parsing {file_path}");

        let source_code = fs::read_to_string(file_path)
            .expect(&format!("Unable to read {file_path}"))
            .chars()
            .collect::<Vec<_>>();
        let mut linkage = Linkage::new(&source_code);
        stdlib::import(&mut linkage).unwrap();

        let return_value = linkage.typecheck_and_interpret().expect("There are errors");

        println!("#### {return_value}");
    } else {
        println!("marmelade <file-path.lady>")
    }
}
