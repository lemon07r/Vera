//! Tree-sitter grammar loading for supported languages.
//!
//! Maps [`Language`] variants to tree-sitter grammar definitions.
//! Tier 1A languages get full AST-based parsing; others fall back to Tier 0.

use tree_sitter::Language as TsLanguage;

use crate::types::Language;

extern crate tree_sitter_hcl;

unsafe extern "C" {
    fn tree_sitter_sql() -> *const ();
    fn tree_sitter_hcl() -> *const ();
    fn tree_sitter_proto() -> *const ();
    fn tree_sitter_vue() -> *const ();
    fn tree_sitter_dockerfile() -> *const ();
}

/// Returns the tree-sitter grammar for a given language, if supported.
///
/// Returns `None` for languages without tree-sitter support (Tier 0 fallback).
pub fn tree_sitter_grammar(lang: Language) -> Option<TsLanguage> {
    let lang_fn = match lang {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::Bash => tree_sitter_bash::LANGUAGE.into(),
        Language::Kotlin => tree_sitter_kotlin_sg::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Zig => tree_sitter_zig::LANGUAGE.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Scala => tree_sitter_scala::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Haskell => tree_sitter_haskell::LANGUAGE.into(),
        Language::Elixir => tree_sitter_elixir::LANGUAGE.into(),
        Language::Dart => tree_sitter_dart::LANGUAGE.into(),
        Language::Sql => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_sql()) },
        Language::Hcl => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_hcl()) },
        Language::Protobuf => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_proto())
        },
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        Language::Scss => tree_sitter_scss::language(),
        Language::Vue => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_vue()) },
        Language::GraphQl => tree_sitter_graphql::LANGUAGE.into(),
        Language::CMake => tree_sitter_cmake::LANGUAGE.into(),
        Language::Dockerfile => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_dockerfile())
        },
        Language::Xml => tree_sitter_xml::LANGUAGE_XML.into(),
        // Tier 2A code languages
        Language::ObjectiveC => tree_sitter_objc::LANGUAGE.into(),
        Language::Perl => tree_sitter_perl::LANGUAGE.into(),
        Language::Julia => tree_sitter_julia::LANGUAGE.into(),
        Language::Nix => tree_sitter_nix::LANGUAGE.into(),
        Language::OCaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        Language::Groovy => tree_sitter_groovy::LANGUAGE.into(),
        Language::Clojure => tree_sitter_clojure_orchard::LANGUAGE.into(),
        Language::CommonLisp => tree_sitter_commonlisp::LANGUAGE_COMMONLISP.into(),
        Language::Erlang => tree_sitter_erlang::LANGUAGE.into(),
        Language::FSharp => tree_sitter_fsharp::LANGUAGE_FSHARP.into(),
        Language::Fortran => tree_sitter_fortran::LANGUAGE.into(),
        Language::PowerShell => tree_sitter_powershell::LANGUAGE.into(),
        Language::R => tree_sitter_r::LANGUAGE.into(),
        // Tier 2A batch 2 code languages
        Language::Matlab => tree_sitter_matlab::LANGUAGE.into(),
        Language::DLang => tree_sitter_d::LANGUAGE.into(),
        Language::Fish => tree_sitter_fish::language(),
        Language::Zsh => tree_sitter_zsh::LANGUAGE.into(),
        Language::Luau => tree_sitter_luau::LANGUAGE.into(),
        Language::Scheme => tree_sitter_scheme::LANGUAGE.into(),
        Language::Racket => tree_sitter_racket::LANGUAGE.into(),
        Language::Elm => tree_sitter_elm::LANGUAGE.into(),
        Language::Glsl => tree_sitter_glsl::LANGUAGE_GLSL.into(),
        Language::Hlsl => tree_sitter_hlsl::LANGUAGE_HLSL.into(),
        // Languages without tree-sitter grammar support → Tier 0 fallback
        Language::Toml
        | Language::Yaml
        | Language::Json
        | Language::Markdown
        | Language::Unknown => return None,
    };
    Some(lang_fn)
}

/// Returns whether a language has tree-sitter grammar support (Tier 1A).
pub fn has_grammar(lang: Language) -> bool {
    tree_sitter_grammar(lang).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_1a_languages_have_grammars() {
        let tier_1a = [
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Python,
            Language::Go,
            Language::Java,
            Language::C,
            Language::Cpp,
            Language::Ruby,
            Language::Bash,
            Language::Kotlin,
            Language::Swift,
            Language::Zig,
            Language::Lua,
            Language::Scala,
            Language::CSharp,
            Language::Php,
            Language::Haskell,
            Language::Elixir,
            Language::Dart,
            Language::Sql,
            Language::Hcl,
            Language::Protobuf,
        ];
        for lang in tier_1a {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn tier_1b_languages_have_grammars() {
        let tier_1b = [
            Language::Html,
            Language::Css,
            Language::Scss,
            Language::Vue,
            Language::GraphQl,
            Language::CMake,
            Language::Dockerfile,
            Language::Xml,
        ];
        for lang in tier_1b {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn tier_0_languages_have_no_grammar() {
        let tier_0 = [
            Language::Unknown,
            Language::Toml,
            Language::Yaml,
            Language::Json,
            Language::Markdown,
        ];
        for lang in tier_0 {
            assert!(
                !has_grammar(lang),
                "{lang} should NOT have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Rust).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&grammar).expect("grammar should load");
    }

    #[test]
    fn html_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Html).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("HTML grammar should load");
        let tree = parser.parse("<div></div>", None).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn css_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Css).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("CSS grammar should load");
        let tree = parser.parse("body { color: red; }", None).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn scss_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Scss).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("SCSS grammar should load");
        let tree = parser
            .parse("$color: red; body { color: $color; }", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn vue_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Vue).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Vue grammar should load");
        let tree = parser
            .parse("<template><div></div></template>", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn graphql_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::GraphQl).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("GraphQL grammar should load");
        let tree = parser.parse("type Query { hello: String }", None).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn cmake_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::CMake).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("CMake grammar should load");
        let tree = parser
            .parse("cmake_minimum_required(VERSION 3.10)", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn dockerfile_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Dockerfile).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Dockerfile grammar should load");
        let tree = parser.parse("FROM ubuntu:20.04\n", None).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn xml_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Xml).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("XML grammar should load");
        let tree = parser
            .parse("<?xml version=\"1.0\"?><root><item/></root>", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    // ── Tier 2A grammar loading tests ─────────────────────────

    #[test]
    fn tier_2a_languages_have_grammars() {
        let tier_2a = [
            Language::ObjectiveC,
            Language::Perl,
            Language::Julia,
            Language::Nix,
            Language::OCaml,
            Language::Groovy,
            Language::Clojure,
            Language::CommonLisp,
            Language::Erlang,
            Language::FSharp,
            Language::Fortran,
            Language::PowerShell,
            Language::R,
        ];
        for lang in tier_2a {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn objectivec_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::ObjectiveC).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("ObjC grammar should load");
        let tree = parser
            .parse("@interface Foo : NSObject\n- (void)bar;\n@end\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn perl_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Perl).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Perl grammar should load");
        let tree = parser
            .parse("sub hello { print \"hello\\n\"; }\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn julia_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Julia).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Julia grammar should load");
        let tree = parser
            .parse("function hello()\n  println(\"hello\")\nend\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn nix_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Nix).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Nix grammar should load");
        let tree = parser
            .parse("{ pkgs ? import <nixpkgs> {} }: pkgs.hello\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn ocaml_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::OCaml).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("OCaml grammar should load");
        let tree = parser
            .parse("let hello () = print_endline \"hello\"\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn groovy_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Groovy).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Groovy grammar should load");
        let tree = parser
            .parse("def hello() { println 'hello' }\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn clojure_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Clojure).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Clojure grammar should load");
        let tree = parser
            .parse("(defn hello [] (println \"hello\"))\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn commonlisp_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::CommonLisp).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Common Lisp grammar should load");
        let tree = parser
            .parse("(defun hello () (format t \"hello~%\"))\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn erlang_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Erlang).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Erlang grammar should load");
        let tree = parser
            .parse("-module(hello).\nhello() -> ok.\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn fsharp_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::FSharp).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("F# grammar should load");
        let tree = parser
            .parse("let hello () = printfn \"hello\"\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn fortran_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Fortran).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Fortran grammar should load");
        let tree = parser
            .parse(
                "program hello\n  print *, 'hello'\nend program hello\n",
                None,
            )
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn powershell_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::PowerShell).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("PowerShell grammar should load");
        let tree = parser
            .parse("function Hello { Write-Host 'hello' }\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn r_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::R).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("R grammar should load");
        let tree = parser
            .parse("hello <- function() { print(\"hello\") }\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    // ── Tier 2A batch 2 grammar loading tests ─────────────────

    #[test]
    fn tier_2a_batch2_languages_have_grammars() {
        let tier_2a_b2 = [
            Language::Matlab,
            Language::DLang,
            Language::Fish,
            Language::Zsh,
            Language::Luau,
            Language::Scheme,
            Language::Racket,
            Language::Elm,
            Language::Glsl,
            Language::Hlsl,
        ];
        for lang in tier_2a_b2 {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn matlab_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Matlab).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("MATLAB grammar should load");
        let tree = parser
            .parse("function y = square(x)\n  y = x^2;\nend\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn dlang_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::DLang).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("D grammar should load");
        let tree = parser
            .parse("void main() { writeln(\"hello\"); }\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn fish_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Fish).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Fish grammar should load");
        let tree = parser
            .parse("function hello\n  echo hello\nend\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn zsh_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Zsh).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Zsh grammar should load");
        let tree = parser
            .parse("function hello() {\n  echo hello\n}\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn luau_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Luau).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Luau grammar should load");
        let tree = parser
            .parse("local function hello()\n  print(\"hello\")\nend\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn scheme_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Scheme).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Scheme grammar should load");
        let tree = parser
            .parse("(define (hello) (display \"hello\"))\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn racket_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Racket).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Racket grammar should load");
        let tree = parser
            .parse(
                "#lang racket\n(define (hello) (displayln \"hello\"))\n",
                None,
            )
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn elm_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Elm).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("Elm grammar should load");
        let tree = parser
            .parse(
                "module Main exposing (main)\n\nmain =\n  text \"hello\"\n",
                None,
            )
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn glsl_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Glsl).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("GLSL grammar should load");
        let tree = parser
            .parse("void main() {\n  gl_FragColor = vec4(1.0);\n}\n", None)
            .unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn hlsl_grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Hlsl).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&grammar)
            .expect("HLSL grammar should load");
        let tree = parser
            .parse(
                "float4 main(float4 pos : SV_Position) : SV_Target {\n  return float4(1, 0, 0, 1);\n}\n",
                None,
            )
            .unwrap();
        assert!(!tree.root_node().has_error());
    }
}
