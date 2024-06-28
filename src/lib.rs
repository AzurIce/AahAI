pub mod types;

#[cfg(test)]
mod test {
    use super::types::Tjw;

    #[test]
    fn foo() {
        let tjw = Tjw::new(0);
        tjw.ff();
    }
}