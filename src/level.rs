#[cfg(test)]
mod test {
    use std::fs;

    use aah_resource::level::Level;

    #[test]
    fn foo() {
        let level_name = "1-4";
        let levels = fs::read_to_string("./levels.json").unwrap();
        let levels = serde_json::from_str::<Vec<Level>>(&levels).unwrap();

        let level = levels
            .into_iter()
            .find(|l| l.code == level_name.to_string())
            .unwrap();
        for i in 0..level.height {
            for j in 0..level.width {
                let pos = level.calc_tile_screen_pos(i, j, false);
                println!("({}, {}) -> {:?}", i, j, pos);
            }
        }
    }
}
