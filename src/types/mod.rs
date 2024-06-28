mod tjw;

pub struct Tjw {
    hair_cnt: u32,
}

impl Tjw {
    pub fn new(hair_cnt: u32) -> Self {
        Self { hair_cnt }
    }

    pub fn ff(&self) {
        println!("tjw has {} hair", self.hair_cnt);
    }
}