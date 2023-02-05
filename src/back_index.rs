use std::ops::{Index, IndexMut};

#[derive(Clone, Copy)]
pub enum Side {
    Rev(usize)
}

impl Side {
    pub fn to_index(&self, len: usize) -> usize {
        match self {
            Side::Rev(i) => len - (i + 1)
        }    
    }
}

impl<T> Index<Side> for Vec<T> {
    type Output=T;

    fn index(&self, index: Side) -> &Self::Output {
        let len = self.len();
        &self[index.to_index(len)]
    }
}


impl<T> IndexMut<Side> for Vec<T> {
    fn index_mut(&mut self, index: Side) -> &mut Self::Output {
        let len = self.len();
        &mut self[index.to_index(len)]
    }
}