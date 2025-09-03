#[derive(Debug)]
pub struct Mempool {
    pub transactions: Vec<u32>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: Vec::new(),
        }
    }

    pub fn add_transaction(&mut self, tx: u32) {
        self.transactions.push(tx);
    }

    pub fn get_transactions(&self) -> Vec<u32> {
        self.transactions.clone()
    }

    pub fn clear(&mut self) {
        self.transactions.clear();
    }

    pub fn size(&self) -> usize {
        self.transactions.len()
    }
}
