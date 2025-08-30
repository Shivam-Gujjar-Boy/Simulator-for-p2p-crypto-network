use crate::chain::Transaction;

#[derive(Debug)]
pub struct Mempool {
    pub transactions: Vec<Transaction>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: Vec::new(),
        }
    }

    pub fn add_transaction(&mut self, tx: Transaction) {
        self.transactions.push(tx);
    }

    pub fn get_transactions(&self) -> Vec<Transaction> {
        self.transactions.clone()
    }

    pub fn clear(&mut self) {
        self.transactions.clear();
    }

    pub fn size(&self) -> usize {
        self.transactions.len()
    }
}
