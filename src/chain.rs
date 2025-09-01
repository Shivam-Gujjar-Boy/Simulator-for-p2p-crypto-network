use std::collections::HashMap;

use ordered_float::OrderedFloat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub id: u32,
    pub from: Option<u32>,
    pub to: u32,
    pub amount: i64,
    pub created_at: OrderedFloat<f64>,
    pub received_at: OrderedFloat<f64>,
    pub received_from: Option<u32>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub block_id: u32,
    pub prev_id: Option<u32>,
    pub transactions: Vec<Transaction>,
    pub timestamp: OrderedFloat<f64>,
    pub block_height: u32,
    pub miner: Option<u32>,
    pub received_at: OrderedFloat<f64>,
}

#[derive(Debug)]
pub struct BlockchainTree {
    pub blocks: HashMap<u32, Block>,
    pub children: HashMap<u32, Vec<u32>>,
    pub tip: u32,
    pub genesis: u32,
}

impl BlockchainTree {
    pub fn new() -> Self {
        let mut blocks = HashMap::new();

        let genesis_block = Block {
            block_id: 0,
            prev_id: None,
            transactions: vec![],
            timestamp: OrderedFloat(0.0),
            block_height: 0,
            miner: None,
            received_at: OrderedFloat(0.0),
        };

        blocks.insert(genesis_block.block_id, genesis_block);

        Self {
            blocks,
            children: HashMap::new(),
            tip: 0,
            genesis: 0,
        }
    }

    // pub fn add_block(&mut self, block: Block) {
    //     self.blocks.push(block);
    // }

    // pub fn get_latest_block(&self) -> Option<&Block> {
    //     self.blocks.last()
    // }

    // pub fn get_length(&self) -> usize {
    //     self.blocks.len()
    // }
}
