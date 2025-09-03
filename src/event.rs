use crate::chain::{Block, Transaction};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    GenerateTransaction {
        transaction: Transaction
    },
    ReceiveTransaction {
        transaction: Transaction,
        sender: u32
    },
    MineBlock {
        block: Block
    },
    ReceiveBlock {
        block: Block,
        sender: u32
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub node_id: u32,
    pub event_type: EventType,
}

impl Event {
    pub fn new(node_id: u32, event_type: EventType) -> Self {
        Self {
            node_id,
            event_type,
        }
    }
}
