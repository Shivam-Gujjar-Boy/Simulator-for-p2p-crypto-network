// 4 Types of Events - {generate Txn, Receive txn, Mine Block, Receive Block}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    GenerateTransaction {
        transaction_id: u32
    },
    ReceiveTransaction {
        transaction_id: u32,
        sender: u32
    },
    MineBlock {
        block_id: u32
    },
    ReceiveBlock {
        block_id: u32,
        sender: u32
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub node_id: u32,
    pub event_type: EventType,
}

impl Event {
    // Create a new event
    pub fn new(node_id: u32, event_type: EventType) -> Self {
        Self {
            node_id,
            event_type,
        }
    }
}
