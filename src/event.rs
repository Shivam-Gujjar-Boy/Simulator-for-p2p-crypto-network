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
    pub fn new(node_id: u32, event_type: EventType) -> Self {
        Self {
            node_id,
            event_type,
        }
    }
}
