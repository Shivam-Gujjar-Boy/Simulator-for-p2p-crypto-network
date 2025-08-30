use std::fmt::Debug;

/// Global simulation configuration
#[derive(Clone)]
pub struct Config {
    pub num_nodes: u32,
    pub end_time_ms: f64,
    pub tx_interval_mean_ms: f64,
    pub mine_interval_ms: f64,
    pub positive_min_latency_ms: f64,   // base network latency
    pub fast_link_speed: u32,           // Fast Link Speed
    pub slow_link_speed: u32,           // Slow Link Speed
    pub max_block_txs: usize,           // max transactions per block
    pub fast_queuing_delay_mean: f64,   // queuing delay mean at fast node
    pub slow_queuing_delay_mean: f64,   // queuing delay mean at slow node
}

impl Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Config{{nodes:{}, end:{}ms, tx_interval:{}ms, mine_interval:{}ms}}",
            self.num_nodes, self.end_time_ms, self.tx_interval_mean_ms, self.mine_interval_ms
        )
    }
}

impl Config {
    pub fn new(
        num_nodes: u32,
        end_time_ms: f64,
        tx_interval_mean_ms: f64,
        positive_min_latency_ms: f64,
        mine_interval_ms: f64
    ) -> Self {
        Self {
            num_nodes,
            end_time_ms,
            tx_interval_mean_ms,
            positive_min_latency_ms,
            fast_link_speed: 100,
            slow_link_speed: 5,
            mine_interval_ms,
            max_block_txs: 999,
            fast_queuing_delay_mean: 0.96,
            slow_queuing_delay_mean: 19.2,
        }
    }
}
