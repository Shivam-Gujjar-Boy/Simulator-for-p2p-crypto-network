use std::collections::BinaryHeap;
use std::cmp::Ordering;
use ordered_float::OrderedFloat;

use crate::event::Event;

/// A wrapper so we can order events in a min-heap by time.
#[derive(Debug, Eq, PartialEq)]
pub struct ScheduledEvent {
    pub time: OrderedFloat<f64>,
    pub event: Event,
}

// Convert default max-heap to min-heap
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other.time.cmp(&self.time)
    }
}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Scheduler (Event Queue)
#[derive(Debug)]
pub struct Scheduler {
    pub queue: BinaryHeap<ScheduledEvent>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    // Add a new event to min-heap
    pub fn schedule(&mut self, event: Event, time: OrderedFloat<f64>) {
        self.queue.push(ScheduledEvent { time, event });
    }

    // pop latest event
    pub fn next_event(&mut self) -> Option<(Event, f64)> {
        self.queue.pop().map(|e| (e.event, e.time.into_inner()))
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
