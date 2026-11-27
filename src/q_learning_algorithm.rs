//epsilon greedy algorithm determines action
//epsilon determines the exploration, (1 - epsilon) is the exploitation
//gradually reduce epsilon as the episodes increases (global)
//previous q value is updated using the action chosen for the next state
//episode will end when: the agent reaches the end or lands on a hole or potentially time limit
//use ratatui to visualize

use std::collections::HashMap;

trait Environment {
    const ACTIONS: u8;
    type Observation;
    //will update agent's current_stage
    fn step(&mut self, action: u8) -> (Self::Observation, f64, bool, bool, HashMap<String, String>);
    fn reset(&mut self) -> Self::Observation;
}

struct FrozenLake {
    current_state: u8,
    action_space: ActionSpace,
}

struct ActionSpace;

impl ActionSpace {
    //exploration
    fn sample(&self, actions: u8) -> u8 {
        todo!();
    }
}

impl Environment for FrozenLake {
    const ACTIONS: u8 = 4;
    type Observation = u8;
    //State, Reward, Terminated, Truncated, Info
    fn step(&mut self, action: u8)-> (Self::Observation, f64, bool, bool, HashMap<String, String>) {
        todo!();
    }

    fn reset(&mut self) -> Self::Observation{
        self.current_state = 0;
        self.current_state
    }
}

struct Agent {
    //[state][action]
    q_table: Vec<Vec<f64>>,
    epsilon: f64,
    learning_rate: f64,
    discount_factor: f64,
    min_epsilon: f64,
    epsilon_decay: f64,
}

struct TrainingConfig {
    episodes: usize,
    //truncation
    max_steps: usize,
}

pub fn new() {}

//exploration / expoitation
fn epsilon_greedy(epsilon: &f64) {}