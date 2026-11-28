//epsilon greedy algorithm determines action
//epsilon determines the exploration, (1 - epsilon) is the exploitation
//gradually reduce epsilon as the episodes increases (global)
//previous q value is updated using the action chosen for the next state
//episode will end when: the agent reaches the end or lands on a hole or potentially time limit
//use ratatui to visualize

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

trait Environment {
    const ACTIONS: u8;
    type Observation;
    //will update agent's current_stage
    fn step(&mut self, action: u8)
    -> (Self::Observation, f64, bool, bool, HashMap<String, String>);
    fn reset(&mut self) -> Self::Observation;
    //provide the guarantee that every type that impl Environment will have a field of type ActionSpace
    fn action_space<'a>(&'a self) -> &'a ActionSpace;

    fn current_state<'a>(&'a self) -> &'a u8;
}

struct FrozenLake {
    current_state: u8,
    action_space: ActionSpace,
    is_slippery: bool,
}

struct ActionSpace;

impl ActionSpace {
    //exploration
    fn sample(&self, actions: u8) -> u8 {
        use rand::prelude::*;

        let mut rng = rand::rng();
        rng.random_range(0..actions)
    }
}

impl Environment for FrozenLake {
    const ACTIONS: u8 = 4;
    type Observation = u8;
    //State, Reward, Terminated, Truncated, Info
    fn step(
        &mut self,
        action: u8,
    ) -> (Self::Observation, f64, bool, bool, HashMap<String, String>) {
        todo!();
    }

    fn reset(&mut self) -> Self::Observation {
        self.current_state = 0;
        self.current_state
    }

    fn action_space<'a>(&'a self) -> &'a ActionSpace {
        &self.action_space
    }

    fn current_state<'a>(&'a self) -> &'a u8 {
        //no cloning/copying, even if its 1 byte big
        &self.current_state
    }
}

struct QTable<const STATES: usize, const ACTIONS: usize> {
    q_values: [[f64; ACTIONS]; STATES],
}

//usize generic corresponds to the type of index
impl<const STATES: usize, const ACTIONS: usize> Index<usize> for QTable<STATES, ACTIONS> {
    type Output = [f64; ACTIONS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.q_values[index]
    }
}

//IndexMut: Index (Index is IndexMut's super trait, IndexMut inherits from Index)
impl<const STATES: usize, const ACTIONS: usize> IndexMut<usize> for QTable<STATES, ACTIONS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.q_values[index]
    }
}

impl<const STATES: usize, const ACTIONS: usize> serde::Serialize for QTable<STATES, ACTIONS> {
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("QTable", 1)?;

        //fill this with the const vec then serialize this vec
        let q_values: Vec<Vec<f64>> = self.q_values.iter().map(|row| row.to_vec()).collect();

        state.serialize_field("q_values", &q_values)?;

        state.end()
    }
}

impl<'de, const STATES: usize, const ACTIONS: usize> serde::Deserialize<'de>
    for QTable<STATES, ACTIONS>
{
    fn deserialize<T>(deserializer: T) -> Result<Self, T::Error>
    where
        T: serde::Deserializer<'de>,
    {
        //serialize into this
        #[derive(serde::Deserialize)]
        struct QTableTemp {
            q_values: Vec<Vec<f64>>,
        }

        let data = QTableTemp::deserialize(deserializer)?;

        let mut q_values = [[0.0; ACTIONS]; STATES];

        for (new_row, old_row) in q_values.iter_mut().zip(&data.q_values) {
            new_row.copy_from_slice(old_row);
        }

        //transfer the value to the actual struct's field
        Ok(Self { q_values })
    }
}

struct Agent<const STATES: usize, const ACTIONS: usize> {
    //[state][action]
    q_table: QTable<STATES, ACTIONS>,
    epsilon: f64,
    learning_rate: f64,
    discount_factor: f64,
    epsilon_decay: f64,
}

impl<const STATES: usize, const ACTIONS: usize> Agent<STATES, ACTIONS> {
    //use when: 1. exploitation 2. finding max when updating q-table
    fn get_action_values<'a>(&'a self, state: u8) -> &'a [f64] {
        &self.q_table[state as usize]
    }
}

struct TrainingConfig {
    episodes: usize,
    //truncation
    max_steps: usize,
}

pub fn new() {
    //3. finish epsilon greedy algorithm
    //4. build training loop (cpu)
    //5. test loop (cpu)
    //6. rewrite logic using gpu kernels

    let training_config: TrainingConfig = TrainingConfig {
        episodes: 100,
        max_steps: 50,
    };

    let environment: FrozenLake = FrozenLake {
        current_state: 0,
        action_space: ActionSpace,
        is_slippery: false,
    };

    //square grids
    const GRID: usize = 4;
    const STATES: usize = GRID * GRID;
    const ACTIONS: usize = 4;
    let agent: Agent<STATES, ACTIONS> = Agent {
        q_table: QTable {
            q_values: [[0.0; ACTIONS]; STATES],
        },
        //exploration probability
        epsilon: 1.0,
        learning_rate: 0.01,
        //between 0 and 1
        discount_factor: 0.9,
        epsilon_decay: 0.001,
    };
}

//exploration / expoitation
fn epsilon_greedy<T, const STATES: usize, const ACTIONS: usize>(
    epsilon: f64,
    q_table: &QTable<STATES, ACTIONS>,
    environment: &T,
) -> u8
//Observation
where
    T: Environment,
{
    use rand::prelude::*;

    let mut rng = rand::rng();
    let rnd: f64 = rng.random();

    //take random action
    if rnd < epsilon {
        return environment.action_space().sample(T::ACTIONS);
    }

    //argmax
    let max: f64 = 0.0;
    let mut action: u8 = 0;
    for (i, val) in q_table.q_values[*environment.current_state() as usize]
        .into_iter()
        .enumerate()
    {
        //max = row.iter().max_by(|a, b| a.total_cmp(b));
        if val > max {
            action = i as u8
        }
    }

    action
}
