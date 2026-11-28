//epsilon greedy algorithm determines action
//epsilon determines the exploration, (1 - epsilon) is the exploitation
//gradually reduce epsilon as the episodes increases (global)
//previous q value is updated using the action chosen for the next state
//episode will end when: the agent reaches the end or lands on a hole or potentially time limit
//use ratatui to visualize

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

const GRID: usize = 4;

trait Environment {
    const ACTIONS: u8;
    type Observation;
    //will update agent's current_stage
    fn step(
        &mut self,
        action: (i8, i8),
        steps: &mut u32,
    ) -> (
        Self::Observation,
        f64,
        bool,
        bool,
        HashMap<&'static str, &'static str>,
    );
    fn reset(&mut self) -> Self::Observation;
    //provide the guarantee that every type that impl Environment will have a field of type ActionSpace
    fn action_space<'a>(&'a self) -> &'a ActionSpace;

    fn current_state<'a>(&'a self) -> &'a u8;
}

struct FrozenLake {
    map: [[u8; GRID]; GRID],
    current_state: u8,
    action_space: ActionSpace,
    is_slippery: bool,
    //truncation
    max_steps: u32,
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
        action: (i8, i8),
        steps: &mut u32,
    ) -> (
        Self::Observation,
        f64,
        bool,
        bool,
        HashMap<&'static str, &'static str>,
    ) {
        let mut info: HashMap<&'static str, &'static str> = HashMap::new();

        let row = (self.current_state / self.map.len() as u8) as i8 + action.0;
        let col = (self.current_state % self.map.len() as u8) as i8 + action.1;

        //tries to go out of bounds
        //assumes square map
        if row < 0 || col < 0 || row >= self.map.len() as i8 || col >= self.map.len() as i8 {
            info.insert("Error", "Out of bounds");
            //although it doesnt move, it will still count as a step
            *steps += 1;
            return (
                self.current_state,
                -0.05,
                false,
                *steps >= self.max_steps,
                info,
            );
        }

        //lands in a hole
        if self.map[row as usize][col as usize] == 1 {
            info.insert("Error", "Landed in a hole");
            *steps += 1;
            return (
                (row as usize * self.map[0].len() + col as usize) as u8,
                -0.25,
                true,
                *steps >= self.max_steps,
                info,
            );
        }
        //reaches goal
        else if self.map[row as usize][col as usize] == 2 {
            info.insert("Success", "Reached Goal!");
            *steps += 1;
            return (
                (row as usize * self.map[0].len() + col as usize) as u8,
                1.0,
                true,
                *steps >= self.max_steps,
                info,
            );
        }

        info.insert("Move", "Successful");
        *steps += 1;
        (
            (row as usize * self.map[0].len() + col as usize) as u8,
            0.0,
            false,
            *steps >= self.max_steps,
            info,
        )
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

#[derive(Debug)]
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
}

pub fn new() {
    //5. test loop (cpu)
    //6. rewrite logic using gpu kernels

    let training_config: TrainingConfig = TrainingConfig { episodes: 100 };

    let mut environment: FrozenLake = FrozenLake {
        map: loop {
            let map = generate_map();
            //break can return a value
            if check_map(&map) {
                break map;
            }
        },
        current_state: 0,
        action_space: ActionSpace,
        is_slippery: false,
        max_steps: 100,
    };

    //square grids
    const STATES: usize = GRID * GRID;
    const ACTIONS: usize = 4;
    let mut agent: Agent<STATES, ACTIONS> = Agent {
        q_table: QTable {
            q_values: [[0.0; ACTIONS]; STATES],
        },
        //exploration probability
        epsilon: 1.0,
        learning_rate: 0.1,
        //between 0 and 1
        discount_factor: 0.9,
        //exponential (non-linear)
        epsilon_decay: 0.995,
    };

    //main training loop

    println!("Q-table before training: {:?}", agent.q_table);

    println!("Map: {:?}", environment.map);
    for i in 0..training_config.episodes {
        let state = &environment.reset();
        let mut steps = 0;
        let mut done = false;
        let mut truncated = false;

        //println!("Episode: {}", i + 1 );
        while !done && !truncated {
            let action: (u8, (i8, i8)) =
                epsilon_greedy(&agent.epsilon, &agent.q_table, &environment);

            let (new_state, reward, d, t, info) = environment.step(action.1, &mut steps);

            done = d;
            truncated = t;

            let max = if done {
                0.0
            } else {
                agent.q_table.q_values[new_state as usize]
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap()
            };

            agent.q_table.q_values[environment.current_state as usize][action.0 as usize] =
                agent.q_table.q_values[environment.current_state as usize][action.0 as usize]
                    + agent.learning_rate
                        * (reward + agent.discount_factor * max
                            - agent.q_table.q_values[environment.current_state as usize]
                                [action.0 as usize]);

            environment.current_state = new_state;
        }
        agent.epsilon = (agent.epsilon * agent.epsilon_decay).max(0.0);
    }

    dbg!(agent.q_table.q_values);

    //store each move and manually check 
}

//exploration / expoitation
fn epsilon_greedy<T, const STATES: usize, const ACTIONS: usize>(
    epsilon: &f64,
    q_table: &QTable<STATES, ACTIONS>,
    environment: &T,
) -> (u8, (i8, i8))
//Observation
where
    T: Environment,
{
    use rand::prelude::*;

    let mut rng = rand::rng();
    let rnd: f64 = rng.random();
    let mut action: u8 = 0;
    //take random action
    if rnd < *epsilon {
        action = environment.action_space().sample(T::ACTIONS);
    } else {
        //argmax
        let mut max: f64 = f64::NEG_INFINITY;
        let mut ties = 0;
        for (i, val) in q_table.q_values[*environment.current_state() as usize]
            .into_iter()
            .enumerate()
        {
            //max = row.iter().max_by(|a, b| a.total_cmp(b));
            if val > max {
                max = val;
                action = i as u8;
                ties = 1;
            } else if val == max {
                ties += 1;
                //brute force but at least no additional memory allocation needed
                if rng.random_range(0..ties) == 0 {
                    action = i as u8;
                }
            }
        }
    }
    match action {
        0 => (0, (0, -1)), // Left
        1 => (1, (1, 0)),  // Down
        2 => (2, (0, 1)),  // Right
        3 => (3, (-1, 0)), // Up
        _ => (0, (0, 0)),
    }
}

fn generate_map() -> [[u8; GRID]; GRID] {
    use rand::prelude::*;

    const GRID: usize = 4;
    let mut rng = rand::rng();
    let mut map = [[0; GRID]; GRID];
    let hole_probability = 5;
    for row in 0..GRID {
        for col in 0..GRID {
            // Keep start and goal clear
            if (row == 0 && col == 0) || (row == GRID - 1 && col == GRID - 1) {
                continue;
            }

            if rng.random_range(0..hole_probability) == 0 {
                map[row][col] = 1;
            }
        }
    }

    map[GRID - 1][GRID - 1] = 2;

    map
}

//just bfs
fn check_map(map: &[[u8; GRID]; GRID]) -> bool {
    //vector double ended queue
    use std::collections::VecDeque;

    let mut visited = [[false; GRID]; GRID];
    let mut queue = VecDeque::new();

    queue.push_back((0, 0));
    visited[0][0] = true;

    let directions: [(i8, i8); 4] = [
        (-1, 0), // Up
        (1, 0),  // Down
        (0, -1), // Left
        (0, 1),  // Right
    ];

    while let Some((row, col)) = queue.pop_front() {
        //gets to goal
        if map[row][col] == 2 {
            return true;
        }

        for (dir_row, dir_col) in directions {
            let next_row = row as i8 + dir_row;
            let next_col = col as i8 + dir_col;

            if next_row < 0 || next_row >= GRID as i8 || next_col < 0 || next_col >= GRID as i8 {
                continue;
            }

            let next_row = next_row as usize;
            let next_col = next_col as usize;

            if !visited[next_row][next_col] && map[next_row][next_col] != 1 {
                visited[next_row][next_col] = true;
                queue.push_back((next_row, next_col));
            }
        }
    }

    false
}
