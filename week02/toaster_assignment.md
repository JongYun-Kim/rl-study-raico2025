# Assignment: ToastMaker-v0 — The Art of Perfect Toasting

---

## Overview
In this assignment, you will design and implement a simple simulated world where an intelligent agent learns to toast bread to perfection.  
The toaster’s heat changes over time, the bread’s doneness evolves in response to that heat, and your goal is to design this process so that a learning algorithm can discover an optimal toasting strategy.

You are free to define the numerical details of how the system behaves, but you must follow the specifications below carefully.

---

## Scenario
Each episode represents one toasting attempt.  


- The agent interacts with a toaster that can be turned **on** or **off** at each time step.  

- When the toaster is on, the internal temperature rises; when it is off, the toaster cools down.  

- The bread gradually becomes toasted depending on how much heat it has accumulated.  

- If it is taken out too early, it will be undercooked; if it stays too long, it burns.

- The aim is to achieve a “golden-brown” toastiness level by the end of each episode while minimizing unnecessary heating.

---

## Variables
The environment must include at least the following numerical quantities:

- **Toastiness** — a real number between `0` and `1`, representing how toasted the bread currently is.  
  - `0`: completely raw  
  - `1`: completely burnt  

- **Heat level** — a real number between `0` and `1`, representing the internal temperature of the toaster.  
  - When the toaster is **on**, this value should increase toward 1.  
  - When the toaster is **off**, it should decay toward 0.  

- Optional additional quantities such as elapsed time, ambient temperature, or accumulated energy cost may be included if desired.

All numerical quantities must evolve in **continuous space** (i.e., not discrete steps or categories).

---

## Interaction
At each environment evolution, the agent must choose between two options:

- Turn the toaster **ON**  
- Turn the toaster **OFF**

You may interpret these as binary control signals (`1` for ON, `0` for OFF).  
Each episode proceeds for a maximum of **64 steps**, unless it ends earlier because the toast is removed or burnt.

---

## Thermal and Toasting Process
You must specify clear equations describing how the toaster’s internal temperature and the bread’s toastiness evolve over time.  
The following requirements must be satisfied:

1. When the toaster is **on**, the internal temperature should rise toward 1 at a controllable rate.  
2. When the toaster is **off**, the internal temperature should decrease toward 0.  
3. The toastiness should increase based on how much heat has been applied over time.  
4. If toastiness exceeds `1.0`, the toast is considered **burnt** and the episode must terminate immediately.  
5. Random variation in heating efficiency must be introduced at each step.  
   - Example: the heat increase per step could vary by ±5%, drawn from a uniform distribution  
     \[
     U(0.95, 1.05)
     \]
   - You must clearly specify any random factors and their probability distributions in your implementation.  

The design should be simple and stable enough for an RL algorithm to learn without numerical instability.

---

## Reward Signal
At the end of an episode, the toast is evaluated for its quality.  
The **target toastiness** is fixed at `0.7`.

The reward function must include two parts:

1. **Quality term:**
   $$R_{quality} = 1 - |\text{toastiness} - 0.7|$$

2. **Energy penalty:**  
   During each step, a small negative reward should be given proportional to power usage  
   (for example, `-0.01` when the toaster is on).

The episode reward is the sum of all step penalties plus the terminal quality term.  
A higher total reward means the toast is close to golden-brown while consuming minimal energy.

---

## Uncertainty
The only required source of uncertainty is the slight randomness in heating efficiency.  
If you add additional randomness (for example, in initial heat level or ambient temperature), you must describe its distribution explicitly.

Example:
- Initial heat level ∼ `Uniform(0.0, 0.2)`
- Heating multiplier ∼ `Uniform(0.95, 1.05)`

---

## Termination
An episode ends when one of the following occurs:

- The bread becomes burnt (`toastiness > 1.0`)
- The agent decides to stop toasting
- The step limit (**64**) is reached

---

## Evaluation Criteria
Your trained agent will be evaluated on the following criteria:

1. **Average reward** across multiple random initial conditions.  
2. **Consistency** — how often it achieves near-target toastiness (for instance, within ±0.05 of 0.7).  
3. **Energy efficiency** — lower average total heating time with similar performance is better.  
4. **Stability** — learning should converge without erratic or oscillating behavior.

---

## Implementation Requirements
- The environment must be implemented using the **gymnasium API**.
- Register the environment when you make the environment.
- Use `NumPy` for matrix calculations and randomness.
- The rules above define the “physics” of the toaster world; you may tune parameters (e.g., rates of heating and cooling) as long as the behavior remains plausible.  
- Visualization or logging of toastiness over time is **encouraged** but not required.

---

## Deliverable
- Implementation of the abvoe Gymnasium environment ("ToasterMaker-v0").
- Simple evaluation script of a random agent in the environment (hint: a spec of gym envs supports random sampling in there)
