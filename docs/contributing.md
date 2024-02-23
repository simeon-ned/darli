- Documentation and Info:
  - [ ] The cheat sheet for all necessary modules
  - [ ] The overall presentation and jupyter nootebook with all features
- Modeling Backends:
  - [x] Refactor the model class, add folder with backends
  - [x] Implement or use some ready implementation `rnea, crba, regressor, forward_kinematics` with JAX/PyTorch 
  - [x] Pure pinocchio backend and 'all-in-one' update  
- Model:
  - [ ] Friction and Joint inertias for active joints
  - [ ] Flexibility in Active Joints
  - [ ] Sparsity Patterns 
  - [ ] Add regular force, contact should inherite from force
- State Space:
  - [ ] State class that define how state is organized and mapped to `q, v`
  - [x] Integrators on manifolds
  - [ ] The discrete state space 
  - [ ] Hadnle all type of joints via proper integration in StateSpace class 
  - [ ] The proper Jacobians with quaternions in the state (tangent bundle)
  - [ ] Observables/Measurements function 
  - [ ] Hamiltonian State Space i.e `state = [position, momentum]`
- Parametric Model: 
  - [ ] Physical consistancy constraints and conversion to pseudo inertia (4x4) matrix and back.
  - [ ] Implement all dynamical attributes with auxilary `parameter` input
  - [ ] `Regressors` class
- Constrained and Contact Dynamics:
  - [ ] Features to create closed loops with holonomic/nonholonomic constraints (Lagrangian/Udwadia-Kalaba)
  - [ ] The minimal coordinates representation
- General:
  - [ ] Add number of bodies `nb` and their names to model attributes.  
  - [x] Setting free-flyer joints (to prevent locking floating body to ground)
  - [x] Locking joints feature to facilitate reduced models (i.e. saggital models of bipeds)
  - [ ] Configuration difference functions
  - [ ] Code generation and just in time compilation
  - [ ] Prepare docs and proper examples for each aspect of the library
  - [ ] Effort/velocity limits from urdf

Tutorials:
- [ ] Introduction
- [ ] Models and Robots
- [ ] State Space
- [ ] Parametric Models 
Each tutorial should have presentation. 



Possible Examples:
Model and Backends:
  - [ ] Joint/Task Inverse Dynamics
  - [ ] Optimization based Inverse Kinematics
  - [ ] Feedback Linearization with Flexible Joints 
  - [ ] Static Poses for Biped  
  - [ ] Trajectory Optimization
State Space: 
  - [ ] Linear Analysis 
  - [ ] LQR with Quaternions
  - [ ] LTV LQR
  - [ ] Model Predictive Control  
  - [ ] Extended and Uncendet Kalman Filters
Parametric:
- [ ] Passivity Based Robust and Adaptive control
- [ ] System Identification and Simulation Error 
- [ ] Sensitivity Analysis 
- [ ] Elipsoidal Uncertainty Propogation 
