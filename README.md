# Reinforcement Learning - Mini Capstone  
## Optimizing the Cutting of Flooring in the Cutting Stock Problem  
### FPT University
### AI17C - Group 5  
### Slide report : https://www.canva.com/design/DAGiWMeB0ao/qTcURPkrzyYd1bCbcBKk4Q/edit?utm_content=DAGiWMeB0ao&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## 1. Introduction  
In manufacturing industries, optimizing material usage is crucial to reducing costs and minimizing waste. The Cutting Stock Problem (CSP) is a classic optimization challenge where large raw materials must be cut into smaller pieces while minimizing leftover waste. This problem is particularly relevant in flooring manufacturing, where efficiently cutting materials can lead to significant economic and environmental benefits.  

### 1.1 Why the Cutting Stock Problem?  
The Cutting Stock Problem (CSP) appears in industries such as textile, metalworking, and flooring production. Traditional approaches, such as Linear Programming (LP) and heuristic algorithms, have been widely used but often struggle with adaptability and efficiency in dynamic environments. Reinforcement Learning (RL) offers a promising alternative by learning from experience and continuously improving cutting strategies.  

## 2. Project Objectives  
- Implement Reinforcement Learning techniques to optimize cutting patterns in the flooring industry.  
- Minimize material waste while maximizing production efficiency.  
- Compare RL-based solutions with traditional heuristic algorithms.  
- Develop a reusable and scalable framework applicable to various cutting stock scenarios.  

## 3. Methodology  
### 3.1 Problem Formulation  
The Cutting Stock Problem is modeled as a sequential decision-making problem, where an RL agent learns an optimal cutting strategy based on material availability and demand constraints.  

### 3.2 Reinforcement Learning Approach  
- **State Space**: Represents the available stock lengths and remaining demand.  
- **Action Space**: Defines the possible cutting patterns.  
- **Reward Function**: Encourages strategies that minimize waste and maximize material usage.  
- **Algorithm**: We experiment with **Q-Learning Algorithm** and **Actor-Critic Algorithms**.  

### 3.3 Baseline Comparisons  
- **Heuristic Algorithms**: First-Fit Algorithm, Best-Fit Algorithm, Combination Algorithm.  
- **Performance Metrics**: Waste reduction, computational efficiency, adaptability to different constraints.  

## 4. Expected Outcomes  
- A trained RL agent capable of making optimal cutting decisions.  
- Performance comparisons showing RL's advantages over traditional methods.  
- A generalized RL-based framework adaptable for multiple manufacturing sectors.  
- Open-source code, documentation, and experiment results for further research and industrial use.  

## 5. Team Members  
### AI17C - Group 5  
- Nguyễn Minh Phát - Leader  
- Nguyễn Trọng Nghĩa  
- Phạm Gia Thịnh  
- Chế Minh Quang  
- Bùi Đình Thanh Danh  

## 6. Technologies Used  
- **Programming Language**: Python  
- **Machine Learning Frameworks**: TensorFlow, PyTorch  
- **Reinforcement Learning Libraries**: Stable-Baselines3, OpenAI Gym  
- **Optimization Tools**: NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  

## 7. How to Run the Project  
Clone the repository:  
```sh  
git git clone https://github.com/PhamGiaThinh03/Cutting-Stock-Reforment-learning.git
cd Cutting-Stock-Reforment-learning
```

## 8. References  
- Gilmore, P. C., & Gomory, R. E. (1961). A linear programming approach to the cutting-stock problem. *Operations Research.*  
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.*  
- OpenAI Research on Reinforcement Learning.  

## 9. Acknowledgments  
We extend our gratitude to our professors, mentors, and peers for their continuous support and insightful discussions throughout this project.  
