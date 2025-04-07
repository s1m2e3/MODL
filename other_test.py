import gym
from stable_baselines3 import DQN

# Create environment
env = gym.make("CartPole-v1")

# Initialize model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, batch_size=32)

# Train the agent
model.learn(total_timesteps=100000)

# Save and load model
model.save("dqn_cartpole")
del model
model = DQN.load("dqn_cartpole")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()
