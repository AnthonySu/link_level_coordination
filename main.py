import gym

from actor_critic_discrete import NewAgent
from utils import plot_learning_curve

if __name__ == '__main__':
    agent = NewAgent(alpha=0.00001, input_dims=[8], gamma=0.99,
                  n_actions=4, layer1_size=2048, layer2_size=512)

    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 100
    for i in range(num_episodes):

        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward

        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)

    filename = 'Lunar-Lander-actor-critic-new-agent-alpha00001-beta00005-2048x512fc-2000games.png'
    plot_learning_curve(score_history, filename=filename)