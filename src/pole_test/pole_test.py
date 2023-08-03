import gym


def pole_test():
    env = gym.make("CartPole-v1")
    env.reset()
    for i_episode in range(100):
        done = False
        obs = env.reset()
        t = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            t += 1
            if done:
                print("Done after {} steps".format(t))
            break
    env.close()


pole_test()