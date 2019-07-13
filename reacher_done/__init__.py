from gym.envs.registration import register

register(
    id='reacher-done-v0',
    entry_point='reacher_done.envs:ReacherDoneEnv',
    max_episode_steps=100,
)

register(
    id='fetch-reacher-done-v0',
    entry_point='reacher_done.envs:FetchReacherDoneEnv',
    max_episode_steps=100,
)
