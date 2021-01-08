from deep_rl import *

set_one_thread()
select_device(0)

config = Config()
name = 'BeamRiderNoFrameskip-v4'
config = Config()
config.history_length = 4
config.task_fn = lambda: PixelAtari(name, frame_skip=4, history_length=config.history_length,
                                    log_dir='logs')
config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
# config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.1, 1e6))
config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
config.state_normalizer = ImageNormalizer()
config.reward_normalizer = SignNormalizer()
config.discount = 0.99
config.target_network_update_freq = 10000
config.exploration_steps= 50000
config.logger = get_logger()
# config.double_q = True
config.double_q = True
agent = (DQNAgent(config))
avg_test_rewards = []
for step in range(40000,int(3e7 + 1), 40000):
    weights = name+'_'+str(step)+".net"
    print(weights)
    agent.network = torch.load(weights)
    totalR = 0
    for _ in range(config.test_repetitions):
        reward, s = agent.episode(deterministic = True)
        totalR += reward

    avg_test_rewards.append(totalR/float(config.test_repetitions))
    if step % 1000000 == 0:
        print(avg_test_rewards)
