import numpy as np

from madp_python3_wrapper import MADPDecPOMDPDiscrete as madp


def make_transition_matrix(decpomdp):
    state_transition_matrix = np.zeros(
        shape=(decpomdp.num_states(), decpomdp.num_states(), decpomdp.num_joint_actions()))
    for next_state in range(decpomdp.num_states()):
        for old_state in range(decpomdp.num_states()):
            for joint_action in range(decpomdp.num_joint_actions()):
                state_transition_matrix[next_state][old_state][joint_action] = decpomdp.transition_probability(
                    next_state, old_state, joint_action)

    return state_transition_matrix


def make_observation_matrix(decpomdp):
    observation_matrix = np.zeros(shape=(
    decpomdp.num_joint_observations(), decpomdp.num_states(), decpomdp.num_joint_actions()))
    for joint_observation in range(decpomdp.num_joint_observations()):
        for state in range(decpomdp.num_states()):
            for joint_action in range(decpomdp.num_joint_actions()):
                observation_matrix[joint_observation][state][joint_action] = decpomdp.observation_probability(
                    joint_observation, state, joint_action)

    return observation_matrix


def make_reward_matrix(decpomdp):
    reward_matrix = np.zeros(shape=(decpomdp.num_states(), decpomdp.num_joint_actions()))
    for state in range(decpomdp.num_states()):
        for joint_action in range(decpomdp.num_joint_actions()):
            reward_matrix[state][joint_action] = decpomdp.reward(state, joint_action)

    return reward_matrix


########################################################################################################################
decpomdp = madp("../problems/dectiger.dpomdp")

num_agents = decpomdp.num_agents()
print("Number of agents: ", num_agents)

num_states = decpomdp.num_states()
print("Number of states: ", num_states)

num_joint_actions = decpomdp.num_joint_actions()
print("Number of joint actions: ", num_joint_actions)

num_joint_observations = decpomdp.num_joint_observations()
print("Number of joint observations: ", num_joint_observations)

for agent in range(num_agents):
    num_actions = decpomdp.num_actions(agent)
    print("Agent %s can perform %s actions" % (agent, num_actions))
    for action in range(num_actions):
        action_name = decpomdp.action_name(agent, action)
        print("Name of action %s of agent %s is %s" % (action, agent, action_name))

    num_observations = decpomdp.num_observations(agent)
    print("Agent %s can measure %s observations" % (agent, num_observations))
    for observation in range(num_observations):
        observation_name = decpomdp.observation_name(agent, observation)
        print("Name of observation %s of agent %s is %s" % (observation, agent, observation_name))


for state in range(num_states):
    state_name = decpomdp.state_name(state)
    print("Name of state %s is %s" % (state, state_name))

    initial_belief = decpomdp.initial_belief_at(state)
    print("Initial belief at state %s is %s" % (state, initial_belief))

    for joint_action in range(num_joint_actions):
        reward = decpomdp.reward(state, joint_action)
        print("Reward in state %s when joint action %s is taken is %s" % (state, joint_action, reward))


observation_matrix = make_observation_matrix(decpomdp)
print("Observation matrix is: ", observation_matrix)

transition_matrix = make_transition_matrix(decpomdp)
print("Transition matrix is: ", transition_matrix)

reward_matrix = make_reward_matrix(decpomdp)
print("Reward matrix is: ", reward_matrix)