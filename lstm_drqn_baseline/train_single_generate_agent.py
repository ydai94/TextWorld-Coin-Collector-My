import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from agent import RLAgent

from helpers.generic import SlidingAverage, to_np
from helpers.generic import get_experiment_dir, dict2list
from helpers.setup_logger import setup_logging, log_git_commit
from test_agent import test
logger = logging.getLogger(__name__)
from textworld.core import EnvInfos
import gym
import gym_textworld  # Register all textworld environments.

import textworld

class TemplateActionGeneratorJeri:
    '''
    Generates actions using the template-action-space.
    :param rom_bindings: Game-specific bindings from :meth:`jericho.FrotzEnv.bindings`.
    :type rom_bindings: Dictionary
    '''
    def __init__(self, rom_bindings):
        self.rom_bindings = rom_bindings
        grammar = rom_bindings['grammar'].split(';')
        max_word_length = rom_bindings['max_word_length']
        self.templates = self._preprocess_templates(grammar, max_word_length)
        # Enchanter and Spellbreaker only recognize abbreviated directions
        if rom_bindings['name'] in ['enchanter', 'spellbrkr', 'murdac']:
            for act in ['northeast','northwest','southeast','southwest']:
                self.templates.remove(act)
            self.templates.extend(['ne','nw','se','sw'])

    def _preprocess_templates(self, templates, max_word_length):
        '''
        Converts templates with multiple verbs and takes the first verb.
        '''
        out = []
        vb_usage_fn = lambda verb: verb_usage_count(verb, max_word_length)
        p = re.compile(r'\S+(/\S+)+')
        for template in templates:
            if not template:
                continue
            while True:
                match = p.search(template)
                if not match:
                    break
                verb = max(match.group().split('/'), key=vb_usage_fn)
                template = template[:match.start()] + verb + template[match.end():]
            ts = template.split()
            out.append(template)
        return out


    def generate_actions(self, objs):
        '''
        Given a list of objects present at the current location, returns
        a list of possible actions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :returns: List of action-strings.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> env.act_gen.generate_actions(interactive_objs)
        ['wake', 'wake up', 'wash', ..., 'examine wallet', 'remove phone', 'taste keys']
        '''
        actions = []
        for template in self.templates:
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(template)
            elif holes == 1:
                actions.extend([template.replace('OBJ', obj) for obj in objs])
            elif holes == 2:
                for o1 in objs:
                    for o2 in objs:
                        if o1 != o2:
                            actions.append(template.replace('OBJ', o1, 1).replace('OBJ', o2, 1))
        return actions


    def generate_template_actions(self, objs, obj_ids):
        '''
        Given a list of objects and their corresponding vocab_ids, returns
        a list of possible TemplateActions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :param obj_ids: List of ids corresponding to the tokens of each object.
        :type obj_ids: List of int
        :returns: List of :class:`jericho.defines.TemplateAction`.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> interactive_obj_ids = [718, 325, 64]
        >>> env.act_gen.generate_template_actions(interactive_objs, interactive_obj_ids)
        [
          TemplateAction(action='wake', template_id=0, obj_ids=[]),
          TemplateAction(action='wake up', template_id=1, obj_ids=[]),
          ...
          TemplateAction(action='turn phone on', template_id=55, obj_ids=[718]),
          TemplateAction(action='put wallet on keys', template_id=65, obj_ids=[64, 325])
         ]
        '''
        assert len(objs) == len(obj_ids)
        actions = []
        for template_idx, template in enumerate(self.templates):
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(defines.TemplateAction(template, template_idx, []))
            elif holes == 1:
                for noun, noun_id in zip(objs, obj_ids):
                    actions.append(
                        defines.TemplateAction(template.replace('OBJ', noun),
                                               template_idx, [noun_id]))
            elif holes == 2:
                for o1, o1_id in zip(objs, obj_ids):
                    for o2, o2_id in zip(objs, obj_ids):
                        if o1 != o2:
                            actions.append(
                                defines.TemplateAction(
                                    template.replace('OBJ', o1, 1).replace('OBJ', o2, 1),
                                    template_idx, [o1_id, o2_id]))
        return actions

import jericho
import textworld
import re
from collections import defaultdict

def _load_bindings_from_tw(state, story_file, seed):
    bindings = {}
    g1 = [re.sub('{.*?}', 'OBJ', s) for s in state.command_templates]
    g = list(set([re.sub('go .*', 'go OBJ', s) for s in g1]))
    g.remove('drop OBJ')
    g.remove('examine OBJ')
    g.remove('inventory')
    g.remove('look')
    bindings['grammar'] = ';'.join(g)
    bindings['max_word_length'] = len(max(state.verbs + state.entities, key=len))
    bindings['minimal_actions'] = '/'.join(state['extra.walkthrough'])
    bindings['name'] = state['extra.uuid']
    bindings['rom'] = story_file.split('/')[-1]
    bindings['seed'] = seed
    bindings['walkthrough'] = bindings['minimal_actions']
    return bindings

class JeriWorld:
    def __init__(self, story_file, seed=None, style='jericho', infos = None):
        self.jeri_style = style.lower() == 'jericho'
        if self.jeri_style:
            self._env = textworld.start(story_file, infos=infos)
            state = self._env.reset()
            self.tw_games = True
            self._seed = seed
            self.bindings = None
            if state.command_templates is None:
                self.tw_games = False
                del self._env
                self._env = jericho.FrotzEnv(story_file, seed)
                self.bindings = self._env.bindings
                self._world_changed = self._env._world_changed
                self.act_gen = self._env.act_gen
            else:
                self.bindings = _load_bindings_from_tw(state, story_file, seed)
                self._world_changed = self._env._jericho._world_changed
                self.act_gen = TemplateActionGeneratorJeri(self.bindings)
                self.seed(seed)
        else:
            self._env = textworld.start(story_file, infos=infos)

    def __del__(self):
        del self._env
 
    
    def reset(self):
        if self.jeri_style:
            if self.tw_games:
                state = self._env.reset()
                raw = state['description']
                return raw, {'moves':state.moves, 'score':state.score}
            return self._env.reset()
        else:
            return self._env.reset()
    
    def load(self, story_file, seed=None):
        if self.jeri_style:
            if self.tw_games:
                self._env.load(story_file)
            else:
                self._env.load(story_file, seed)
        else:
            self._env.load(story_file)

    def step(self, action):
        if self.jeri_style:
            if self.tw_games:
                old_score = self._env.state.score
                next_state = self._env.step(action)[0]
                s_action = re.sub(r'\s+', ' ', action.strip())
                score = self._env.state.score
                reward = score - old_score
                self._world_changed = self._env._jericho._world_changed
                return next_state.description, reward, (next_state.lost or next_state.won),\
                  {'moves':next_state.moves, 'score':next_state.score}
            else:
                self._world_changed = self._env._world_changed
            return self._env.step(action)
        else:
            return self._env.step(action)

    def bindings(self):
        if self.jeri_style:
            return self.bindings
        else:
            return None

    def _emulator_halted(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env._env._emulator_halted()
            return self._env._emulator_halted()
        else:
            return None

    def game_over(self):
        if self.jeri_style:
            if self.tw_games:
                self._env.state['lost']
            return self._env.game_over()
        else:
            return None

    def victory(self):
        if self.jeri_style:
            if self.tw_games:
                self._env.state['won']
            return self._env.victory()
        else:
            return None

    def seed(self, seed=None):
        if self.jeri_style:
            self._seed = seed
            return self._env.seed(seed)
        else:
            return None
    
    def close(self):
        if self.jeri_style:
            self._env.close()
        else:
            pass

    def copy(self):
        return self._env.copy()

    def get_walkthrough(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['extra.walkthrough']
            return self._env.get_walkthrough()
        else:
            return None

    def get_score(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['score']
            return self._env.get_score()
        else:
            return None

    def get_dictionary(self):
        if self.jeri_style:
            if self.tw_games:
                state = self._env.state
                return state.entities + state.verbs
            return self._env.get_dictionary()
        else:
            state = self._env.state
            return state.entities + state.verbs

    def get_state(self):
        if self.jeri_style:
            if self.tw_games:
                return self._env._jericho.get_state()
            return self._env.get_state
        else:
            return None
    
    def set_state(self, state):
        if self.jeri_style:
            if self.tw_games:
                self._env._jericho.set_state(state)
            else:
                self._env.get_state
        else:
            pass

    def get_valid_actions(self, use_object_tree=True, use_ctypes=True, use_parallel=True):
        if self.jeri_style:
            if self.tw_games:
                return self._env.state['admissible_commands']
            return self._env.get_valid_actions(use_object_tree, use_ctypes, use_parallel)
        else:
            pass
    
    def _identify_interactive_objects(self, observation='', use_object_tree=False):
        """
        Identifies objects in the current location and inventory that are likely
        to be interactive.
        :param observation: (optional) narrative response to the last action, used to extract candidate objects.
        :type observation: string
        :param use_object_tree: Query the :doc:`object_tree` for names of surrounding objects.
        :type use_object_tree: boolean
        :returns: A list-of-lists containing the name(s) for each interactive object.
        :Example:
        >>> from jericho import *
        >>> env = FrotzEnv('zork1.z5')
        >>> obs, info = env.reset()
        'You are standing in an open field west of a white house with a boarded front door. There is a small mailbox here.'
        >>> env.identify_interactive_objects(obs)
        [['mailbox', 'small'], ['boarded', 'front', 'door'], ['white', 'house']]
        .. note:: Many objects may be referred to in a variety of ways, such as\
        Zork1's brass latern which may be referred to either as *brass* or *lantern*.\
        This method groups all such aliases together into a list for each object.
        """
        if self.jeri_style:
            if self.tw_games:
                objs = set()
                state = self.get_state()

                if observation:
                    # Extract objects from observation
                    obs_objs = extract_objs(observation)
                    obs_objs = [o + ('OBS',) for o in obs_objs]
                    objs = objs.union(obs_objs)

                # Extract objects from location description
                self.set_state(state)
                look = clean(self.step('look')[0])
                look_objs = extract_objs(look)
                look_objs = [o + ('LOC',) for o in look_objs]
                objs = objs.union(look_objs)

                # Extract objects from inventory description
                self.set_state(state)
                inv = clean(self.step('inventory')[0])
                inv_objs = extract_objs(inv)
                inv_objs = [o + ('INV',) for o in inv_objs]
                objs = objs.union(inv_objs)
                self.set_state(state)

                # Filter out the objects that aren't in the dictionary
                dict_words = [w for w in self.get_dictionary()]
                max_word_length = max([len(w) for w in dict_words])
                to_remove = set()
                for obj in objs:
                    if len(obj[0].split()) > 1:
                        continue
                    if obj[0][:max_word_length] not in dict_words:
                        to_remove.add(obj)
                objs.difference_update(to_remove)
                objs_set = set()
                for obj in objs:
                    if obj[0] not in objs_set:
                        objs_set.add(obj[0])
                return objs_set
            return self._env._identify_interactive_objects(observation=observation, use_object_tree=use_object_tree)
        else:
            return None

    def find_valid_actions(self, possible_acts=None):
        if self.jeri_style:
            if self.tw_games:
                diff2acts = {}
                state = self.get_state()
                candidate_actions = self.get_valid_actions()
                for act in candidate_actions:
                    self.set_state(state)
                    self.step(act)
                    diff = self._env._jericho._get_world_diff()
                    if diff in diff2acts:
                        if act not in diff2acts[diff]:
                            diff2acts[diff].append(act)
                    else:
                        diff2acts[diff] = [act]
                self.set_state(state)
                return diff2acts
            else:
                admissible = []
                candidate_acts = self._env._filter_candidate_actions(possible_acts).values()
                true_actions = self._env.get_valid_actions()
                for temp_list in candidate_acts:
                    for template in temp_list:
                        if template.action in true_actions:
                            admissible.append(template)
                return admissible
        else:
            return None


    def _score_object_names(self, interactive_objs):
        """ Attempts to choose a sensible name for an object, typically a noun. """
        if self.jeri_style:
            def score_fn(obj):
                score = -.01 * len(obj[0])
                if obj[1] == 'NOUN':
                    score += 1
                if obj[1] == 'PROPN':
                    score += .5
                if obj[1] == 'ADJ':
                    score += 0
                if obj[2] == 'OBJTREE':
                    score += .1
                return score
            best_names = []
            for desc, objs in interactive_objs.items():
                sorted_objs = sorted(objs, key=score_fn, reverse=True)
                best_names.append(sorted_objs[0][0])
            return best_names
        else:
            return None

    def get_world_state_hash(self):
        if self.jeri_style:
            if self.tw_games:
                return None
            else:
                return self._env.get_world_state_hash()
        else:
            return None


def train(config):
    # train env
    print('Setting up TextWorld environment...')
    batch_size = config['training']['scheduling']['batch_size']
    '''env_id = gym_textworld.make_batch(env_id=config['general']['env_id'],
                                      batch_size=batch_size,
                                      parallel=True)
'''
    info = EnvInfos(objective=True,description=True,inventory=True,feedback=True,intermediate_reward=True,admissible_commands=True)
    env = JeriWorld('/content/tw_games/coin_collector.z8', infos=info, style = 'textworld')
    env.reset()
    # valid and test env
    run_test = config['general']['run_test']
    if run_test:
        test_batch_size = config['training']['scheduling']['test_batch_size']
        # valid
        valid_env_name = config['general']['valid_env_id']

        valid_env_id = gym_textworld.make_batch(env_id=valid_env_name, batch_size=test_batch_size, parallel=True)
        valid_env = gym.make(valid_env_id)
        valid_env.seed(config['general']['random_seed'])

        # test
        test_env_name_list = config['general']['test_env_id']
        assert isinstance(test_env_name_list, list)

        test_env_id_list = [gym_textworld.make_batch(env_id=item, batch_size=test_batch_size, parallel=True) for item in test_env_name_list]
        test_env_list = [gym.make(test_env_id) for test_env_id in test_env_id_list]
        for i in range(len(test_env_list)):
            test_env_list[i].seed(config['general']['random_seed'])
    print('Done.')

    # Set the random seed manually for reproducibility.
    np.random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        if not config['general']['use_cuda']:
            logger.warning("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(config['general']['random_seed'])
    else:
        config['general']['use_cuda'] = False  # Disable CUDA.
    revisit_counting = config['general']['revisit_counting']
    replay_batch_size = config['general']['replay_batch_size']
    history_size = config['general']['history_size']
    update_from = config['general']['update_from']
    replay_memory_capacity = config['general']['replay_memory_capacity']
    replay_memory_priority_fraction = config['general']['replay_memory_priority_fraction']

    word_vocab = env.get_dictionary()
    word2id = {}
    for i, w in enumerate(word_vocab):
        word2id[w] = i

    # collect all nouns
    # verb_list, object_name_list = get_verb_and_object_name_lists(env)
    verb_list = ["go", "take"]
    object_name_list = ["east", "west", "north", "south", "coin"]
    verb_map = [word2id[w] for w in verb_list if w in word2id]
    noun_map = [word2id[w] for w in object_name_list if w in word2id]
    agent = RLAgent(config, word_vocab, verb_map, noun_map,
                    replay_memory_capacity=replay_memory_capacity, replay_memory_priority_fraction=replay_memory_priority_fraction)

    init_learning_rate = config['training']['optimizer']['learning_rate']
    exp_dir = get_experiment_dir(config)
    summary = SummaryWriter(exp_dir)

    parameters = filter(lambda p: p.requires_grad, agent.model.parameters())
    if config['training']['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif config['training']['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    log_every = 100
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    step_avg = SlidingAverage('step avg', steps=log_every)
    loss_avg = SlidingAverage('loss avg', steps=log_every)

    # save & reload checkpoint only in 0th agent
    best_avg_reward = -10000
    best_avg_step = 10000

    # step penalty
    discount_gamma = config['general']['discount_gamma']
    provide_prev_action = config['general']['provide_prev_action']

    # epsilon greedy
    epsilon_anneal_epochs = config['general']['epsilon_anneal_epochs']
    epsilon_anneal_from = config['general']['epsilon_anneal_from']
    epsilon_anneal_to = config['general']['epsilon_anneal_to']

    # counting reward
    revisit_counting_lambda_anneal_epochs = config['general']['revisit_counting_lambda_anneal_epochs']
    revisit_counting_lambda_anneal_from = config['general']['revisit_counting_lambda_anneal_from']
    revisit_counting_lambda_anneal_to = config['general']['revisit_counting_lambda_anneal_to']

    epsilon = epsilon_anneal_from
    revisit_counting_lambda = revisit_counting_lambda_anneal_from
    avg_rewards = []
    for epoch in range(config['training']['scheduling']['epoch']):

        agent.model.train()
        state = env.reset()
        obs, infos = state.feedback, state
        agent.reset(infos)
        print_command_string, print_rewards = [[]], [[]]
        print_interm_rewards = [[]]
        print_rc_rewards = [[]]

        dones = [False] * batch_size
        rewards = None
        avg_loss_in_this_game = []

        curr_observation_strings = agent.get_observation_strings(infos)
        if revisit_counting:
            agent.reset_binarized_counter(batch_size)
            revisit_counting_rewards = agent.get_binarized_count(curr_observation_strings)

        current_game_step = 0
        prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None
        input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)
        curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer
        memory_cache = [[] for _ in range(batch_size)]
        solved = [0 for _ in range(batch_size)]

        while not all(dones):
            agent.model.train()
            v_idx, n_idx, chosen_strings, curr_ras_hidden, curr_ras_cell = agent.generate_one_command(input_description, curr_ras_hidden, curr_ras_cell, epsilon=epsilon)
            state = env.step(chosen_strings[0])
            obs, rewards, dones, infos = state[0].feedback, [state[1]], state[2], state[0]
            curr_observation_strings = agent.get_observation_strings(infos)
            if provide_prev_action:
                prev_actions = chosen_strings
            # counting
            if revisit_counting:
                revisit_counting_rewards = agent.get_binarized_count(curr_observation_strings, update=True)
            else:
                revisit_counting_rewards = [0.0 for b in range(batch_size)]
            agent.revisit_counting_rewards.append(revisit_counting_rewards)
            revisit_counting_rewards = [float(format(item, ".3f")) for item in revisit_counting_rewards]

            for i in range(batch_size):
                print_command_string[i].append(chosen_strings[i])
                print_rewards[i].append(rewards)
                print_interm_rewards[i].append(infos["intermediate_reward"])
                print_rc_rewards[i].append(revisit_counting_rewards[i])
            if type(dones) is bool:
                dones = [dones] * batch_size
            agent.rewards.append(rewards)
            agent.dones.append(dones)
            agent.intermediate_rewards.append([infos["intermediate_reward"]])
            # computer rewards, and push into replay memory
            rewards_np, rewards_pt, mask_np, mask_pt, memory_mask = agent.compute_reward(revisit_counting_lambda=revisit_counting_lambda, revisit_counting=revisit_counting)

            curr_description_id_list = description_id_list
            input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)

            for b in range(batch_size):
                if memory_mask[b] == 0:
                    continue
                if dones[b] == 1 and rewards[b] == 0:
                    # last possible step
                    is_final = True
                else:
                    is_final = mask_np[b] == 0
                if rewards[b] > 0.0:
                    solved[b] = 1
                # replay memory
                memory_cache[b].append((curr_description_id_list[b], v_idx[b], n_idx[b], rewards_pt[b], mask_pt[b], dones[b], is_final, curr_observation_strings[b]))

            if current_game_step > 0 and current_game_step % config["general"]["update_per_k_game_steps"] == 0:
                policy_loss = agent.update(replay_batch_size, history_size, update_from, discount_gamma=discount_gamma)
                if policy_loss is None:
                    continue
                loss = policy_loss
                # Backpropagate
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config['training']['optimizer']['clip_grad_norm'])
                optimizer.step()  # apply gradients
                avg_loss_in_this_game.append(to_np(policy_loss))
            current_game_step += 1

        for i, mc in enumerate(memory_cache):
            for item in mc:
                if replay_memory_priority_fraction == 0.0:
                    # vanilla replay memory
                    agent.replay_memory.push(*item)
                else:
                    # prioritized replay memory
                    agent.replay_memory.push(solved[i], *item)

        agent.finish()
        avg_loss_in_this_game = np.mean(avg_loss_in_this_game)
        reward_avg.add(agent.final_rewards.mean())
        step_avg.add(agent.step_used_before_done.mean())
        loss_avg.add(avg_loss_in_this_game)
        # annealing
        if epoch < epsilon_anneal_epochs:
            epsilon -= (epsilon_anneal_from - epsilon_anneal_to) / float(epsilon_anneal_epochs)
        if epoch < revisit_counting_lambda_anneal_epochs:
            revisit_counting_lambda -= (revisit_counting_lambda_anneal_from - revisit_counting_lambda_anneal_to) / float(revisit_counting_lambda_anneal_epochs)

        # Tensorboard logging #
        # (1) Log some numbers
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            summary.add_scalar('avg_reward', reward_avg.value, epoch + 1)
            summary.add_scalar('curr_reward', agent.final_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_interm_reward', agent.final_intermediate_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_counting_reward', agent.final_counting_rewards.mean(), epoch + 1)
            summary.add_scalar('avg_step', step_avg.value, epoch + 1)
            summary.add_scalar('curr_step', agent.step_used_before_done.mean(), epoch + 1)
            summary.add_scalar('loss_avg', loss_avg.value, epoch + 1)
            summary.add_scalar('curr_loss', avg_loss_in_this_game, epoch + 1)

        msg = 'E#{:03d}, R={:.3f}/{:.3f}/IR{:.3f}/CR{:.3f}, S={:.3f}/{:.3f}, L={:.3f}/{:.3f}, epsilon={:.4f}, lambda_counting={:.4f}'
        msg = msg.format(epoch,
                         np.mean(reward_avg.value), agent.final_rewards.mean(), agent.final_intermediate_rewards.mean(), agent.final_counting_rewards.mean(),
                         np.mean(step_avg.value), agent.step_used_before_done.mean(),
                         np.mean(loss_avg.value), avg_loss_in_this_game,
                         epsilon, revisit_counting_lambda)
        avg_rewards.append(np.mean(reward_avg.value))
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            print("=========================================================")
            for prt_cmd, prt_rew, prt_int_rew, prt_rc_rew in zip(print_command_string, print_rewards, print_interm_rewards, print_rc_rewards):
                print("------------------------------")
                print(prt_cmd)
                print(prt_rew)
                print(prt_int_rew)
                print(prt_rc_rew)
        print(msg)
        # test on a different set of games
        if run_test and (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            valid_R, valid_IR, valid_S = test(config, valid_env, agent, test_batch_size, word2id)
            summary.add_scalar('valid_reward', valid_R, epoch + 1)
            summary.add_scalar('valid_interm_reward', valid_IR, epoch + 1)
            summary.add_scalar('valid_step', valid_S, epoch + 1)

            # save & reload checkpoint by best valid performance
            model_checkpoint_path = config['training']['scheduling']['model_checkpoint_path']
            if valid_R > best_avg_reward or (valid_R == best_avg_reward and valid_S < best_avg_step):
                best_avg_reward = valid_R
                best_avg_step = valid_S
                torch.save(agent.model.state_dict(), model_checkpoint_path)
                print("========= saved checkpoint =========")
                for test_id in range(len(test_env_list)):
                    R, IR, S = test(config, test_env_list[test_id], agent, test_batch_size, word2id)
                    summary.add_scalar('test_reward_' + str(test_id), R, epoch + 1)
                    summary.add_scalar('test_interm_reward_' + str(test_id), IR, epoch + 1)
                    summary.add_scalar('test_step_' + str(test_id), S, epoch + 1)
    df = pd.DataFrame({'avg_reward': avg_rewards})
    df.to_csv('results', index=False)
    print('Finished, please terminate by yourself.')

if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    args = parser.parse_args()

    if args.very_verbose:
        args.verbose = args.very_verbose
        warnings.simplefilter("default", textworld.TextworldGenerationWarning)

    # Read config from yaml file.
    config_file = pjoin(args.config_dir, 'config.yaml')
    with open(config_file) as reader:
        config = yaml.safe_load(reader)

    default_logs_path = get_experiment_dir(config)
    setup_logging(default_config_path=pjoin(args.config_dir, 'logging_config.yaml'),
                  default_level=logging.INFO, add_time_stamp=True,
                  default_logs_path=default_logs_path)
    log_git_commit(logger)

    train(config=config)
