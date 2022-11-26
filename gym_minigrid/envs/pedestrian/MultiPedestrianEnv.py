from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions, PedAgent
from gym_minigrid.envs.pedestrian.PedGrid import PedGrid
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.ForwardAction import ForwardAction
# from pyee.base import EventEmitter

class MultiPedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        agents: List[Agent]=None,
        width=8,
        height=8,
    ):
        if agents is None:
            self.agents = []
        else:
            self.agents = agents

        super().__init__(
            width=width,
            height=height,
            max_steps=1000, #4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        self.stepParallel1 = []
        self.stepParallel2 = []
        self.stepParallel3 = []
        self.stepParallel4 = []

        self._actionHandlers = {
            LaneAction: self.executeLaneAction,
            ForwardAction: self.executeForwardAction,
        }

    pass

    #region agent management
    def addAgents(self, agents: List[Agent]):
        for agent in agents:
            self.addAgent(agent)
            
    def addAgent(self, agent: Agent):
        self.agents.append(agent)

        ## attach event handlers
        # TODO: this subscription must not be done in side the evironment as only the research knows about its decision and action phases.
        self.subscribe("stepParallel1", agent.parallel1) # TODO event name should be enums
        self.subscribe("stepParallel2", agent.parallel2) # TODO event name should be enums
        
    def getNumAgents(self):
        return len(self.agents)

    def resetAgents(self):
        for agent in self.agents:
            agent.reset()

    def getDensity(self):
        cells = (self.width - 1) * (self.height - 1)
        agents = len(self.agents)
        return agents/cells

    def forwardAgent(self, agent: Agent):
        # TODO DONE
        
        # Get the position in front of the agent
        assert agent.direction >= 0 and agent.direction < 4
        fwd_pos = agent.position + agent.speed * DIR_TO_VEC[agent.direction]
        if fwd_pos[0] < 0 or fwd_pos[0] >= self.width:
            self.agents.remove(agent)
            return
        # Terry - implemented speed ^ by multiplying speed with direction unit vector

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward if no overlap
        if fwd_cell == None or fwd_cell.can_overlap():
                agent.position = fwd_pos
        # Terry - Once we get validateAgentPositions working, we won't need to check
        pass

    # Terry - move left and right functions are below
    def shiftLeft(self, agent: Agent):
        assert agent.direction >= 0 and agent.direction < 4
        #Terry - uses the direction to left of agent to find vector to move left
        # left_dir = agent.direction - 1
        # if left_dir < 0:
        #     left_dir += 4
        # left_pos = agent.position + DIR_TO_VEC[left_dir]

        # agent.position[0] = left_pos
        agent.position = (agent.position[0], agent.position[1] - 1)

    def shiftRight(self, agent: Agent):
        # assert agent.direction >= 0 and agent.direction < 4
        # #Terry - uses the direction to left of agent to find vector to move left
        # right_dir = (agent.direction + 1) % 4
        # right_pos = agent.position + DIR_TO_VEC[right_dir]
        
        # agent.position = right_pos
        agent.position = (agent.position[0], agent.position[1] + 1)
        
    #endregion

    #region sidewalk

    def genSidewalks(self):

        # TODO turn this into 2 side walks. DONE

        # Terry - added goals to the left side
        # not sure if we are making the sidewalks go horizontally or vertically
        for i in range(1, self.height-1):
            self.put_obj(Goal(), 1, i)
            self.put_obj(Goal(), self.width - 2, i)
        pass

    #endregion

    #region gym env overrides

    def validateAgentPositions(self):
        # TODO iterate over our agents and make sure that they can be placed there
        
        # # Check that the agent doesn't overlap with an object
        # start_cell = self.grid.get(*self.agent_pos)
        # assert start_cell is None or start_cell.can_overlap()
        pass


    def reset(self):
        
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        self.resetAgents()
        self.validateAgentPositions()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs



    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = PedGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.genSidewalks()
        self.mission = "switch sidewalks"


    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        img = self.grid.render(
            tile_size,
            self.agents,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None
            # highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def eliminateConflict(self):
        for agent in self.agents:
            if agent.position[1] == 1:
                agent.canShiftLeft = False
            if agent.position[1] == self.height - 2:
                agent.canShiftRight = False
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 == agent2 or agent1.position[0] != agent2.position[0]:
                    continue

                if (agent1.position[1] - agent2.position[1]) == 1:
                    # they are adjacent
                    agent1.canShiftLeft = False
                    agent2.canShiftRight = False
                elif (agent1.position[1] - agent2.position[1]) == 2 and agent1.canShiftLeft == True and agent2.canShiftRight == True:  
                    # they have one cell between them
                    if np.random.random() > 0.5:
                        agent1.canShiftLeft = False 
                    else: 
                        agent2.canShiftRight = False

        # for agent in self.agents:
        #     if agent.position[0] < 1 or agent.position[0] == self.width - 1:
        #         self.agents.remove(agent)
        #         print('removed')

    # One step after parallel1 and parallel2
    # Save plans from parallel1 and parallel2 before actually executing it

    def subscribe(self, eventName, handler):
        if eventName == "stepParallel1": # TODO convert to enum
            self.stepParallel1.append(handler)

        if eventName == "stepParallel2": # TODO convert to enum
            self.stepParallel2.append(handler)
    
    def emitEventAndGetResponse(self, eventName) -> List[Action]:

        print(f"executing {eventName}")
        # print(self.stepParallel1)
        # print(self.stepParallel2)
        if eventName == "stepParallel1": # TODO convert to enum
            return [handler(self) for handler in self.stepParallel1] # TODO fix for multiple actions by a single handler.

        if eventName == "stepParallel2": # TODO convert to enum
            return [handler(self) for handler in self.stepParallel2]


    

    def step(self, action=None):
        """This step is tightly coupled with the research, how can we decouple it?

        Args:
            action (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.step_count += 1

        reward = 0
        done = False

        self.eliminateConflict()

        actions = self.emitEventAndGetResponse("stepParallel1")
        self.executeActions(actions)

        actions = self.emitEventAndGetResponse("stepParallel2")
        self.executeActions(actions)


        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        
        return obs, reward, done, {}

    def executeActions(self, actions: List[Action]):
        if len(actions) == 0:
            return
        # TODO

        for action in actions:
            if action is not None:
                self._actionHandlers[action.action.__class__](action)
            # self.executeAction(action)
        pass

    def executeLaneAction(self, action: Action):
        if action is None:
            return 
        if action == 1:
            self.shiftLeft(self.agents[i])
        elif action == 2:
            self.shiftRight(self.agents[i])
        pass

    def executeForwardAction(self, action: Action):
        if action is None:
            return 


            
        agent = action.agent

        print(f"forwarding agent {agent.id}")

        self.forwardAgent(agent)
        agent.canShiftLeft = True
        agent.canShiftRight = True
        pass

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # TODO
        # 1. There is no observation
        obs = {
        }

        return obs

    #endregion

    

class MultiPedestrianEnv20x80(MultiPedestrianEnv):
    def __init__(self):
        width = 100
        height = 60
        super().__init__(
            width=width,
            height=height,
            agents=None
        )

register(
    id='MultiPedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:MultiPedestrianEnv20x80'
)