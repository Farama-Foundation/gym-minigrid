import logging
from typing import Tuple, List

import numpy as np

from gym_minigrid.agents import Lanes

from .PedAgent import PedAgent
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction


class BlueAdlerPedAgent(PedAgent):

    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: int, # TODO convert direction to enum,
        maxSpeed: float = 4,
        speed: float = 3,
        DML: bool = False, # TODO this is not a property of the agent.
        p_exchg: float = 0.0,
        pedVmax: int = 4
        ):
    
        super().__init__(
            id=id,
            position=position,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed,
            DML=DML,
            p_exchg=p_exchg
        )

        self.pedVmax = pedVmax



    # [0] = x axis [1] = y-axis
    # 1 = down face
    # 3 = up face
    def parallel1(self, env): # TODO add type
        """_summary_

        Args:
            agents (_type_): _description_

        Returns:
            _type_: 0 to keep lane, 1 shiftLeft, 2 shiftRight
        """
        agents = env.agents
        #TODO Simulate lane change
        gaps = np.zeros((3, 4)).astype(int)
        gaps[0] = self.computeGap(agents, Lanes.currentLane, env)
        if self.canShiftLeft == True:
            gaps[1] = self.computeGap(agents, Lanes.leftLane, env)
        else:
            gaps[1] = -1, -1, -1, -10 # as backup in case gaps of 0 mess up the code, -10 is on purpose to avoid conflict with DML checking
        if self.canShiftRight == True:
            gaps[2] = self.computeGap(agents, Lanes.rightLane, env)
        else:
            gaps[2] = -1, -1, -1, -10 # as backup in case gaps of 0 mess up the code, -10 is on purpose to avoid conflict with DML checking
        # logging.info(gaps)
        
        goodLanes = []
        logging.debug('gaps', gaps)
        # DML(Dynamic Multiple Lanes)
        if self.DML and gaps[0][3] != -1: # if agentOppIndex exists, then gapOpp <= 4, prevents default of gapOpp = 0 from messing up code
            gaps[0][0] = 0 # set gap = 0
            if gaps[1][1] == 0: # check if left lane gapSame == 0
                goodLanes.append(1)
            if gaps[2][1] == 0: # check if right lane gapSame == 0
                goodLanes.append(2)
        
        # rest of algo
        if self.DML == False or len(goodLanes) == 0:
            maxGap = 0
            for i in range(3):
                maxGap = max(maxGap, gaps[i][0])
            logging.debug('maxgap', maxGap)
            for i in range(3):
                if maxGap == gaps[i][0]:
                    goodLanes.append(i)
        
        if len(goodLanes) == 1:
            lane = goodLanes[0]
        elif len(goodLanes) == 2:
            if goodLanes[0] == Lanes.currentLane:
                if np.random.random() > 0.2:
                    lane = goodLanes[0]
                else:
                    lane = goodLanes[1]
            else: #no current lane
                if np.random.random() > 0.5:
                    lane = goodLanes[0]
                else:
                    lane = goodLanes[1]
        else:
            prob = np.random.random()
            if prob > 0.2:
                lane = goodLanes[0]
            elif prob > 0.1:
                lane = goodLanes[1]
            else:
                lane = goodLanes[2]

        self.gap = gaps[lane][0]
        self.gapSame = gaps[lane][1]
        self.gapOpp = gaps[lane][2]
        self.agentOppIndex = gaps[lane][3]

        # return lane
        
        return self.convertLaneDecisionToAction(lane)

    def convertLaneDecisionToAction(self, laneDecision: int) -> LaneAction:
        if laneDecision == 0:
            return None
        if laneDecision == 1:
            return Action(self, LaneAction.LEFT)
        if laneDecision == 2:
            return Action(self, LaneAction.RIGHT)

    def parallel2(self, env): # TODO add type
        agents = env.agents
        self.speed = self.gap
        if self.gap <= 1 and self.gap == self.gapOpp: # self.gap may have to be 0 instead of 0 or 1
            if np.random.random() < self.p_exchg:
                self.speed = self.gap + 1
                agents[self.agentOppIndex].speed = self.gap + 1
            else:
                self.speed = 0
        # logging.info("Gap: " + str(self.gap) + " GapOpp: " + str(self.gapOpp))
        
        return Action(self, ForwardAction.KEEP)
        

    def computeGap(self, agents, lane, env=None):
        """
        Compute the gap (basically the possible speed ) according to the paper
        """

        laneOffset = 0
        if lane == Lanes.leftLane:
            laneOffset = -1
        elif lane == Lanes.rightLane:
            laneOffset = 1

        sameAgents, oppositeAgents = self.getSameAndOppositeAgents(agents, laneOffset=laneOffset)
       
        gap_same = self.computeSameGap(sameAgents)

        gap_opposite, agentOppIndex = self.computeOppGapAndIndex(oppositeAgents)

        # if gap > maxSpeed, we only use maxSpeed since we pick lanes in parallel1 with gap size
        # Anything > maxSpeed is irrelevant because it doesn't affect agent movement
        # doesn't affect parallel2 because maxSpeed >= 2 and parallel2 checks for == 0 or <= 1
        gap = min(self.maxSpeed, min(gap_same, gap_opposite))
        return gap, gap_same, gap_opposite, agentOppIndex
        
    def getSameAndOppositeAgents(self, agents: List[PedAgent], laneOffset=0) -> Tuple[List[PedAgent], List[PedAgent]]:

        # TODO handle all the corner cases
        opps = []
        sames = []
        for agent2 in agents:

            if self.inTheRelevantLane(agent2, laneOffset=laneOffset):
                if self.direction == agent2.direction:
                    # TODO if self is following agent2
                    sames.append(agent2)
                else:
                    # TODO if self is actually facing agent2
                    opps.append(agent2)

        return sames, opps
    
    def inTheRelevantLane(self, agent2: PedAgent, laneOffset=0) -> bool:
        return (self.position[1] + laneOffset) == agent2.position[1]

    def computeSameGap(self, sameAgents: List[PedAgent]) -> int:
        gap_same = 2 * self.pedVmax
        for agent2 in sameAgents:
            gap = abs(self.position[0] - agent2.position[0]) - 1
            assert gap >= 0

        if gap >= 0 and gap <= 2 * self.pedVmax: # gap must not be negative and less than 8
            gap_same = min(gap_same, gap)
        return gap_same

    def computeOppGapAndIndex(self, opps: List[PedAgent]) -> Tuple[int, int]:
        """
        Warning, does not check if opps have same direction agents
        """
        agentOppIndex = -1
        gap_opposite = self.pedVmax

        for i, agent2 in enumerate(opps):
            
            gap = abs(self.position[0] - agent2.position[0]) - 1
            assert gap >= 0
            if gap >= 0 and gap <= 2 * self.pedVmax: # gap must not be negative and less than 4
                if min(gap_opposite, gap // 2) == gap // 2:
                    agentOppIndex = i
                gap_opposite = min(gap_opposite, gap // 2)
        
        return gap_opposite, agentOppIndex


