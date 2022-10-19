from gym_minigrid.envs.babaisyou.core.flexible_world_object import FBall, FWall
from gym_minigrid.envs.babaisyou.core.rule_block import RuleObject, RuleIs, RuleProperty
from gym_minigrid.minigrid import Grid, MissionSpace, MiniGridEnv


class ChangeRuleEnv(MiniGridEnv):
    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: ""
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)
        self.put_obj(FBall(), width - 2, height - 2)

        self.grid.horz_wall(1, height-4, length=width-2, obj_type=FWall)
        # self.put_obj(RuleBlock("can_overlap", "fwall"), width-2, 1)

        self.put_obj(RuleObject('fwall'), 2, 2)
        # self.put_obj(RuleWall('fwall'), 2, 2)
        self.put_obj(RuleIs(), 3, 2)
        # self.put_obj(RuleProperty('can_overlap'), 5, 2)
        self.put_obj(RuleProperty('is_block'), 4, 2)

        self.put_obj(RuleProperty('is_goal'), 5, height-3)
        # self.put_obj(RuleObject('fwall'), 3, height-3)

        self.put_obj(RuleObject('fball'), 2, height-3)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
