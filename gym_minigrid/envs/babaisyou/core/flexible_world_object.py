from gym_minigrid.minigrid import WorldObj, COLORS, OBJECT_TO_IDX
from gym_minigrid.rendering import fill_coords, point_in_circle, point_in_rect

from .rule_block import properties


def add_object_types(object_types):
    last_idx = len(OBJECT_TO_IDX)-1
    OBJECT_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(object_types)})


add_object_types(['fwall', 'fball', 'rule', 'rule_object', 'rule_is', 'rule_property'])


def make_prop_fn(prop, typ):
    """
    Make a method that retrieve the property of a FlexibleWorldObj in the ruleset
    """
    def get_prop(self):
        ruleset = self.get_ruleset()
        return ruleset[prop].get(typ, False)
    return get_prop


class FlexibleWorldObj(WorldObj):
    def __init__(self, type, color):
        super().__init__(type, color)

        for prop in properties:
            setattr(self.__class__, prop, make_prop_fn(prop, self.type))

    # TODO: might be better to use a Ruleset object
    def set_ruleset_getter(self, get_ruleset):
        self._get_ruleset = get_ruleset

    def get_ruleset(self):
        return self._get_ruleset()

    # compatibility with WorldObj
    def can_overlap(self):
        return not self.is_block()


class FWall(FlexibleWorldObj):
    def __init__(self, color="grey"):
        super().__init__("fwall", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), COLORS[self.color])


class FBall(FlexibleWorldObj):
    def __init__(self, color="green"):
        super().__init__("fball", color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
