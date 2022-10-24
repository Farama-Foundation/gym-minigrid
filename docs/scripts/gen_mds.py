__author__ = "Feng Gu"
__email__ = "contact@fenggu.me"

"""
   isort:skip_file
"""

import os
import re

from gymnasium.envs.registration import registry
from tqdm import tqdm
from utils import trim
from itertools import chain

readme_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "README.md",
)

LAYOUT = "env"

pattern = re.compile(r"(?<!^)(?=[A-Z])")

all_envs = list(registry.values())

filtered_envs_by_type = {}
env_names = []
babyai_envs = {}

# Obtain filtered list
for env_spec in tqdm(all_envs):
    # minigrid.envs:Env
    split = env_spec.entry_point.split(".")
    # ignore gymnasium.envs.env_type:Env
    env_module = split[0]

    if len(split) > 2 and "babyai" in split[2]:
        curr_babyai_env = split[2]
        babyai_env_name = curr_babyai_env.split(":")[1]
        babyai_envs[babyai_env_name] = env_spec
    elif env_module == "minigrid":
        env_name = split[1]
        filtered_envs_by_type[env_name] = env_spec
    # if env_module != "minigrid":
    else:
        continue


filtered_envs = {
    env[0]: env[1]
    for env in sorted(
        filtered_envs_by_type.items(),
        key=lambda item: item[1].entry_point.split(".")[1],
    )
}

filtered_babyai_envs = {
    env[0]: env[1]
    for env in sorted(
        babyai_envs.items(),
        key=lambda item: item[1].entry_point.split(".")[1],
    )
}

for env_name, env_spec in chain(filtered_envs.items(), filtered_babyai_envs.items()):
    made = env_spec.make()

    docstring = trim(made.unwrapped.__doc__)

    # minigrid.envs:Env or minigrid.envs.babyai:Env
    split = env_spec.entry_point.split(".")
    # ignore minigrid.envs.env_type:Env
    env_module = split[0]
    env_name = split[-1].split(":")[-1]
    env_type = env_module if len(split) == 2 else split[-1].split(":")[0]

    path_name = ""
    os.makedirs(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "environments",
        env_type),
        exist_ok=True)

    v_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "environments",
        env_type,
        f"{env_name}.md",
    )

    front_matter = f"""---
autogenerated:
title: {env_name}
---
"""
    title = f"# {env_name}"
    gif = "```{figure} " + f"""/_static/videos/{env_type}/{env_name}.gif
:alt: {env_name}
:width: 200px
```
"""

    if docstring is None:
        docstring = "No information provided"
    all_text = f"""{front_matter}

{title}

{gif}

{docstring}
"""
    file = open(v_path, "w+", encoding="utf-8")
    file.write(all_text)
    file.close()
