import minigrid.wrappers
__author__ = "Feng Gu"
__email__ = "contact@fenggu.me"

"""
   isort:skip_file
"""

import inspect
import os
import re

from gymnasium.envs.registration import registry
from tqdm import tqdm
from utils import trim

readme_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "README.md",
)

LAYOUT = "env"

pattern = re.compile(r"(?<!^)(?=[A-Z])")

all_envs = list(registry.values())

filtered_envs_by_type = {}
env_names = []

# Obtain filtered list
for env_spec in tqdm(all_envs):
    # minigrid.envs:Env
    split = env_spec.entry_point.split(".")
    # ignore gymnasium.envs.env_type:Env
    env_module = split[0]
    if env_module != "minigrid":
        continue

    env_name = split[1]
    filtered_envs_by_type[env_name] = env_spec

filtered_envs = {
    env[0]: env[1]
    for env in sorted(
        filtered_envs_by_type.items(),
        key=lambda item: item[1].entry_point.split(".")[1],
    )
}

for env_name, env_spec in filtered_envs.items():
    made = env_spec.make()

    docstring = trim(made.unwrapped.__doc__)

    pascal_env_name = env_spec.id.split("-")[1]
    # remove suffix
    p = re.compile(r"([A-Z][a-z]+)*")
    name = p.search(pascal_env_name).group()

    snake_env_name = pattern.sub("_", name).lower()
    env_names.append(snake_env_name)
    title_env_name = snake_env_name.replace("_", " ").title()

    v_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "environments",
        snake_env_name + ".md",
    )

    front_matter = f"""---
AUTOGENERATED: DO NOT EDIT FILE DIRECTLY
title: {title_env_name}
---
"""
    title = f"# {title_env_name}"

    if docstring is None:
        docstring = "No information provided"
    all_text = f"""{front_matter}

{title}

{docstring}
"""
    file = open(v_path, "w+", encoding="utf-8")
    file.write(all_text)
    file.close()


# gen /environments/index.md
index_texts = """---
firstpage:
lastpage:
---

"""
env_index_toctree = """
```{toctree}
:hidden:
"""
sections = []

with open(readme_path) as f:
    readme = f.read()

    """
    sections = [description, publications, installation, basic usage, wrappers, design, included environments&etc]
    """
    sections = readme.split("<br>")
    index_texts += sections[6]
    index_texts += env_index_toctree

    for env_name in env_names:
        index_texts += env_name + "\n"

    index_texts += """\n```\n"""
    f.close()

output_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "environments",
    "index.md",
)

# output index.md
with open(output_path, "w+") as f:
    f.write(index_texts)
    f.close()

# gen /environments/design.md
design_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "environments",
    "design.md",
)

design_texts = """---
layout: "contents"
title: Design
firstpage:
---\n"""

design_texts += sections[5]

with open(design_path, "w+") as f:
    f.write(design_texts)
    f.close()


# gen /environments/wrappers.md

wrappers_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "api",
    "wrappers.md",
)

wrappers_texts = """---
title: Wrappers
lastpage:
---\n""" + sections[4] + "\n"

for name, obj in inspect.getmembers(minigrid.wrappers):
    if inspect.isclass(obj) and obj.__doc__ is not None:
        formatted_doc = ' '.join(trim(obj.__doc__).split())
        wrappers_texts += f"""## {name}
{formatted_doc}\n\n"""

with open(wrappers_path, "w+") as f:
    f.write(wrappers_texts)
    f.close()