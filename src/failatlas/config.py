"""Portable JSON/YAML configs with relative bases and explicit environment paths."""
import copy
import json
import os
import re
from pathlib import Path

TOKEN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

def expand(value, environ=None):
    environ = os.environ if environ is None else environ
    if isinstance(value, str):
        def replacement(match):
            name = match.group(1)
            if not environ.get(name):
                raise ValueError("unset_configuration_variable:" + name)
            return environ[name]
        return TOKEN.sub(replacement, value)
    if isinstance(value, list):
        return [expand(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, environ) for key, item in value.items()}
    return value

def merge(base, override):
    """Merge mappings recursively; explicit lists and scalar values replace bases."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def resolve(path, _parents=()):
    """Resolve _base_ paths relative to each file, without expanding env variables."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("configuration_inheritance_cycle:" + str(path))
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration_must_be_mapping:" + str(path))
    bases = value.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or any(not isinstance(x, str) or not x for x in bases):
        raise ValueError("invalid_configuration_base:" + str(path))
    result = {}
    for base in bases:
        result = merge(result, resolve(path.parent / base, (*_parents, path)))
    return merge(result, value)

def load(path):
    return expand(resolve(path))

def write_new(path, value):
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
