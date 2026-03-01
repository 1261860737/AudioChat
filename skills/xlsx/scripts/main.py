"""Skill entrypoint stub.

This provides a minimal execute/run/main API so SkillRegistry can load the skill.
"""
from __future__ import annotations


def main(**_kwargs):
    """Minimal entrypoint placeholder for the xlsx skill."""
    message = "TODO: xlsx skill entrypoint invoked (no-op)"
    print(message)
    return message


def execute(**kwargs):
    return main(**kwargs)


def run(**kwargs):
    return main(**kwargs)
