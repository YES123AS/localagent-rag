"""Deterministic V5 admission and bounded task decomposition."""

from __future__ import annotations

from src.agent.planner import COMBINATION_MARKERS, LOCAL_MARKERS, WEB_MARKERS

from .models import AgentRole, MultiAgentPlan, ResearchTask


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def should_use_multi_agent(question: str) -> bool:
    """Use multiple agents only when local and current web evidence must be combined."""
    return (
        _contains(question, LOCAL_MARKERS)
        and _contains(question, WEB_MARKERS)
        and _contains(question, COMBINATION_MARKERS)
    )


class Supervisor:
    def __init__(self, *, max_agents: int = 4, max_subtasks: int = 4):
        self.max_agents = max(1, min(int(max_agents), 4))
        self.max_subtasks = max(1, min(int(max_subtasks), 8))

    def needs_multi_agent(self, goal: str) -> bool:
        return should_use_multi_agent(goal)

    def decompose(self, goal: str, *, research_round: int = 0) -> MultiAgentPlan:
        if not self.needs_multi_agent(goal):
            raise ValueError("The goal does not require multi-agent research")
        tasks = [
            ResearchTask(
                title="Local document research",
                query=goal,
                assigned_agent=AgentRole.LOCAL_RESEARCH,
                research_round=research_round,
            ),
            ResearchTask(
                title="Current web research",
                query=goal,
                assigned_agent=AgentRole.WEB_RESEARCH,
                research_round=research_round,
            ),
        ]
        return MultiAgentPlan(goal=goal, tasks=tasks[: min(self.max_agents, self.max_subtasks)])

    def supplemental_tasks(
        self,
        goal: str,
        missing_topics: list[str],
        *,
        research_round: int,
        remaining: int,
    ) -> list[ResearchTask]:
        tasks: list[ResearchTask] = []
        for topic in missing_topics[: max(0, remaining)]:
            role = (
                AgentRole.LOCAL_RESEARCH
                if "local" in topic.lower() or "本地" in topic
                else AgentRole.WEB_RESEARCH
            )
            tasks.append(
                ResearchTask(
                    title=f"Supplemental research: {topic}"[:300],
                    query=f"{goal}\n补充验证主题：{topic}",
                    assigned_agent=role,
                    research_round=research_round,
                )
            )
        return tasks
