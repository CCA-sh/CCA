"""Tool registry — imports and registers all built-in tools."""

from cca.tools.registry import ToolRegistry
from cca.tools.file_tools import (
    ReadTool, WriteTool, EditTool, GlobTool, GrepTool, SearchInFileTool,
)
from cca.tools.bash_tool import BashTool
from cca.tools.web_tools import WebFetchTool, WebSearchTool
from cca.tools.notebook_tool import NotebookEditTool
from cca.tools.work_tools import (
    WorkCreateTool, WorkGetTool, WorkListTool, WorkUpdateTool,
)
from cca.tools.job_tools import JobCancelTool, JobOutputTool
from cca.tools.cron_tools import CronCreateTool, CronDeleteTool, CronListTool
from cca.tools.monitor_tool import MonitorTool
from cca.tools.agent_tool import AgentTool
from cca.tools.worktree_tools import EnterWorktreeTool, ExitWorktreeTool
from cca.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
from cca.tools.ask_user_tool import AskUserQuestionTool
from cca.tools.schedule_tool import ScheduleWakeupTool
from cca.tools.skill_tool import SkillTool, ToolSearchTool
from cca.tools.lsp_tool import LspTool, DETECTED_SERVERS
from cca.tools.mcp_tools import (
    McpServerManager,
    ListMcpResourcesTool,
    ReadMcpResourceTool,
    McpAuthTool,
)


def build_registry(
    config=None,
    permissions=None,
    backend=None,
    cwd: str = "",
    ask_fn=None,
    plan_fn=None,
    feature_set=None,
) -> ToolRegistry:
    """Create and populate a ToolRegistry honoring the resolved FeatureSet.

    ask_fn  — async (question, options, multi, header) -> str
              Injected into AskUserQuestionTool.
    plan_fn — async (question, options, multi, header) -> str
              Injected into ExitPlanModeTool. Falls back to ask_fn when None.
              In machine mode, emits exit_plan_mode events instead of ask_user.
    feature_set — resolved FeatureSet; None resolves defaults from config.
    """
    from cca.features import resolve_features

    fs = feature_set if feature_set is not None else resolve_features(config)
    registry = ToolRegistry()

    # Core tools — never gated
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(SearchInFileTool())
    registry.register(BashTool(allow_background=fs.feature("background_bash")))

    if fs.tool_enabled("WebFetch"):
        registry.register(WebFetchTool())
    if fs.tool_enabled("WebSearch"):
        search_config = {}
        if config:
            search_config = {
                "provider": config.search_provider,
                "api_key": config.search_api_key,
            }
        registry.register(WebSearchTool(**search_config))

    if fs.tool_enabled("NotebookEdit"):
        registry.register(NotebookEditTool())

    for name, cls in (
        ("WorkCreate", WorkCreateTool), ("WorkGet", WorkGetTool),
        ("WorkList", WorkListTool), ("WorkUpdate", WorkUpdateTool),
        ("JobCancel", JobCancelTool), ("JobOutput", JobOutputTool),
        ("CronCreate", CronCreateTool), ("CronDelete", CronDeleteTool),
        ("CronList", CronListTool), ("Monitor", MonitorTool),
        ("EnterWorktree", EnterWorktreeTool), ("ExitWorktree", ExitWorktreeTool),
        ("ScheduleWakeup", ScheduleWakeupTool),
    ):
        if fs.tool_enabled(name):
            registry.register(cls())

    # Sub-agents (needs backend reference)
    if fs.tool_enabled("Agent"):
        registry.register(AgentTool(
            backend=backend,
            config=config,
            feature_set=fs,
            cwd=cwd,
            permissions=permissions,
        ))

    # Plan mode — plan_fn falls back to ask_fn if not provided separately
    if fs.feature("plan_mode"):
        registry.register(EnterPlanModeTool())
        registry.register(ExitPlanModeTool(ask_fn=plan_fn if plan_fn is not None else ask_fn))

    if fs.tool_enabled("AskUserQuestion"):
        registry.register(AskUserQuestionTool(ask_fn=ask_fn))

    # Skills — still additionally gated by the skills_enabled config flag
    skills_on = (config is not None and getattr(config, "skills_enabled", False)) and fs.tool_enabled("Skill")
    if skills_on:
        registry.register(SkillTool())
    if fs.tool_enabled("ToolSearch"):
        registry.register(ToolSearchTool(registry=registry))

    # LSP — only registered if at least one language server is found in PATH
    if DETECTED_SERVERS and fs.tool_enabled("LSP"):
        registry.register(LspTool(cwd=cwd))

    # MCP — a tool *source*; features.mcp gates the manager and static tools
    if fs.feature("mcp"):
        mcp_manager = McpServerManager(cwd=cwd)
        registry.register(ListMcpResourcesTool(mcp_manager))
        registry.register(ReadMcpResourceTool(mcp_manager))
        registry.register(McpAuthTool())
        registry._mcp_manager = mcp_manager  # type: ignore[attr-defined]

    return registry
