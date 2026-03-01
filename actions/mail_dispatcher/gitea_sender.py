from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GiteaIssueResult:
    number: int
    url: str


_ENV_LOADED = False


def _repo_issues_web_url(base: str, owner: str, repo: str) -> str:
    """仓库 issues 网页地址（可直接在浏览器打开）。"""
    return f"{base}/{owner}/{repo}/issues"


def _normalize_issue_url(url: str, base: str, owner: str, repo: str, issue_number: int) -> str:
    """把 Gitea 返回的 URL 规范成可打开的网页地址。"""
    if not url:
        return f"{_repo_issues_web_url(base, owner, repo)}/{issue_number}"

    api_prefix = f"{base}/api/v1/repos/{owner}/{repo}/issues"
    if url.startswith(api_prefix):
        suffix = url[len(api_prefix):]
        return f"{_repo_issues_web_url(base, owner, repo)}{suffix}"
    return url


def _maybe_load_env_file() -> None:
    """在缺少环境变量时，尝试从 `actions/.env` 加载（按当前写法 `KEY=VALUE`）。

    说明：
    - 本仓库的测试脚本 `actions/mail_dispatcher/test.py` 直接依赖环境变量。
    - 为了降低本地测试成本，这里在读取 env 失败时，会懒加载 `actions/.env`。
    - 仅按当前约定解析：每行 `KEY=VALUE`（允许可选引号与注释行）。
    - 不会覆盖已存在的 `os.environ` 值（使用 setdefault 语义）。
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    actions_dir = Path(__file__).resolve().parents[1]  # .../actions
    env_path = actions_dir / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if "#" in value and not (value.startswith("'") or value.startswith('"')):
            value = value.split("#", 1)[0].rstrip()
        value = value.strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, value)


def _env(name: str) -> str:
    """读取必需环境变量；缺失时抛出可读的 RuntimeError。"""
    v = os.environ.get(name)
    if not v:
        _maybe_load_env_file()
        v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def create_issue(
    title: str,
    body: str,
    labels: Optional[list[str]] = None,
) -> GiteaIssueResult:
    """
    Create a Gitea issue in a target repo.

    Required env:
      - GITEA_BASE_URL  e.g. https://git.dtx.openpie.com
      - GITEA_TOKEN     personal access token
      - GITEA_OWNER     org/user
      - GITEA_REPO      repo name

    中文说明：
    - 该函数是“真发送”（会创建 issue），适合在生成内容并人工确认后再调用。
    - `labels` 在不同 Gitea 部署/版本里可能要求 label id；这里先按“可接受名字”的方式传。
    - 若你没有手动 `source actions/.env`，本模块会在缺少 env 时尝试懒加载该文件。
    """
    base = _env("GITEA_BASE_URL").rstrip("/")
    token = _env("GITEA_TOKEN")
    owner = _env("GITEA_OWNER")
    repo = _env("GITEA_REPO")

    # 期望的 API 路径通常是 `/api/v1/...`（不同部署可能略有差异）
    url = f"{base}/api/v1/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    payload: dict = {"title": title, "body": body}

    # labels (optional): Gitea API expects label IDs in some versions,
    # but some deployments accept label names. If your server requires IDs,
    # we can add a helper to resolve name->id.
    if labels:
        payload["labels"] = labels

    # 本地调试：允许只打印不发送，避免误创建 issue
    if os.environ.get("GITEA_DRY_RUN", "0") == "1":
        print("[GITEA_DRY_RUN] POST", url)
        print("[GITEA_DRY_RUN] payload:", payload)
        return GiteaIssueResult(number=0, url=_repo_issues_web_url(base, owner, repo))

    try:
        import requests  # heavy dependency, import lazily
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: requests. Please `pip install requests`.") from exc

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    number = int(data["number"])
    issue_url = _normalize_issue_url(
        str(data.get("html_url") or data.get("url") or ""),
        base=base,
        owner=owner,
        repo=repo,
        issue_number=number,
    )
    return GiteaIssueResult(number=number, url=issue_url)
