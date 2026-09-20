"""两两决策台 —— FastAPI 后端。

逐对比较候选项，按 Schulze 最强路径法实时排名。
进度持久化在 SQLite 中，刷新页面不丢失。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from itertools import combinations
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .conflict import minimum_tie_resolution, shortest_cycle
from .paths import pair_explanation
from .priority import recommend_next_pair
from .schulze import rank_groups, strongest_paths

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = os.environ.get("DECISION_DB", str(BASE_DIR / "data" / "decision.db"))

MIN_CANDIDATES = 3
MAX_CANDIDATES = 12
MAX_NAME_LEN = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    candidates TEXT NOT NULL,
    -- 已停用候选项名单（JSON 数组，按候选录入顺序排列）；
    -- candidates 始终保留全部候选与最初录入顺序，停用不删除任何判断记录
    disabled TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('a', 'b', 'tie'))
);
CREATE INDEX IF NOT EXISTS idx_judgments_session ON judgments(session_id);
-- 情景沙盘草稿：与 judgments 完全隔离的暂存覆盖层。
-- changes 为 JSON 数组 [{a,b,result}]（组合按候选录入序排列，result 相对该 a），
-- base_version 为创建/最后一次编辑草稿时基础判断的指纹，应用前据此检测冲突。
CREATE TABLE IF NOT EXISTS sandbox_drafts (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    changes TEXT NOT NULL,
    base_version TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # 旧库迁移（Docker 卷中可能已存在建表时没有 disabled 列的库）
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
        if "disabled" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN disabled TEXT NOT NULL DEFAULT '[]'"
            )


init_db()  # 模块加载即建库建表（幂等），lifespan 中再执行一次以确保任意部署方式下可用


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="两两决策台", lifespan=lifespan)


# ---------- 请求模型 ----------

class SessionCreate(BaseModel):
    candidates: list[str] = Field(min_length=MIN_CANDIDATES, max_length=MAX_CANDIDATES)


class JudgmentCreate(BaseModel):
    a: str
    b: str
    result: Literal["a", "b", "tie"]


class SuggestionApply(BaseModel):
    # 可选：客户端回传待改判组合用于一致性校验；缺省时服务器直接采用当前建议
    pairs: list[dict[str, str]] | None = None


class CandidateToggle(BaseModel):
    name: str


class SandboxChange(BaseModel):
    a: str
    b: str
    result: Literal["a", "b", "tie"]


class SandboxDraft(BaseModel):
    # 暂存覆盖层：每条 {a, b, result}，a/b 为启用候选，result 相对组合的录入序 a
    changes: list[SandboxChange]


# ---------- 情景沙盘 ----------

def _base_version(
    all_candidates: list[str], disabled_set: set[str], all_history: list[dict]
) -> str:
    """基础判断指纹：候选名单、停用名单与全部判断记录（含停用候选）的哈希。

    沙盘应用全部改动前重新计算并与草稿创建时的指纹比对；基础判断发生任何
    变化（新增 / 撤销 / 复核 / 批量改判 / 重置 / 停用恢复改变计算口径）都会
    使指纹不同，应用据此拒绝并保留草稿。
    """
    payload = json.dumps(
        {
            "candidates": all_candidates,
            "disabled": [c for c in all_candidates if c in disabled_set],
            "judgments": [
                [h["a"], h["b"], h["result"]] for h in all_history
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sandbox_merged_history(
    history: list[dict], overlay: list[dict]
) -> list[dict]:
    """把暂存覆盖层叠加到启用候选的基础判断上，合成沙盘虚拟判断序列。

    - 对未判组合的暂存：按覆盖层顺序补写到原判断序列末尾；
    - 对已判组合的暂存：原位替换结果，保持其在基础判断记录中的位置；
    - 覆盖层只包含启用候选组合（PUT 时已校验），与 history 口径一致。
    """
    index: dict[tuple[str, str], int] = {}
    for i, h in enumerate(history):
        index[(h["a"], h["b"])] = i
    merged = [dict(h) for h in history]
    for ch in overlay:
        key = (ch["a"], ch["b"])
        if key in index:
            merged[index[key]] = {"a": ch["a"], "b": ch["b"], "result": ch["result"]}
        else:
            index[key] = len(merged)
            merged.append({"a": ch["a"], "b": ch["b"], "result": ch["result"]})
    return merged


def _rank_index(groups: list[list[str]]) -> dict[str, int]:
    """候选项 → 名次（并列同名次；竞争排名，并列后跳号）。"""
    rank_of: dict[str, int] = {}
    rank = 1
    for group in groups:
        for name in group:
            rank_of[name] = rank
        rank += len(group)
    return rank_of


def _rank_moves(
    base_groups: list[list[str]], sandbox_groups: list[list[str]]
) -> list[dict]:
    """每个候选项在沙盘结果中相对原结果的名次升降（正 = 上升，负 = 下降）。"""
    base_rank = _rank_index(base_groups)
    sandbox_rank = _rank_index(sandbox_groups)
    return [
        {
            "name": name,
            "base_rank": base_rank[name],
            "sandbox_rank": sandbox_rank[name],
            "delta": base_rank[name] - sandbox_rank[name],
        }
        for name in base_rank
    ]


def _normalize_sandbox_changes(
    candidates: list[str],
    disabled_names: set[str],
    changes: list[SandboxChange],
) -> list[dict]:
    """校验并规范化暂存改动：组合按候选录入序排列，result 相对该序解释。"""
    order = {name: i for i, name in enumerate(candidates)}
    all_names = set(candidates) | disabled_names
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for ch in changes:
        if ch.a == ch.b or ch.a not in all_names or ch.b not in all_names:
            raise HTTPException(422, "无效的比较组合（候选项不存在）")
        if ch.a not in order or ch.b not in order:
            raise HTTPException(409, "该组合含已停用候选，请先恢复后再沙盘推演")
        a, b = sorted((ch.a, ch.b), key=order.__getitem__)
        if (a, b) in seen:
            raise HTTPException(422, f"组合「{a} vs {b}」在暂存列表中重复")
        seen.add((a, b))
        if ch.result == "tie":
            result = "tie"
        else:
            winner = ch.a if ch.result == "a" else ch.b
            result = "a" if winner == a else "b"
        normalized.append({"a": a, "b": b, "result": result})
    return normalized


def _sandbox_payload(
    conn: sqlite3.Connection,
    sid: str,
    all_candidates: list[str],
    disabled_set: set[str],
    all_history: list[dict],
) -> dict:
    """组装沙盘 GET 响应：草稿、基础指纹与「原判断 + 覆盖层」重算结果。"""
    candidates = [name for name in all_candidates if name not in disabled_set]
    enabled = set(candidates)
    history = [
        h for h in all_history if h["a"] in enabled and h["b"] in enabled
    ]
    base_version = _base_version(all_candidates, disabled_set, all_history)
    base_state = _build_state(sid, all_candidates, disabled_set, all_history)

    row = conn.execute(
        "SELECT changes, base_version FROM sandbox_drafts WHERE session_id = ?",
        (sid,),
    ).fetchone()

    draft = None
    sandbox_state = base_state
    moves = _rank_moves(base_state["ranking"], base_state["ranking"])
    stale = False
    if row is not None:
        raw_changes: list[dict] = json.loads(row["changes"])
        valid: list[dict] = []
        changes_out: list[dict] = []
        judged = {(h["a"], h["b"]): h["result"] for h in history}
        order = {name: i for i, name in enumerate(candidates)}
        for ch in raw_changes:
            a, b, result = ch["a"], ch["b"], ch["result"]
            # 防御性过滤：草稿创建后候选被停用等情况下，失效条目保留在草稿中但不参与计算
            if a not in order or b not in order:
                changes_out.append(
                    {"a": a, "b": b, "result": result, "original": None,
                     "effective": False}
                )
                continue
            valid.append(ch)
            changes_out.append(
                {
                    "a": a,
                    "b": b,
                    "result": result,
                    "original": judged.get((a, b)),
                    "effective": judged.get((a, b)) != result,
                }
            )
        sandbox_history = _sandbox_merged_history(history, valid)
        sandbox_state = _build_state(
            sid, all_candidates, disabled_set, sandbox_history
        )
        moves = _rank_moves(base_state["ranking"], sandbox_state["ranking"])
        stale = row["base_version"] != base_version
        draft = {
            "changes": changes_out,
            "base_version": row["base_version"],
            "stale": stale,
        }

    return {
        "session_id": sid,
        "base_version": base_version,
        "draft": draft,
        "state": sandbox_state,
        "base_state": base_state,
        "rank_moves": moves,
    }


def _load_sandbox_payload(conn: sqlite3.Connection, sid: str) -> dict:
    all_candidates, disabled_set, all_history = _load_rows(conn, sid)
    return _sandbox_payload(
        conn, sid, all_candidates, disabled_set, all_history
    )


# ---------- 状态组装 ----------

def _load_rows(
    conn: sqlite3.Connection, session_id: str
) -> tuple[list[str], set[str], list[dict]]:
    """读取会话的全部候选（最初录入顺序）、停用名单与全部判断记录。

    判断记录**始终全量返回**——停用候选不删除其任何判断，撤销仍按记录
    id 顺序工作。
    """
    row = conn.execute(
        "SELECT candidates, disabled FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在或已被删除")
    all_candidates: list[str] = json.loads(row["candidates"])
    disabled: set[str] = set(json.loads(row["disabled"] or "[]"))
    # 防御性规整：停用名单只保留仍存在的候选
    disabled &= set(all_candidates)

    all_history: list[dict] = []
    for r in conn.execute(
        "SELECT a, b, result FROM judgments WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        all_history.append({"a": r["a"], "b": r["b"], "result": r["result"]})
    return all_candidates, disabled, all_history


def _build_state(
    session_id: str,
    all_candidates: list[str],
    disabled_set: set[str],
    all_history: list[dict],
) -> dict:
    """根据会话的候选、停用名单与判断记录组装完整状态。

    沙盘以「原判断 + 暂存覆盖层」合成虚拟 history 后复用本函数，
    因此排名、矩阵、循环冲突、最少改判、进度与下一对推荐的计算口径完全一致。
    """
    # 启用候选：按**最初录入顺序**过滤；所有计算只使用启用候选
    candidates = [name for name in all_candidates if name not in disabled_set]
    enabled = set(candidates)
    # 与停用候选有关的判断记录保留在库中，但不参与本轮任何计算
    history = [
        h for h in all_history if h["a"] in enabled and h["b"] in enabled
    ]

    judged: dict[tuple[str, str], str] = {}
    for h in history:
        judged[(h["a"], h["b"])] = h["result"]

    pairs, pending = [], []
    for a, b in combinations(candidates, 2):
        result = judged.get((a, b))
        pairs.append({"a": a, "b": b, "result": result})
        if result is None:
            pending.append({"a": a, "b": b})

    p = strongest_paths(candidates, history)
    groups = rank_groups(candidates, p)

    # 胜负矩阵：行 vs 列
    n = len(candidates)
    matrix: list[list[str]] = []
    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("self")
                continue
            a, b = (candidates[i], candidates[j]) if i < j else (candidates[j], candidates[i])
            result = judged.get((a, b))
            if result is None:
                cells.append("pending")
            elif result == "tie":
                cells.append("tie")
            elif (result == "a") == (i < j):
                cells.append("win")
            else:
                cells.append("lose")
        matrix.append(cells)

    # 循环冲突：严格胜负边中的最短有向环（平局与未比较不成边）
    conflict = None
    cycle = shortest_cycle(candidates, history)
    if cycle is not None:
        order = {name: i for i, name in enumerate(candidates)}
        edges = []
        for i, winner in enumerate(cycle):
            loser = cycle[(i + 1) % len(cycle)]
            a, b = (winner, loser) if order[winner] < order[loser] else (loser, winner)
            edges.append({"a": a, "b": b, "result": "a" if winner == a else "b"})
        conflict = {"cycle": cycle, "edges": edges}

    # 最少改判方案：把最少数量的严格胜负组合改为平局即可消除全部有向环。
    # 无冲突时不生成建议；平局与未比较组合不参与计算。
    suggestion = None
    if conflict is not None:
        change_pairs = minimum_tie_resolution(candidates, history)
        change_set = {(a, b) for a, b in change_pairs}
        affected = []
        preview_history = []
        for h in history:
            a, b, result = h["a"], h["b"], h["result"]
            if (a, b) in change_set:
                affected.append(
                    {
                        "a": a,
                        "b": b,
                        "from": result,  # 严格胜负（"a"/"b"，相对存储顺序的 a）
                        "to": "tie",
                    }
                )
                preview_history.append({"a": a, "b": b, "result": "tie"})
            else:
                preview_history.append({"a": a, "b": b, "result": result})
        preview_groups = rank_groups(
            candidates, strongest_paths(candidates, preview_history)
        )
        suggestion = {
            # 组合按判断记录顺序排列，stable 选择结果对前端确定可见
            "pairs": [{"a": a, "b": b} for a, b in change_pairs],
            "change_count": len(change_pairs),
            "affected": affected,
            "preview_ranking": preview_groups,
        }

    return {
        "session_id": session_id,
        # candidates 仅含启用候选（按最初录入顺序）；disabled 为停用名单（同序）
        "candidates": candidates,
        "disabled": [name for name in all_candidates if name in disabled_set],
        "progress": {"done": len(history), "total": len(pairs)},
        "current_pair": pending[0] if pending else None,
        "pending": pending,
        "ranking": groups,
        "matrix": matrix,
        "conflict": conflict,
        "suggestion": suggestion,
        # 下一对优先级推荐：纯内存模拟，不写库、不影响进度
        "next_pair": recommend_next_pair(candidates, history, pending),
        # 撤销按原始判断记录顺序工作（停用候选的记录仍可被撤销）
        "can_undo": bool(all_history),
    }


def load_state(conn: sqlite3.Connection, session_id: str) -> dict:
    """从库中读取会话并组装状态（沙盘之外的所有接口使用）。"""
    all_candidates, disabled_set, all_history = _load_rows(conn, session_id)
    return _build_state(session_id, all_candidates, disabled_set, all_history)


# ---------- API ----------

@app.post("/api/sessions", status_code=201)
def create_session(payload: SessionCreate):
    names, seen = [], set()
    for raw in payload.candidates:
        name = " ".join(raw.split())
        if not name:
            continue
        if len(name) > MAX_NAME_LEN:
            raise HTTPException(422, f"候选项名称过长（>{MAX_NAME_LEN} 字符）：{name[:20]}…")
        if name in seen:
            raise HTTPException(422, f"候选项重复：{name}")
        seen.add(name)
        names.append(name)
    if not (MIN_CANDIDATES <= len(names) <= MAX_CANDIDATES):
        raise HTTPException(422, f"有效候选项数量须为 {MIN_CANDIDATES}～{MAX_CANDIDATES} 个")

    sid = uuid.uuid4().hex[:12]
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (id, candidates) VALUES (?, ?)",
            (sid, json.dumps(names, ensure_ascii=False)),
        )
        conn.commit()
        return load_state(conn, sid)
    finally:
        conn.close()


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    conn = get_conn()
    try:
        return load_state(conn, sid)
    finally:
        conn.close()


@app.get("/api/sessions/{sid}/ranking-basis")
def ranking_basis(sid: str, a: str, b: str):
    """「排名依据」：两个候选项之间的双向最强路径与最终相对判定。

    路径在与排名相同的 Schulze 计算中重建（并在内部与 Floyd–Warshall
    强度矩阵交叉核对）：同强度路径优先边数更少，再按候选录入顺序稳定
    选择，路径不重复节点；任一方向不可达时对应路径明确标记为无有效路径。
    纯只读接口，不写库、不影响任何现有接口；面板随各写操作返回的状态刷新。
    """
    conn = get_conn()
    try:
        all_candidates, disabled_set, _ = _load_rows(conn, sid)
        candidates = [name for name in all_candidates if name not in disabled_set]
        if a == b or a not in candidates or b not in candidates:
            raise HTTPException(422, "无效的候选项组合（候选项不存在或已停用）")
        enabled = set(candidates)
        history = [
            {"a": r["a"], "b": r["b"], "result": r["result"]}
            for r in conn.execute(
                "SELECT a, b, result FROM judgments WHERE session_id = ? ORDER BY id",
                (sid,),
            )
            if r["a"] in enabled and r["b"] in enabled
        ]
        return pair_explanation(candidates, history, a, b)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/judgments", status_code=201)
def judge(sid: str, payload: JudgmentCreate):
    """记录一对组合的判断。

    默认按「当前待比较组合」依次评审；为支持「下一对优先级」推荐的一键跳转，
    也接受任意尚未比较的组合。与复核接口一致：组合统一按候选录入顺序存储，
    result 始终相对请求中先出现的 a 解释。已判定或无效组合分别返回 409/422。
    """
    conn = get_conn()
    try:
        state = load_state(conn, sid)
        order = {name: i for i, name in enumerate(state["candidates"])}
        all_names = set(state["candidates"]) | set(state["disabled"])
        if payload.a == payload.b or payload.a not in all_names or payload.b not in all_names:
            raise HTTPException(422, "无效的比较组合")
        if payload.a not in order or payload.b not in order:
            raise HTTPException(409, "该组合含已停用候选，请先恢复后再评审")
        # 统一按候选录入顺序存储；result 始终相对请求中先出现的 a 解释
        a, b = sorted((payload.a, payload.b), key=order.__getitem__)
        if not any(p["a"] == a and p["b"] == b for p in state["pending"]):
            if state["current_pair"] is None:
                raise HTTPException(409, "本轮所有组合均已比较完毕")
            raise HTTPException(409, "该组合已判定，请刷新后重试")
        if payload.result == "tie":
            result = "tie"
        else:
            winner = payload.a if payload.result == "a" else payload.b
            result = "a" if winner == a else "b"
        conn.execute(
            "INSERT INTO judgments (session_id, a, b, result) VALUES (?, ?, ?, ?)",
            (sid, a, b, result),
        )
        conn.commit()
        return load_state(conn, sid)
    finally:
        conn.close()


@app.put("/api/sessions/{sid}/judgments")
def revise(sid: str, payload: JudgmentCreate):
    """循环冲突复核：原位替换一条已判定组合的记录。

    UPDATE 不改变行 id 与记录条数，因此进度总数、已完成数与
    撤销顺序均保持不变；返回重算后的完整状态（排名、待比较、下一冲突）。
    """
    conn = get_conn()
    try:
        state = load_state(conn, sid)
        order = {name: i for i, name in enumerate(state["candidates"])}
        all_names = set(state["candidates"]) | set(state["disabled"])
        if payload.a == payload.b or payload.a not in all_names or payload.b not in all_names:
            raise HTTPException(422, "无效的比较组合")
        if payload.a not in order or payload.b not in order:
            raise HTTPException(409, "该组合含已停用候选，请先恢复后再评审")
        # 统一按候选录入顺序存储；result 始终相对请求中先出现的 a 解释
        a, b = sorted((payload.a, payload.b), key=order.__getitem__)
        if payload.result == "tie":
            result = "tie"
        else:
            winner = payload.a if payload.result == "a" else payload.b
            result = "a" if winner == a else "b"
        cur = conn.execute(
            "UPDATE judgments SET result = ? WHERE session_id = ? AND a = ? AND b = ?",
            (result, sid, a, b),
        )
        if cur.rowcount == 0:
            raise HTTPException(409, "该组合尚未判定，请通过正常流程完成首次比较")
        conn.commit()
        return load_state(conn, sid)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/resolve-conflicts")
def resolve_conflicts(sid: str, payload: SuggestionApply | None = None):
    """一次确认后批量应用「最少改判方案」：把建议组合全部改为平局。

    全部 UPDATE 与后续状态读取在**单个 SQLite 事务**中完成，任一更新未命中或
    发生异常即整体回滚。UPDATE 不新增/删除判断记录，因此判断记录 id、
    进度统计（已完成数/总数）与撤销顺序均保持不变。提交后重算排名与冲突，
    返回完整状态（成功时 conflict/suggestion 均应为空）。
    """
    conn = get_conn()
    try:
        with conn:  # 单事务：无异常提交，异常自动回滚
            state = load_state(conn, sid)
            suggestion = state["suggestion"]
            if suggestion is None:
                raise HTTPException(409, "当前不存在循环冲突，无需应用改判方案")

            expected = [(p["a"], p["b"]) for p in suggestion["pairs"]]
            if payload is not None and payload.pairs is not None:
                submitted = []
                for p in payload.pairs:
                    if set(p.keys()) != {"a", "b"}:
                        raise HTTPException(422, "组合字段须为 a、b")
                    submitted.append((p["a"], p["b"]))
                if submitted != expected:
                    raise HTTPException(
                        409, "建议方案已过期或与当前冲突不符，请刷新后重试"
                    )

            for a, b in expected:
                # 仅把严格胜负记录改为平局；平局行 rowcount 仍为 1 但语义上不该出现
                cur = conn.execute(
                    "UPDATE judgments SET result = 'tie' "
                    "WHERE session_id = ? AND a = ? AND b = ? AND result != 'tie'",
                    (sid, a, b),
                )
                if cur.rowcount != 1:
                    raise HTTPException(
                        409, f"组合「{a} vs {b}」当前不是严格胜负判定，方案已失效"
                    )
            return load_state(conn, sid)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/undo")
def undo(sid: str):
    conn = get_conn()
    try:
        if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone() is None:
            raise HTTPException(404, "会话不存在或已被删除")
        cur = conn.execute(
            "DELETE FROM judgments WHERE id = "
            "(SELECT MAX(id) FROM judgments WHERE session_id = ?)",
            (sid,),
        )
        if cur.rowcount == 0:
            raise HTTPException(409, "没有可撤销的判断")
        conn.commit()
        return load_state(conn, sid)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/reset")
def reset(sid: str):
    conn = get_conn()
    try:
        if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone() is None:
            raise HTTPException(404, "会话不存在或已被删除")
        conn.execute("DELETE FROM judgments WHERE session_id = ?", (sid,))
        conn.commit()
        return load_state(conn, sid)
    finally:
        conn.close()


def _toggle_candidate(
    conn: sqlite3.Connection, sid: str, name: str, *, disable: bool
) -> dict:
    """停用（disable=True）或恢复（disable=False）一个候选项。

    - 停用/恢复只改写 sessions.disabled 名单，**不删除任何判断记录**；
    - 恢复后候选按最初录入顺序回到评审，已有组合沿用原判断，仅缺失组合
      重新进入待比较；
    - 任何时刻至少保留 MIN_CANDIDATES 个启用候选。
    """
    all_candidates, disabled_set, _ = _load_rows(conn, sid)
    if name not in all_candidates:
        raise HTTPException(422, "候选项不存在")
    if disable and name in disabled_set:
        raise HTTPException(409, "该候选项已停用")
    if not disable and name not in disabled_set:
        raise HTTPException(409, "该候选项处于启用状态")
    if disable and len(all_candidates) - len(disabled_set) <= MIN_CANDIDATES:
        raise HTTPException(
            409, f"至少保留 {MIN_CANDIDATES} 个启用候选，无法继续停用"
        )

    if disable:
        disabled_set.add(name)
    else:
        disabled_set.discard(name)
    # 持久化时按最初录入顺序排列，结果稳定可复现
    disabled_list = [c for c in all_candidates if c in disabled_set]
    conn.execute(
        "UPDATE sessions SET disabled = ? WHERE id = ?",
        (json.dumps(disabled_list, ensure_ascii=False), sid),
    )
    conn.commit()
    return load_state(conn, sid)


@app.post("/api/sessions/{sid}/candidates/deactivate", status_code=200)
def deactivate_candidate(sid: str, payload: CandidateToggle):
    """停用一个候选项：其全部判断记录保留，只是不参与任何计算。

    停用后进度、待比较组合、Schulze 排名、矩阵、循环冲突、最少改判与
    下一对推荐均只按启用候选重算；至少保留 3 个启用候选。
    """
    conn = get_conn()
    try:
        return _toggle_candidate(conn, sid, payload.name, disable=True)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/candidates/reactivate", status_code=200)
def reactivate_candidate(sid: str, payload: CandidateToggle):
    """恢复一个已停用候选项：按最初录入顺序回到评审。

    其与其他候选的已有组合沿用原判断记录，仅缺失（此前未比较）的组合
    重新进入待评审。
    """
    conn = get_conn()
    try:
        return _toggle_candidate(conn, sid, payload.name, disable=False)
    finally:
        conn.close()


# ---------- 情景沙盘 API ----------

@app.get("/api/sessions/{sid}/sandbox")
def get_sandbox(sid: str):
    """读取沙盘：原状态与「原判断 + 暂存覆盖层」重算状态并列返回。

    纯只读：绝不写 judgments，也不修改草稿。无草稿时沙盘状态即原状态。
    草稿存在且其保存后基础判断发生过变化时，draft.stale = true（应用会被拒绝）。
    """
    conn = get_conn()
    try:
        return _load_sandbox_payload(conn, sid)
    finally:
        conn.close()


@app.put("/api/sessions/{sid}/sandbox")
def save_sandbox(sid: str, payload: SandboxDraft):
    """保存（整体替换）沙盘暂存草稿。

    仅写 sandbox_drafts 覆盖层，绝不写 judgments；与当前基础判断结果相同的
    暂存条目不保留（空草稿即删除）。刷新页面后通过 GET 恢复。
    """
    if len(payload.changes) > MAX_CANDIDATES * (MAX_CANDIDATES - 1) // 2:
        raise HTTPException(422, "暂存改动数量超出组合总数上限")
    conn = get_conn()
    try:
        all_candidates, disabled_set, all_history = _load_rows(conn, sid)
        candidates = [name for name in all_candidates if name not in disabled_set]
        normalized = _normalize_sandbox_changes(
            candidates, set(all_candidates) - set(candidates), payload.changes
        )
        enabled = set(candidates)
        judged = {
            (h["a"], h["b"]): h["result"]
            for h in all_history
            if h["a"] in enabled and h["b"] in enabled
        }
        # 只保留相对当前基础判断真正有差异的覆盖层（无差异即无暂存意义）
        effective = [ch for ch in normalized if judged.get((ch["a"], ch["b"])) != ch["result"]]
        base_version = _base_version(all_candidates, disabled_set, all_history)
        if effective:
            conn.execute(
                "INSERT INTO sandbox_drafts (session_id, changes, base_version) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "changes = excluded.changes, base_version = excluded.base_version, "
                "updated_at = datetime('now')",
                (sid, json.dumps(effective, ensure_ascii=False), base_version),
            )
        else:
            conn.execute("DELETE FROM sandbox_drafts WHERE session_id = ?", (sid,))
        conn.commit()
        return _load_sandbox_payload(conn, sid)
    finally:
        conn.close()


@app.delete("/api/sessions/{sid}/sandbox")
def discard_sandbox(sid: str):
    """丢弃沙盘草稿（删除覆盖层）；judgments 不受任何影响。"""
    conn = get_conn()
    try:
        if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone() is None:
            raise HTTPException(404, "会话不存在或已被删除")
        conn.execute("DELETE FROM sandbox_drafts WHERE session_id = ?", (sid,))
        conn.commit()
        return _load_sandbox_payload(conn, sid)
    finally:
        conn.close()


@app.post("/api/sessions/{sid}/sandbox/apply")
def apply_sandbox(sid: str):
    """一次确认把沙盘全部暂存改动写入正式判断。

    - 应用前在同一事务内重新计算基础判断指纹，与草稿保存时的指纹比对；
      基础判断已变化（新增 / 撤销 / 复核 / 批量改判 / 重置 / 停用恢复）即
      **整体拒绝（回滚）且保留草稿**，返回 409；
    - 已判组合原位 UPDATE（行 id、进度与撤销顺序不变），未判组合按覆盖层
      顺序 INSERT 补写；所有读写在**单个 SQLite 事务**中完成，任一异常回滚；
    - 提交成功后删除草稿，返回写入后的完整正式状态。
    """
    conn = get_conn()
    try:
        with conn:  # 单事务：无异常提交，异常自动回滚
            all_candidates, disabled_set, all_history = _load_rows(conn, sid)
            candidates = [name for name in all_candidates if name not in disabled_set]
            order = {name: i for i, name in enumerate(candidates)}
            enabled = set(candidates)

            row = conn.execute(
                "SELECT changes, base_version FROM sandbox_drafts WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                raise HTTPException(409, "沙盘草稿为空，没有可应用的暂存改动")

            current_version = _base_version(
                all_candidates, disabled_set, all_history
            )
            if row["base_version"] != current_version:
                raise HTTPException(
                    409,
                    "基础判断在沙盘保存后已发生变化，为避免覆盖已拒绝本次应用；"
                    "沙盘草稿已保留，请核对后重新暂存",
                )

            raw_changes: list[dict] = json.loads(row["changes"])
            judged = {
                (h["a"], h["b"]): h["result"]
                for h in all_history
                if h["a"] in enabled and h["b"] in enabled
            }
            updates, inserts = [], []
            for ch in raw_changes:
                a, b, result = ch["a"], ch["b"], ch["result"]
                # 组合按候选录入序排列；含停用候选的失效条目直接拒绝
                if a not in order or b not in order:
                    raise HTTPException(
                        409, f"暂存组合「{a} vs {b}」已失效（候选可能已停用），请重新暂存"
                    )
                a, b = sorted((a, b), key=order.__getitem__)
                if (a, b) in judged:
                    if judged[(a, b)] != result:
                        updates.append((result, a, b))
                else:
                    inserts.append((a, b, result))

            for result, a, b in updates:
                cur = conn.execute(
                    "UPDATE judgments SET result = ? "
                    "WHERE session_id = ? AND a = ? AND b = ?",
                    (result, sid, a, b),
                )
                if cur.rowcount != 1:  # 指纹一致却未命中，视为并发写入冲突
                    raise HTTPException(
                        409, f"组合「{a} vs {b}」的基础判断已变化，应用已拒绝，草稿已保留"
                    )
            for a, b, result in inserts:
                conn.execute(
                    "INSERT INTO judgments (session_id, a, b, result) "
                    "VALUES (?, ?, ?, ?)",
                    (sid, a, b, result),
                )

            conn.execute("DELETE FROM sandbox_drafts WHERE session_id = ?", (sid,))
            return load_state(conn, sid)
    finally:
        conn.close()


# ---------- 静态页面 ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
