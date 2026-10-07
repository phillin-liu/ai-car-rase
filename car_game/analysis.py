"""Post-match analysis: metrics, scoring, grades and improvement advice."""
from __future__ import annotations

import math
from datetime import datetime


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _grade(score: float) -> str:
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
def analyze_match(result: dict) -> dict:
    """Turn a raw MatchResult dict into a scored, human readable analysis."""
    result = dict(result)
    cars = result.get("cars", [])
    driver_stats = result.get("driver_stats", [])
    models = result.get("models", [f"P{i+1}" for i in range(len(cars))])
    laps = max(1, int(result.get("laps", 1)))
    winner = result.get("winner")

    finishers = [c for c in cars if c.get("finished")]
    best_time = min((_num(c.get("finish_time")) for c in finishers), default=0.0)

    analyses = []
    for idx, car in enumerate(cars):
        dist_km = max(_num(car.get("distance")) / 1000.0, 0.25)
        wall = _num(car.get("wall_hits"))
        obs = _num(car.get("obstacle_hits"))
        hazard = _num(car.get("hazard_hits"))
        avg_speed = _num(car.get("avg_speed"))
        lat = _num(car.get("avg_lateral_ratio"))
        off_center = _num(car.get("off_center_time"))
        items = _num(car.get("item_use_total"))
        finished = bool(car.get("finished"))
        ftime = _num(car.get("finish_time"))
        progress = _num(car.get("progress"))
        dstat = driver_stats[idx] if idx < len(driver_stats) else {}

        # --- sub score: result ---------------------------------------
        if finished and best_time > 0:
            result_score = _clamp(100.0 * best_time / max(ftime, 1e-3))
            if winner == idx:
                result_score = 100.0
        elif finished:
            result_score = 90.0
        else:
            result_score = _clamp(progress * 100.0 * 0.55)

        # --- sub score: driving quality ------------------------------
        wall_per_km = wall / dist_km
        obs_per_km = obs / dist_km
        haz_per_km = hazard / dist_km
        penalty = wall_per_km * 3.0 + obs_per_km * 8.0 + haz_per_km * 3.0
        penalty += lat * 25.0 + min(off_center, 30.0) * 0.35
        driving_score = _clamp(100.0 - penalty)

        # --- sub score: speed ----------------------------------------
        reference = 38.0
        speed_score = _clamp(100.0 * avg_speed / reference)

        # --- sub score: items ----------------------------------------
        expected = max(4.0, laps * 7.0)
        use_ratio = items / expected
        item_score = _clamp(100.0 * min(use_ratio, 1.15) / 1.0)
        if use_ratio < 0.35:
            item_score *= 0.85

        # --- sub score: reliability ----------------------------------
        success = _num(dstat.get("success_rate"), 1.0)
        api_errors = _num(dstat.get("api_errors"))
        reliability = _clamp(success * 100.0 - min(api_errors, 10) * 1.5)

        total = (result_score * 0.35 + driving_score * 0.25 +
                 speed_score * 0.15 + item_score * 0.13 +
                 reliability * 0.12)

        subscores = {
            "完成度": round(result_score, 1),
            "驾驶质量": round(driving_score, 1),
            "平均速度": round(speed_score, 1),
            "道具运用": round(item_score, 1),
            "稳定性": round(reliability, 1),
        }
        metrics = {
            "完赛": finished,
            "用时(s)": round(ftime, 2) if finished else None,
            "平均速度": round(avg_speed, 2),
            "最高速度": round(_num(car.get("max_speed")), 2),
            "行驶距离(m)": round(_num(car.get("distance")), 1),
            "撞墙次数": int(wall),
            "撞障碍物次数": int(obs),
            "中陷阱次数": int(hazard),
            "道具使用次数": int(items),
            "平均横向偏移": round(lat, 4),
            "偏离中心时间(s)": round(off_center, 2),
            "进度": f"{progress*100:.1f}%",
        }
        advices = _suggestions(metrics, subscores, dstat, use_ratio)
        analyses.append({
            "index": idx,
            "name": car.get("name", f"P{idx+1}"),
            "model": models[idx] if idx < len(models) else f"P{idx+1}",
            "metrics": metrics,
            "subscores": subscores,
            "total": round(total, 1),
            "grade": _grade(total),
            "advice": advices,
            "driver_stats": dstat,
        })

    table = _comparison_table(analyses)
    summary = _summary(result, analyses)

    return {
        "match_id": result.get("match_id"),
        "seed": result.get("seed"),
        "laps": laps,
        "mode": result.get("mode"),
        "duration": round(_num(result.get("duration")), 2),
        "winner": winner,
        "end_reason": result.get("end_reason", ""),
        "timestamp": result.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "analyses": analyses,
        "table": table,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
def _suggestions(metrics: dict, subscores: dict, dstat: dict, use_ratio: float):
    out = []
    if not metrics["完赛"]:
        out.append("未能完赛：需要提升稳定性，避免在弯道失控或被卡住。")
    if metrics["撞墙次数"] >= 8:
        out.append("撞墙偏多：提前刹车、优化过弯线路，减少贴墙摩擦。")
    elif metrics["撞墙次数"] >= 4:
        out.append("仍有撞墙：注意弯心与出弯的横向控制。")
    if metrics["撞障碍物次数"] >= 3:
        out.append("障碍物规避不足：扩大预判距离，提前选择绕行线路。")
    if metrics["平均横向偏移"] > 0.22:
        out.append(f"平均横向偏移较大({metrics['平均横向偏移']:.2f})：应更贴近赛道中心线。")
    if metrics["偏离中心时间(s)"] > 12:
        out.append("偏离中心时间较长：加强横向稳定性与线路规划。")
    if use_ratio < 0.5:
        out.append("道具使用偏少：在直线使用氮气、在危险处使用护盾可提升成绩。")
    if metrics["平均速度"] < 26:
        out.append("平均速度偏低：可在保证不撞墙的前提下提高弯道速度。")
    if _num(dstat.get("api_errors")) > 0:
        out.append("API 调用存在错误：检查密钥、网络与超时设置，或降低决策频率。")
    if not out:
        out.append("表现优秀：保持当前策略，可尝试更激进的速度目标。")
    return out


def _comparison_table(analyses: list) -> list:
    if len(analyses) < 2:
        return []
    a, b = analyses[0], analyses[1]
    rows = []
    keys = ["完赛", "用时(s)", "平均速度", "最高速度", "撞墙次数",
            "撞障碍物次数", "道具使用次数", "平均横向偏移", "进度", "总分"]
    for k in keys:
        if k == "总分":
            rows.append([k, a["total"], b["total"]])
        else:
            rows.append([k, a["metrics"].get(k), b["metrics"].get(k)])
    for k in a["subscores"]:
        rows.append([k, a["subscores"][k], b["subscores"].get(k)])
    return rows


def _summary(result: dict, analyses: list) -> str:
    if result.get("end_reason"):
        return str(result["end_reason"])
    if len(analyses) < 2:
        return ""
    winner = result.get("winner")
    if winner is None:
        head = "本局无获胜方（超时/平局）。"
    else:
        w = analyses[winner] if winner < len(analyses) else analyses[0]
        head = f"本局 {w['name']} ({w['model']}) 率先到达终点获胜。"
    a, b = analyses[0], analyses[1]
    diff = abs(a["total"] - b["total"])
    if diff < 3:
        tail = "双方综合评分接近，胜负更多取决于临场道具与线路选择。"
    else:
        better = a if a["total"] > b["total"] else b
        tail = f"{better['model']} 综合评分更高（{better['total']:.1f}），整体表现更优。"
    return head + tail


# ---------------------------------------------------------------------------
def aggregate(matches: list) -> dict:
    """Aggregate per-model statistics across many analysed matches."""
    stats: dict = {}
    for m in matches:
        for a in m.get("analyses", []):
            model = a["model"]
            s = stats.setdefault(model, {
                "model": model, "matches": 0, "wins": 0, "finishes": 0,
                "total_sum": 0.0, "speed_sum": 0.0, "wall_sum": 0.0,
                "obs_sum": 0.0, "item_sum": 0.0,
                "subscore_sum": {k: 0.0 for k in a["subscores"]},
            })
            s["matches"] += 1
            s["total_sum"] += a["total"]
            s["speed_sum"] += a["metrics"]["平均速度"]
            s["wall_sum"] += a["metrics"]["撞墙次数"]
            s["obs_sum"] += a["metrics"]["撞障碍物次数"]
            s["item_sum"] += a["metrics"]["道具使用次数"]
            if a["metrics"]["完赛"]:
                s["finishes"] += 1
            if m.get("winner") == a["index"]:
                s["wins"] += 1
            for k, v in a["subscores"].items():
                s["subscore_sum"][k] += v
    out = []
    for model, s in stats.items():
        n = max(1, s["matches"])
        out.append({
            "model": model,
            "matches": s["matches"],
            "wins": s["wins"],
            "win_rate": round(100.0 * s["wins"] / n, 1),
            "finish_rate": round(100.0 * s["finishes"] / n, 1),
            "avg_score": round(s["total_sum"] / n, 1),
            "avg_speed": round(s["speed_sum"] / n, 2),
            "avg_wall": round(s["wall_sum"] / n, 2),
            "avg_obs": round(s["obs_sum"] / n, 2),
            "avg_items": round(s["item_sum"] / n, 2),
            "subscores": {k: round(v / n, 1) for k, v in s["subscore_sum"].items()},
        })
    out.sort(key=lambda x: x["avg_score"], reverse=True)
    return out


def format_comparison_table(analysis: dict) -> str:
    """Render the comparison table as an aligned text table."""
    exams = analysis.get("analyses", [])
    if len(exams) < 2:
        return "(至少需要两辆车才能生成对比表)"
    a, b = exams[0], exams[1]
    rows = analysis.get("table", [])
    header = ["指标", f"{a['name']}({a['model']})", f"{b['name']}({b['model']})"]
    widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(3)]
    lines = []
    lines.append("  ".join(str(header[i]).ljust(widths[i]) for i in range(3)))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(3)))
    lines.append("")
    for a in exams:
        lines.append(f"[{a['name']}] 综合评分 {a['total']} 等级 {a['grade']}")
        for s in a["advice"]:
            lines.append(f"   - {s}")
    return "\n".join(lines)


def format_aggregate_table(agg: list) -> str:
    if not agg:
        return "(暂无数据)"
    cols = ["模型", "场次", "胜率%", "完赛率%", "平均分", "平均速度", "平均撞墙", "平均撞障碍"]
    keys = ["model", "matches", "win_rate", "finish_rate", "avg_score",
            "avg_speed", "avg_wall", "avg_obs"]
    data = [cols] + [[str(r[k]) for k in keys] for r in agg]
    widths = [max(len(row[i]) for row in data) for i in range(len(cols))]
    lines = ["  ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    for r in agg:
        lines.append("  ".join(str(r[k]).ljust(widths[i]) for i, k in enumerate(keys)))
    return "\n".join(lines)
