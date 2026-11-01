"""Session match storage, plus the cross-session history.

There is no project concept any more: every run writes its matches into a
single transient session directory under ``runtime/session/``.  The training
report is built from whatever matches that store currently holds, and the
console clears it when a new run starts (reminding the user to export the
report first).

:class:`RuntimeHistoryStore` is the counterpart that *survives* that clear: it
lives in ``runtime/history/`` and nothing ever wipes it automatically, so a
run can always be looked at again.  Deleting from it is explicit and manual.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from datetime import datetime

from .analysis import aggregate
from .config import RUNTIME_DIR, ensure_dirs

SESSION_DIR = os.path.join(RUNTIME_DIR, "session")
HISTORY_DIR = os.path.join(RUNTIME_DIR, "history")


class MatchStore:
    """Persist the matches of the current session and aggregate them."""

    def __init__(self, root: str | None = None):
        self.root = root or SESSION_DIR
        ensure_dirs()
        os.makedirs(self.root, exist_ok=True)
        # parsed results, keyed on the directory fingerprint (see signature)
        self._cache_sig = None
        self._cache_list = None
        self._cache_agg = None

    # ------------------------------------------------------------------
    @property
    def matches_dir(self) -> str:
        return os.path.join(self.root, "matches")

    def signature(self):
        """Cheap fingerprint of the match directory.

        ``list_matches`` and ``aggregate`` open and parse *every* stored file,
        which at a few hundred matches is megabytes of JSON on the GUI thread.
        Both the console and the history window poll them for live updates, so
        the parsed result is cached and only rebuilt when a file is added,
        removed or rewritten.  Checking the fingerprint is a directory scan --
        no JSON is parsed to decide.
        """
        mdir = self.matches_dir
        if not os.path.isdir(mdir):
            return ()
        sig = []
        try:
            for fn in os.listdir(mdir):
                if not (fn.startswith("match_") and fn.endswith(".json")):
                    continue
                try:
                    st = os.stat(os.path.join(mdir, fn))
                except OSError:
                    continue
                sig.append((fn, st.st_mtime_ns, st.st_size))
        except OSError:
            return ()
        sig.sort()
        return tuple(sig)

    def clear(self) -> None:
        """Wipe the stored matches so a fresh run starts from zero."""
        shutil.rmtree(self.matches_dir, ignore_errors=True)
        os.makedirs(self.matches_dir, exist_ok=True)

    def count(self) -> int:
        mdir = self.matches_dir
        if not os.path.isdir(mdir):
            return 0
        return sum(1 for fn in os.listdir(mdir)
                   if fn.startswith("match_") and fn.endswith(".json"))

    # ------------------------------------------------------------------
    def save_match(self, analysis: dict, result: dict, config: dict) -> str:
        os.makedirs(self.matches_dir, exist_ok=True)
        mid = analysis.get("match_id", 0)
        path = os.path.join(self.matches_dir, f"match_{int(mid):04d}.json")
        payload = {
            "analysis": analysis,
            "result": result,
            "config": config,
            "saved": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_json(path, payload)
        return path

    def _invalidate(self, sig) -> None:
        """Drop both parsed caches when the directory fingerprint moved."""
        if sig != self._cache_sig:
            self._cache_sig = sig
            self._cache_list = None
            self._cache_agg = None

    def list_matches(self) -> list:
        sig = self.signature()
        self._invalidate(sig)
        if self._cache_list is None:
            out = self._read_matches()
            out.sort(key=lambda m: (m.get("match_id") is None,
                                    m.get("match_id") or 0))
            self._cache_list = out
        # a deep copy: the rows are handed to whoever asks, and one caller
        # editing a row must not poison the cache for every later reader
        return copy.deepcopy(self._cache_list)

    def _read_matches(self) -> list:
        mdir = self.matches_dir
        out = []
        if not os.path.isdir(mdir):
            return out
        for fn in sorted(os.listdir(mdir)):
            if not (fn.startswith("match_") and fn.endswith(".json")):
                continue
            path = os.path.join(mdir, fn)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                analysis = payload.get("analysis", {})
                out.append({
                    "file": fn,
                    "match_id": analysis.get("match_id"),
                    "timestamp": analysis.get("timestamp"),
                    "mode": analysis.get("mode"),
                    "models": analysis.get("models"),
                    "winner": analysis.get("winner"),
                    "scores": [a.get("total") for a in analysis.get("analyses", [])],
                    "path": path,
                })
            except (OSError, ValueError):
                continue
        return out

    def load_match(self, match_id: int) -> dict:
        path = os.path.join(self.matches_dir, f"match_{int(match_id):04d}.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_results(self) -> list:
        """Return the raw MatchResult dicts stored for this session."""
        results = []
        mdir = self.matches_dir
        if not os.path.isdir(mdir):
            return results
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(mdir, fn), "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                result = payload.get("result")
                if isinstance(result, dict):
                    results.append(result)
            except (OSError, ValueError):
                continue
        return results

    def aggregate(self) -> list:
        sig = self.signature()
        self._invalidate(sig)
        if self._cache_agg is None:
            analyses = []
            mdir = self.matches_dir
            if os.path.isdir(mdir):
                for fn in sorted(os.listdir(mdir)):
                    if fn.endswith(".json"):
                        try:
                            with open(os.path.join(mdir, fn), "r",
                                      encoding="utf-8") as fh:
                                analyses.append(json.load(fh).get("analysis", {}))
                        except (OSError, ValueError):
                            continue
            self._cache_agg = aggregate(analyses)
        return copy.deepcopy(self._cache_agg)

    def verify(self, persist: bool = True) -> dict:
        """Batch anti-cheat verification over every stored match."""
        from .anticheat import verify_matches

        results = self.load_results()
        verdict = verify_matches(results)
        verdict["checked"] = len(results)
        if persist:
            data = dict(verdict)
            data["updated"] = datetime.now().isoformat(timespec="seconds")
            self._write_json(os.path.join(self.root, "session.json"), data)
        return verdict

    # ------------------------------------------------------------------
    def export_report(self, path: str) -> str:
        """Write a PDF training report for the current session."""
        from .report import build_report
        return build_report(self, path)

    @staticmethod
    def _write_json(path: str, data) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
class RuntimeHistoryStore(MatchStore):
    """Every match ever run, kept across sessions and deleted only by hand.

    A plain :class:`MatchStore` rooted at ``runtime/history`` -- the runner's
    session ``clear()`` never touches it -- with two additions:

    * **Ids are reassigned on save.**  Each session numbers its matches from 1,
      so storing them verbatim would make every session overwrite the last
      one's ``match_0001.json`` (``load_match`` resolves a record by its
      filename).  The session's own number is kept alongside as
      ``session_match_id``.
    * :meth:`delete` / :meth:`clear_all`, which the session store has no
      concept of.

    The reassignment works on a *copy*: the runner hands the same analysis dict
    to both stores, and mutating it here would renumber the live session too.
    """

    def __init__(self, root: str | None = None):
        super().__init__(root or HISTORY_DIR)

    # ------------------------------------------------------------------
    def next_id(self) -> int:
        """The id a record saved right now would get (max existing + 1)."""
        highest = 0
        for m in self.list_matches():
            try:
                highest = max(highest, int(m.get("match_id") or 0))
            except (TypeError, ValueError):
                continue
        return highest + 1

    def save_match(self, analysis: dict, result: dict, config: dict) -> str:
        rec = copy.deepcopy(analysis or {})
        rec["session_match_id"] = (analysis or {}).get("match_id")
        rec["match_id"] = self.next_id()
        return super().save_match(rec, result, config)

    # ------------------------------------------------------------------
    def delete(self, match_id: int) -> bool:
        """Remove one record.  Returns True when a file was actually removed."""
        path = os.path.join(self.matches_dir, f"match_{int(match_id):04d}.json")
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    def clear_all(self) -> None:
        """Wipe the whole history (the console asks for confirmation first)."""
        self.clear()

    def record_path(self, match_id: int) -> str:
        """Where a record lives -- used by the console's "open folder" action."""
        return os.path.join(self.matches_dir, f"match_{int(match_id):04d}.json")
