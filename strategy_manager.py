"""
Solana AI Quant Agent - Custom Strategy Manager & Lifecycle Engine
Handles persistence, parameter tuning, dynamic compilation, paper trading deployment,
and lifecycle operations (Save, Edit, Pause, Resume, Delete).
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from strategy_base import BaseStrategy
from strategy_transpiler import StrategyTranspiler


class StrategyManager:
    """
    Manages custom quantitative trading strategies:
    - Persistence in data/custom_strategies.json
    - Dynamic compilation into BaseStrategy
    - Virtual paper broker binding
    - Lifecycle controls (Deploy, Pause, Resume, Edit, Delete)
    """

    def __init__(self, data_path: str = "data/custom_strategies.json"):
        self.data_path = data_path
        self._active_strategy_instance: Optional[BaseStrategy] = None
        self._active_strategy_id: Optional[str] = None
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensures storage directory and initial presets exist."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        if not os.path.exists(self.data_path):
            presets = self._build_default_presets()
            self._write_file(presets)

    def _read_file(self) -> List[Dict[str, Any]]:
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[StrategyManager] 警告: 读取策略存储失败 ({e})，返回空列表")
        return []

    def _write_file(self, data: List[Dict[str, Any]]):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _build_default_presets(self) -> List[Dict[str, Any]]:
        """Generates ready-to-run presets across the 4 major trading languages."""
        presets = []
        langs = [
            ("tbquant", "TBQuant 经典双均线波幅策略", "TBQuant (开拓者)"),
            ("mylanguage", "文华财经 唐奇安通道策略", "文华财经 (麦语言)"),
            ("tdx", "通达信 均线放量共振选股", "通达信 (公式系统)"),
            ("tradingview", "TradingView Pine 动量均线", "TradingView (Pine Script)")
        ]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for lang_key, name, desc in langs:
            source = StrategyTranspiler.get_template(lang_key)
            py_code, meta = StrategyTranspiler.transpile(source, language=lang_key, strategy_name=name)
            strat_id = f"preset_{lang_key}"
            presets.append({
                "id": strat_id,
                "name": name,
                "language": lang_key,
                "description": desc,
                "source_code": source,
                "python_code": py_code,
                "parameters": meta.get("parameters", {}),
                "status": "STANDBY",  # STANDBY, RUNNING, PAUSED
                "is_active_paper": False,
                "created_at": now_str,
                "updated_at": now_str,
                "backtest_summary": None
            })
        return presets

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Returns all registered custom strategies."""
        return self._read_file()

    def get_strategy(self, strat_id: str) -> Optional[Dict[str, Any]]:
        """Finds strategy by ID."""
        for item in self._read_file():
            if item.get("id") == strat_id:
                return item
        return None

    def save_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves or updates a strategy.
        Automatically transpiles source code if python_code is not provided or source changed.
        """
        strategies = self._read_file()
        strat_id = data.get("id")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        source_code = data.get("source_code", "").strip()
        lang = data.get("language") or StrategyTranspiler.detect_language(source_code)
        name = data.get("name") or "自定义量化策略"

        # Transpile if needed
        python_code = data.get("python_code")
        params = data.get("parameters") or {}
        if not python_code and source_code:
            py_code, meta = StrategyTranspiler.transpile(source_code, language=lang, strategy_name=name)
            python_code = py_code
            if not params:
                params = meta.get("parameters", {})

        # Verify compilation
        if python_code:
            try:
                _ = StrategyTranspiler.compile_strategy_instance(python_code, params)
            except Exception as e:
                raise ValueError(f"策略编译校验失败: {str(e)}")

        existing = None
        for i, s in enumerate(strategies):
            if s.get("id") == strat_id:
                existing = s
                break

        if existing:
            existing["name"] = name
            existing["language"] = lang
            existing["source_code"] = source_code
            existing["python_code"] = python_code
            existing["parameters"] = params
            existing["updated_at"] = now_str
            if "status" in data:
                existing["status"] = data["status"]
            if "is_active_paper" in data:
                existing["is_active_paper"] = data["is_active_paper"]
            result = existing
        else:
            strat_id = strat_id or f"strat_{uuid.uuid4().hex[:8]}"
            new_strat = {
                "id": strat_id,
                "name": name,
                "language": lang,
                "description": data.get("description", "用户自定义量化策略"),
                "source_code": source_code,
                "python_code": python_code,
                "parameters": params,
                "status": data.get("status", "STANDBY"),
                "is_active_paper": data.get("is_active_paper", False),
                "created_at": now_str,
                "updated_at": now_str,
                "backtest_summary": None
            }
            strategies.insert(0, new_strat)
            result = new_strat

        self._write_file(strategies)

        # Invalidate cached active instance if modified
        if self._active_strategy_id == strat_id:
            self._active_strategy_instance = None

        return result

    def delete_strategy(self, strat_id: str) -> bool:
        """Deletes a strategy by ID."""
        strategies = self._read_file()
        new_list = [s for s in strategies if s.get("id") != strat_id]
        if len(new_list) < len(strategies):
            self._write_file(new_list)
            if self._active_strategy_id == strat_id:
                self._active_strategy_id = None
                self._active_strategy_instance = None
            return True
        return False

    def pause_strategy(self, strat_id: str) -> Optional[Dict[str, Any]]:
        """Pauses a running strategy."""
        strategies = self._read_file()
        target = None
        for s in strategies:
            if s.get("id") == strat_id:
                s["status"] = "PAUSED"
                s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                target = s
                break
        if target:
            self._write_file(strategies)
        return target

    def resume_strategy(self, strat_id: str) -> Optional[Dict[str, Any]]:
        """Resumes a paused strategy."""
        strategies = self._read_file()
        target = None
        for s in strategies:
            if s.get("id") == strat_id:
                s["status"] = "RUNNING"
                s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                target = s
                break
        if target:
            self._write_file(strategies)
        return target

    def deploy_to_paper(self, strat_id: str) -> Dict[str, Any]:
        """
        Deploys strategy to virtual paper trading execution.
        Sets status to RUNNING and is_active_paper to True.
        """
        strategies = self._read_file()
        target = None
        for s in strategies:
            if s.get("id") == strat_id:
                s["is_active_paper"] = True
                s["status"] = "RUNNING"
                s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                target = s
            else:
                s["is_active_paper"] = False
                if s["status"] == "RUNNING":
                    s["status"] = "STANDBY"

        if not target:
            raise ValueError(f"未找到 ID 为 {strat_id} 的策略")

        # Compile and verify instance
        py_code = target.get("python_code")
        params = target.get("parameters", {})
        instance = StrategyTranspiler.compile_strategy_instance(py_code, params)

        self._active_strategy_id = strat_id
        self._active_strategy_instance = instance
        self._write_file(strategies)
        return target

    def undeploy_paper(self) -> None:
        """Detaches any active paper strategy, returning to default trend agent."""
        strategies = self._read_file()
        for s in strategies:
            s["is_active_paper"] = False
            if s["status"] == "RUNNING":
                s["status"] = "STANDBY"
        self._write_file(strategies)
        self._active_strategy_id = None
        self._active_strategy_instance = None

    def get_active_paper_strategy(self) -> Optional[BaseStrategy]:
        """
        Returns the compiled BaseStrategy instance active for paper trading,
        or None if no custom strategy is active or if active strategy is paused.
        """
        if self._active_strategy_id and self._active_strategy_instance:
            # Check if paused
            curr = self.get_strategy(self._active_strategy_id)
            if curr and curr.get("status") == "PAUSED":
                return None
            return self._active_strategy_instance

        # Check file for active strategy
        strategies = self._read_file()
        for s in strategies:
            if s.get("is_active_paper") and s.get("status") == "RUNNING":
                try:
                    instance = StrategyTranspiler.compile_strategy_instance(
                        s.get("python_code"), s.get("parameters", {})
                    )
                    self._active_strategy_id = s.get("id")
                    self._active_strategy_instance = instance
                    return instance
                except Exception as e:
                    print(f"[StrategyManager] 激活策略实例化失败: {e}")
                    return None
        return None

    def get_active_paper_info(self) -> Optional[Dict[str, Any]]:
        """Returns metadata of currently active paper strategy."""
        strategies = self._read_file()
        for s in strategies:
            if s.get("is_active_paper"):
                return {
                    "id": s["id"],
                    "name": s["name"],
                    "language": s["language"],
                    "status": s["status"]
                }
        return None

    def update_backtest_summary(self, strat_id: str, summary: Dict[str, Any]):
        """Caches latest backtest metrics for quick reference."""
        strategies = self._read_file()
        for s in strategies:
            if s.get("id") == strat_id:
                s["backtest_summary"] = summary
                break
        self._write_file(strategies)


# Global singleton manager
strategy_manager = StrategyManager()
