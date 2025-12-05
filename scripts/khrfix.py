# kohya_hires_fix_unified_v2.4.py
# Версия: 2.4 (Ultimate Edition)
# Совместимость: A1111 / modules.scripts API, PyTorch >= 1.12
# Изменения:
# - Все исправления безопасности v2.3 (NoneType, d1==d2 fix, Model validation)
# - Добавлено: Quick Presets, Tooltips, Status Indicator, Parameter Validation
# - Добавлено: Debug Mode, Config Export/Import, Built-in Help
# - Улучшена валидация индексов в Legacy режиме

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from modules import scripts, script_callbacks

CONFIG_PATH = Path(__file__).with_suffix(".yaml")
PRESETS_PATH = Path(__file__).with_name(Path(__file__).stem + ".presets.yaml")

# ---- Предустановленные разрешения ----

RESOLUTION_GROUPS = {
    "Квадрат": [(1024, 1024)],
    "Портрет": [(640, 1536), (768, 1344), (832, 1216), (896, 1152), (768, 1152)],
    "Альбом": [(1536, 640), (1344, 768), (1216, 832), (1152, 896), (1024, 1536)],
}
RESOLUTION_CHOICES: List[str] = ["— не применять —"]
for group, dims in RESOLUTION_GROUPS.items():
    for w, h in dims:
        RESOLUTION_CHOICES.append(f"{group}: {w}x{h}")


def parse_resolution_label(label: str) -> Optional[Tuple[int, int]]:
    if not label or label.startswith("—"):
        return None
    try:
        _, wh = label.split(":")
        w, h = wh.strip().lower().split("x")
        return int(w), int(h)
    except Exception:
        return None


# ---- Вспомогательные утилиты ----

def _safe_mode(mode: str) -> str:
    if mode == "nearest-exact":
        return mode
    if mode in {"bicubic", "bilinear", "nearest"}:
        return mode
    return "bilinear"


def _load_yaml(path: Path, default: dict) -> dict:
    try:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or default
    except Exception:
        return default


def _atomic_save_yaml(path: Path, data: dict) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        OmegaConf.save(DictConfig(data), tmp)
        tmp.replace(path)
    except Exception:
        pass


def _load_presets() -> Dict[str, dict]:
    data = _load_yaml(PRESETS_PATH, {})
    return {str(k): dict(v) for k, v in data.items()}


def _save_presets(presets: Dict[str, dict]) -> None:
    _atomic_save_yaml(PRESETS_PATH, presets)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _norm_mode_choice(value: str, default_: str = "auto") -> str:
    try:
        v = str(value).strip().lower()
    except Exception:
        v = ""
    if v in {"true", "t", "1", "yes", "y"}:
        return "true"
    if v in {"false", "f", "0", "no", "n"}:
        return "false"
    if v in {"auto", "a", "авто"}:
        return "auto"
    return str(default_).strip().lower()


def _compute_adaptive_params(
    width: int,
    height: int,
    profile: str,
    base_s1: float,
    base_s2: float,
    base_d1: int,
    base_d2: int,
    base_down: float,
    base_up: float,
    keep_unitary_product: bool,
) -> Tuple[float, float, int, int, float, float]:
    rel_mpx = (max(1, int(width)) * max(1, int(height))) / float(1024 * 1024)
    aspect = max(width, height) / float(min(width, height))

    prof = (profile or "").strip().lower()
    s1 = float(base_s1)
    s2 = float(base_s2)
    d1 = int(base_d1)
    d2 = int(base_d2)
    down = float(base_down)
    up = float(base_up)

    if prof.startswith("конс"):
        s_add = -0.02
        d_add = 0
    elif prof.startswith("агре"):
        s_add = 0.05
        d_add = 1
    else:
        s_add = 0.0
        d_add = 0

    if rel_mpx >= 1.5:
        s_add += 0.08
        down -= 0.10
    elif rel_mpx >= 1.1:
        s_add += 0.05
        down -= 0.05
    elif rel_mpx <= 0.8:
        s_add -= 0.02
        down += 0.05

    if aspect >= 1.6:
        d_add += 1
    elif aspect <= 1.1:
        d_add -= 1

    s1 = _clamp(s1 + s_add * 0.7, 0.0, 1.0)
    s2 = _clamp(s2 + s_add, 0.0, 1.0)

    d1 = max(1, d1 + d_add)
    d2 = max(1, d2 + d_add)

    down = _clamp(down, 0.1, 1.0)
    if keep_unitary_product:
        up = 1.0 / max(1e-6, down)
    else:
        up = _clamp(up * (base_down / max(1e-6, down)), 1.0, 4.0)

    return s1, s2, d1, d2, down, up


# ---- Класс пресета ----

class HiresPreset:
    def __init__(self, **kwargs: Any) -> None:
        self.category: str = "Общие"
        self.algo_mode: str = "Enhanced (RU+)"

        self.d1: int = 3
        self.d2: int = 4
        self.s1: float = 0.15
        self.s2: float = 0.30

        self.scaler: str = "bicubic"
        self.downscale: float = 0.5
        self.upscale: float = 2.0

        self.smooth_scaling_enh: bool = True
        self.smooth_scaling_legacy: bool = True

        self.smoothing_curve: str = "Линейная"

        self.early_out: bool = False

        self.only_one_pass_enh: bool = True
        self.only_one_pass_legacy: bool = True

        self.depth_guard: bool = True

        self.keep_unitary_product: bool = False
        self.align_corners_mode: str = "False"
        self.recompute_scale_factor_mode: str = "False"
        self.smoothing_mode: str = "Авто (по алгоритму)"
        self.one_pass_mode: str = "Авто (по алгоритму)"

        self.resolution_choice: str = RESOLUTION_CHOICES[0]
        self.apply_resolution: bool = False
        self.adaptive_by_resolution: bool = True
        self.adaptive_profile: str = "Сбалансированный"

        # Legacy keys support
        legacy_smooth = kwargs.pop("smooth_scaling", None)
        legacy_one = kwargs.pop("only_one_pass", None)

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if legacy_smooth is not None:
            self.smooth_scaling_enh = bool(legacy_smooth)
            self.smooth_scaling_legacy = bool(legacy_smooth)
        if legacy_one is not None:
            self.only_one_pass_enh = bool(legacy_one)
            self.only_one_pass_legacy = bool(legacy_one)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "algo_mode": self.algo_mode,
            "d1": self.d1,
            "d2": self.d2,
            "s1": self.s1,
            "s2": self.s2,
            "scaler": self.scaler,
            "downscale": self.downscale,
            "upscale": self.upscale,
            "smooth_scaling_enh": self.smooth_scaling_enh,
            "smooth_scaling_legacy": self.smooth_scaling_legacy,
            "smoothing_curve": self.smoothing_curve,
            "early_out": self.early_out,
            "only_one_pass_enh": self.only_one_pass_enh,
            "only_one_pass_legacy": self.only_one_pass_legacy,
            "depth_guard": self.depth_guard,
            "keep_unitary_product": self.keep_unitary_product,
            "align_corners_mode": self.align_corners_mode,
            "recompute_scale_factor_mode": self.recompute_scale_factor_mode,
            "smoothing_mode": self.smoothing_mode,
            "one_pass_mode": self.one_pass_mode,
            "resolution_choice": self.resolution_choice,
            "apply_resolution": self.apply_resolution,
            "adaptive_by_resolution": self.adaptive_by_resolution,
            "adaptive_profile": self.adaptive_profile,
        }


DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "XL · портрет (безопасный)": {
        "category": "XL",
        "algo_mode": "Enhanced (RU+)",
        "resolution_choice": "Портрет: 832x1216",
        "apply_resolution": True,
        "adaptive_by_resolution": True,
        "adaptive_profile": "Сбалансированный",
        "d1": 3,
        "s1": 0.18,
        "d2": 5,
        "s2": 0.32,
        "scaler": "bicubic",
        "downscale": 0.6,
        "upscale": 1.8,
        "smooth_scaling_enh": True,
        "smooth_scaling_legacy": True,
        "smoothing_curve": "Smoothstep",
        "early_out": False,
        "only_one_pass_enh": True,
        "only_one_pass_legacy": True,
        "keep_unitary_product": True,
    },
    "SD15 · Legacy Old Style": {
        "category": "SD15",
        "algo_mode": "Legacy (Original)",
        "resolution_choice": "— не применять —",
        "apply_resolution": False,
        "adaptive_by_resolution": False,
        "d1": 3,
        "s1": 0.15,
        "d2": 4,
        "s2": 0.30,
        "scaler": "bicubic",
        "downscale": 0.5,
        "upscale": 2.0,
        "smooth_scaling_enh": True,
        "smooth_scaling_legacy": True,
        "early_out": False,
        "only_one_pass_enh": True,
        "only_one_pass_legacy": True,
    },
}


class PresetManager:
    def __init__(self) -> None:
        self._cache: Dict[str, HiresPreset] = {}
        self.reload()

    def reload(self) -> None:
        raw = _load_presets()
        if raw is None:
            raw = {}
        if not raw:
            raw = {name: pdata.copy() for name, pdata in DEFAULT_PRESETS.items()}

        self._cache.clear()
        for name, data in raw.items():
            base = HiresPreset().to_dict()
            if isinstance(data, dict):
                base.update(data or {})
            try:
                self._cache[str(name)] = HiresPreset(**base)
            except Exception:
                continue

    def _save(self) -> None:
        raw = {name: preset.to_dict() for name, preset in self._cache.items()}
        _save_presets(raw)

    def names(self) -> List[str]:
        return sorted(self._cache.keys())

    def get(self, name: str) -> Optional[HiresPreset]:
        return self._cache.get(name)

    def upsert(self, name: str, preset: HiresPreset) -> None:
        self._cache[name] = preset
        self._save()

    def delete(self, name: str) -> None:
        if name in self._cache:
            del self._cache[name]
            self._save()

    def categories(self) -> List[str]:
        cats = {(p.category or "Общие") for p in self._cache.values()}
        return sorted(cats) if cats else []

    def names_for_category(self, category: Optional[str]) -> List[str]:
        if not category or category == "Все":
            return self.names()
        cat = category or "Общие"
        return sorted(
            name for name, preset in self._cache.items()
            if (preset.category or "Общие") == cat
        )


class Scaler(torch.nn.Module):
    def __init__(
        self,
        scale: float,
        block: torch.nn.Module,
        scaler: str,
        align_mode: str = "false",
        recompute_mode: str = "false",
    ) -> None:
        super().__init__()
        self.scale: float = float(scale)
        self.block: torch.nn.Module = block
        self.scaler: str = _safe_mode(scaler)
        self.align_mode: str = _norm_mode_choice(align_mode, "false")
        self.recompute_mode: str = _norm_mode_choice(recompute_mode, "false")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.scale == 1.0:
            return self.block(x, *args, **kwargs)

        h, w = x.shape[-2:]
        new_h = max(1, int(h * self.scale))
        new_w = max(1, int(w * self.scale))
        # Keep the effective scale factors aligned to the clamped integer output
        # sizes so that "recompute_scale_factor" applies to the actual resize
        # ratio instead of the raw user input.
        scale_factor = (new_h / h, new_w / w)

        align_corners = None
        if self.scaler in {"bilinear", "bicubic", "linear", "trilinear"}:
            if self.align_mode == "true":
                align_corners = True
            elif self.align_mode == "false":
                align_corners = False
            else:
                align_corners = None

        recompute_scale_factor = None
        if self.recompute_mode == "true":
            recompute_scale_factor = True
        elif self.recompute_mode == "false":
            recompute_scale_factor = False

        x_scaled = F.interpolate(
            x,
            scale_factor=scale_factor,
            mode=self.scaler,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
        )

        out = self.block(x_scaled, *args, **kwargs)
        return out


class KohyaHiresFix(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.config: DictConfig = DictConfig(_load_yaml(CONFIG_PATH, {}))
        self.disable: bool = False
        self.step_limit: int = 0
        self.infotext_fields = []
        self._cb_registered: bool = False

        # Debug features
        self.debug_mode: bool = False
        self.debug_log: List[str] = []
        self.debug_log_max_size: int = 100

    def title(self) -> str:
        return "Kohya Hires.fix · Unified"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def _append_debug(self, msg: str) -> None:
        self.debug_log.append(str(msg))
        if len(self.debug_log) > self.debug_log_max_size:
            self.debug_log = self.debug_log[-self.debug_log_max_size :]

    @staticmethod
    def _unwrap_all(model) -> None:
        if not model:
            return
        # Protection against None blocks
        for i, b in enumerate(getattr(model, "input_blocks", [])):
            if b is not None and isinstance(b, Scaler):
                model.input_blocks[i] = b.block
        for i, b in enumerate(getattr(model, "output_blocks", [])):
            if b is not None and isinstance(b, Scaler):
                model.output_blocks[i] = b.block

    @staticmethod
    def _map_output_index(model, in_idx: int, early_out: bool) -> Optional[int]:
        outs = getattr(model, "output_blocks", None)
        if not outs:
            return None
        n = len(outs)
        if early_out:
            return max(0, min(int(in_idx), n - 1))
        mirror = (n - 1) - int(in_idx)
        return max(0, min(mirror, n - 1))

    def ui(self, is_img2img: bool):
        self.infotext_fields = []
        pm = PresetManager()
        cfg = self.config

        # Load defaults
        last_algo_mode = cfg.get("algo_mode", "Enhanced (RU+)")
        last_resolution_choice = cfg.get("resolution_choice", RESOLUTION_CHOICES[0])
        last_apply_resolution = cfg.get("apply_resolution", False)
        last_adaptive_by_resolution = cfg.get("adaptive_by_resolution", True)
        last_adaptive_profile = cfg.get("adaptive_profile", "Сбалансированный")
        def _coerce_int(val, default: int) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return int(default)

        def _coerce_float(val, default: float) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        last_s1 = _coerce_float(cfg.get("s1", 0.15), 0.15)
        last_s2 = _coerce_float(cfg.get("s2", 0.30), 0.30)
        last_d1 = _coerce_int(cfg.get("d1", 3), 3)
        last_d2 = _coerce_int(cfg.get("d2", 4), 4)
        last_scaler = cfg.get("scaler", "bicubic")
        last_downscale = _coerce_float(cfg.get("downscale", 0.5), 0.5)
        last_upscale = _coerce_float(cfg.get("upscale", 2.0), 2.0)
        last_smooth_enh = cfg.get("smooth_scaling_enh", True)
        last_smooth_leg = cfg.get("smooth_scaling_legacy", True)
        last_only_enh = cfg.get("only_one_pass_enh", True)
        last_only_leg = cfg.get("only_one_pass_legacy", True)
        last_depth_guard = cfg.get("depth_guard", True)
        last_smoothing_curve = cfg.get("smoothing_curve", "Линейная")
        last_early_out = cfg.get("early_out", False)
        last_keep1 = cfg.get("keep_unitary_product", False)
        last_align_mode = cfg.get("align_corners_mode", "False")
        last_recompute_mode = cfg.get("recompute_scale_factor_mode", "False")
        last_smoothing_mode = cfg.get("smoothing_mode", "Авто (по алгоритму)")
        last_one_pass_mode = cfg.get("one_pass_mode", "Авто (по алгоритму)")
        last_simple_mode = cfg.get("simple_mode", True)
        last_stop_preview_enabled = cfg.get("stop_preview_enabled", False)
        last_stop_preview_steps = _coerce_int(cfg.get("stop_preview_steps", 30), 30)

        # Legacy fallback
        legacy_smooth = cfg.get("smooth_scaling", None)
        if legacy_smooth is not None:
            last_smooth_enh = bool(legacy_smooth)
            last_smooth_leg = bool(legacy_smooth)
        legacy_one = cfg.get("only_one_pass", None)
        if legacy_one is not None:
            last_only_enh = bool(legacy_one)
            last_only_leg = bool(legacy_one)

        is_enhanced = (last_algo_mode == "Enhanced (RU+)")

        def _format_stop_preview_text(total_steps: int, s1_v: float, s2_v: float) -> str:
            total = max(1, int(total_steps))
            lines = [f"Всего шагов (Sampling Steps): **{total}**"]

            def _line(label: str, ratio: float) -> str:
                safe_ratio = max(0.0, float(ratio))
                if safe_ratio <= 0:
                    return f"{label}: значение 0 → эффект не применяется"
                stop_step = max(0, math.ceil(total * safe_ratio))
                return f"{label}: эффект прекращается на шаге **{min(total, stop_step)}** (s={safe_ratio:.2f})"

            lines.append(_line("Пара 1 (s1)", s1_v))
            lines.append(_line("Пара 2 (s2)", s2_v))
            return "\n".join(lines)

        with gr.Accordion(label="Kohya Hires.fix", open=False):
            
            # --- Row 1: Enable & Mode & Status ---
            with gr.Row():
                enable = gr.Checkbox(label="Включить расширение", value=False)
                algo_mode = gr.Radio(
                    choices=["Enhanced (RU+)", "Legacy (Original)"],
                    value=last_algo_mode,
                    label="Алгоритм работы / Algorithm Mode",
                )
                status_indicator = gr.Markdown(
                    "🔴 **Отключено**",
                    elem_classes=["status-indicator"]
                )

            # --- Row 2: Quick Presets ---
            with gr.Row():
                gr.Markdown("**⚡ Быстрые пресеты:**")
                btn_quick_safe = gr.Button("🛡️ Безопасный", size="sm", variant="secondary")
                btn_quick_balanced = gr.Button("⚖️ Сбалансированный", size="sm", variant="secondary")
                btn_quick_aggressive = gr.Button("🔥 Агрессивный", size="sm", variant="secondary")

            # --- Row 3: Simple Mode Toggle ---
            with gr.Row():
                simple_mode = gr.Checkbox(
                    label="Простой режим (скрыть продвинутые настройки)",
                    value=last_simple_mode,
                )

            # --- Group: Resolutions ---
            with gr.Group():
                gr.Markdown("**Предустановленные разрешения**")
                with gr.Row():
                    resolution_choice = gr.Dropdown(
                        choices=RESOLUTION_CHOICES,
                        value=last_resolution_choice,
                        label="Выбрать разрешение",
                    )
                    apply_resolution = gr.Checkbox(
                        label="Применить разрешение к width/height",
                        value=last_apply_resolution,
                    )

            # --- Group: Base Parameters ---
            with gr.Group():
                gr.Markdown("**Базовые параметры hires.fix**")
                with gr.Row():
                    s1 = gr.Slider(0.0, 1.0, step=0.01, label="Остановить на (доля шага) — Пара 1", value=last_s1,
                                   info="На какой доле шагов (0.0-1.0) начать применять downscale для первой пары блоков")
                    d1 = gr.Slider(1, 10, step=1, label="Глубина блока — Пара 1", value=last_d1,
                                   info="Индекс блока UNet (1-10). Меньше = раньше в сети")
                with gr.Row():
                    s2 = gr.Slider(0.0, 1.0, step=0.01, label="Остановить на (доля шага) — Пара 2", value=last_s2,
                                   info="На какой доле шагов (0.0-1.0) начать применять downscale для второй пары блоков")
                    d2 = gr.Slider(1, 10, step=1, label="Глубина блока — Пара 2", value=last_d2,
                                   info="Индекс блока UNet (1-10). Меньше = раньше в сети")

                with gr.Row():
                    stop_preview_toggle = gr.Checkbox(
                        label="Показывать визуализацию шага остановки",
                        value=last_stop_preview_enabled,
                        info="Вспомогательный расчёт шага, на котором прекращается эффект для выбранных s1/s2",
                    )
                    stop_preview_steps = gr.Slider(
                        1, 200, step=1, label="Всего шагов (Sampling Steps)", value=last_stop_preview_steps,
                        visible=last_stop_preview_enabled,
                        info="Общее число шагов семплера для расчёта шага остановки",
                    )
                stop_preview_md = gr.Markdown(
                    value=_format_stop_preview_text(last_stop_preview_steps, last_s1, last_s2) if last_stop_preview_enabled else "",
                    visible=last_stop_preview_enabled,
                )

                with gr.Row():
                    depth_guard = gr.Checkbox(
                        label="Автокоррекция глубины блоков",
                        value=last_depth_guard,
                        info="Ограничивает выбранные индексы допустимым диапазоном модели и сортирует d1/d2 при необходимости",
                    )

                with gr.Row():
                    scaler = gr.Dropdown(
                        choices=["bicubic", "bilinear", "nearest", "nearest-exact"],
                        label="Режим интерполяции слоя",
                        value=last_scaler,
                    )
                    downscale = gr.Slider(0.1, 1.0, step=0.05, label="Коэффициент даунскейла (вход)", value=last_downscale,
                                          info="Уменьшение размера входного тензора. 0.5 = половина размера")
                    upscale = gr.Slider(1.0, 4.0, step=0.1, label="Коэффициент апскейла (выход)", value=last_upscale,
                                        info="Увеличение размера на выходе. Обычно = 1/downscale")

                with gr.Row():
                    smooth_scaling_enh = gr.Checkbox(
                        label="Плавное изменение масштаба (Enhanced)", value=last_smooth_enh, visible=is_enhanced)
                    smooth_scaling_legacy = gr.Checkbox(
                        label="Плавное изменение масштаба (Legacy old)", value=last_smooth_leg, visible=not is_enhanced)
                    smoothing_mode_select = gr.Dropdown(
                        choices=["Авто (по алгоритму)", "Использовать Enhanced", "Использовать Legacy old"],
                        value=last_smoothing_mode,
                        label="Использовать логику сглаживания",
                        info="Позволяет включить сглаживание другого режима (например, Legacy сглаживание в Enhanced)."
                    )
                    smoothing_curve = gr.Dropdown(
                        choices=["Линейная", "Smoothstep"], value=last_smoothing_curve,
                        label="Кривая сглаживания", visible=is_enhanced)
                    keep_unitary_product = gr.Checkbox(
                        label="Сохранять суммарный масштаб = 1", value=last_keep1, visible=is_enhanced,
                        info="Автоматически корректирует upscale так, чтобы down*up=1")

                with gr.Row():
                    early_out = gr.Checkbox(label="Ранний апскейл (Early Out)", value=last_early_out)
                    only_one_pass_enh = gr.Checkbox(
                        label="Только один проход (Enhanced)", value=last_only_enh, visible=is_enhanced)
                    only_one_pass_legacy = gr.Checkbox(
                        label="Только один проход (Legacy old)", value=last_only_leg, visible=not is_enhanced)
                    one_pass_mode_select = gr.Dropdown(
                        choices=["Авто (по алгоритму)", "Использовать Enhanced", "Использовать Legacy old"],
                        value=last_one_pass_mode,
                        label="Использовать логику одного прохода",
                        info="Позволяет применять алгоритм одного прохода из другого режима."
                    )

                # Validation Warnings
                with gr.Row():
                    param_warnings = gr.Markdown("", elem_classes=["warning-box"])

            # --- Group: Advanced Settings (Hidden in Simple Mode) ---
            with gr.Group(visible=not last_simple_mode) as advanced_group:
                with gr.Group():
                    gr.Markdown("**Интерполяция (только Enhanced)**")
                    with gr.Row():
                        align_corners_mode = gr.Dropdown(
                            choices=["False", "True", "Авто"], value=last_align_mode,
                            label="align_corners режим", visible=is_enhanced)
                        recompute_scale_factor_mode = gr.Dropdown(
                            choices=["False", "True", "Авто"], value=last_recompute_mode,
                            label="recompute_scale_factor режим", visible=is_enhanced)

                with gr.Group():
                    gr.Markdown("**Адаптация под разрешение**")
                    with gr.Row():
                        adaptive_by_resolution = gr.Checkbox(
                            label="Адаптировать параметры под текущее разрешение",
                            value=last_adaptive_by_resolution,
                        )
                        adaptive_profile = gr.Dropdown(
                            choices=["Консервативный", "Сбалансированный", "Агрессивный"],
                            value=last_adaptive_profile,
                            label="Профиль адаптации",
                        )
                    preview_md = gr.Markdown("Нажмите кнопку ниже для предпросмотра параметров.")
                    btn_preview = gr.Button("Показать рассчитанные параметры", variant="secondary")
                
                # Debug Section
                with gr.Group():
                    gr.Markdown("**🐛 Отладка**")
                    with gr.Row():
                        debug_mode = gr.Checkbox(label="Режим отладки (логировать шаги)", value=False)
                    debug_output = gr.Textbox(
                        label="Лог последней генерации", interactive=False, lines=6, max_lines=15
                    )
                    btn_clear_log = gr.Button("Очистить лог", size="sm")

                # Presets Section
                with gr.Group():
                    gr.Markdown("**Пресеты**")
                    with gr.Row():
                        preset_category_filter = gr.Dropdown(
                            choices=["Все"] + pm.categories(), value="Все", label="Категория (фильтр)")
                        preset_select = gr.Dropdown(
                            choices=pm.names_for_category(None), value=None, label="Выбрать пресет")

                    with gr.Row():
                        preset_name = gr.Textbox(label="Имя пресета", placeholder="имя...", value="")
                        preset_category_input = gr.Textbox(label="Категория", placeholder="Общие...", value="Общие")

                    with gr.Row():
                        btn_save = gr.Button("Сохранить", variant="primary")
                        btn_load = gr.Button("Загрузить")
                        btn_delete = gr.Button("Удалить", variant="stop")
                    preset_status = gr.Markdown("")
                
                # Export/Import Section
                with gr.Group():
                    gr.Markdown("**💾 Экспорт/Импорт настроек**")
                    with gr.Row():
                        btn_export_config = gr.Button("📤 Экспорт в JSON", variant="secondary")
                        btn_import_config = gr.Button("📥 Импорт из JSON", variant="secondary")
                    config_json = gr.Textbox(
                        label="JSON конфигурация", lines=3, max_lines=10, 
                        placeholder='{"version": "2.4", ...}'
                    )
                    import_status = gr.Markdown("")

            # --- Accordion: Help ---
            with gr.Accordion("ℹ️ Справка и советы", open=False):
                gr.Markdown("""
                  ### 📖 Описание параметров:
                  **Базовые:**
                  - **s1, s2**: Доля шагов (0.0-1.0), когда начинать применять эффект.
                  - **s1, s2**: Доля шагов (0.0-0.5), когда начинать применять эффект.
                  - **d1, d2**: Глубина блоков UNet (1-10). Меньшие значения = ранние слои.
                  - **Автокоррекция глубины**: автоматически ограничивает индексы доступными блоками и при необходимости упорядочивает d1 и d2.
                  - **downscale**: Уменьшение размера на входе. 0.5 = половина размера.
                
                **Режимы:**
                - **Enhanced (RU+)**: Современный алгоритм с плавным масштабированием.
                - **Legacy (Original)**: Классический режим.
                
                ### 🎯 Рекомендуемые настройки:
                **SDXL портреты:** s1=0.18, d1=3, s2=0.32, d2=5, down=0.6, up=1.8 (Smoothstep)
                **SD1.5 квадраты:** s1=0.15, d1=3, s2=0.30, d2=4, down=0.5, up=2.0 (Linear)
                """)

            # --- UI Logic ---

            # 1. Validation Logic
            def _validate_params(d1_v, d2_v, s1_v, s2_v, down_v, up_v, keep1):
                def _num_or_none(func, val):
                    try:
                        return func(val)
                    except (TypeError, ValueError):
                        return None

                d1_i = _num_or_none(int, d1_v)
                d2_i = _num_or_none(int, d2_v)
                s1_f = _num_or_none(float, s1_v)
                s2_f = _num_or_none(float, s2_v)
                down_f = _num_or_none(float, down_v)
                up_f = _num_or_none(float, up_v)

                if None in (d1_i, d2_i, s1_f, s2_f, down_f, up_f):
                    return "⚠️ Заполните все значения: одно или несколько полей не распознаны"

                warnings = []
                if d1_i == d2_i and abs(s1_f - s2_f) > 0.01:
                    warnings.append("⚠️ **d1 == d2**, но **s1 ≠ s2**: будет использован max(s1, s2)")
                if s1_f > s2_f:
                    warnings.append("⚠️ **s1 > s2**: параметры будут автоматически скорректированы")
                if s1_f == 0 and s2_f == 0:
                    warnings.append("⚠️ **s1 и s2 равны 0**: эффект не применится")
                if keep1 and abs(down_f * up_f - 1.0) > 0.1:
                    warnings.append(f"ℹ️ down×up будет скорректирован до 1.0 (сейчас {down_f*up_f:.2f})")
                return "\n".join(warnings) if warnings else "✅ **Параметры корректны**"

            for param in [d1, d2, s1, s2, downscale, upscale, keep_unitary_product]:
                param.change(_validate_params,
                             inputs=[d1, d2, s1, s2, downscale, upscale, keep_unitary_product],
                             outputs=[param_warnings])

            def _render_stop_preview(enabled, total_steps, s1_v, s2_v):
                if not enabled:
                    return gr.update(visible=False, value="")
                return gr.update(visible=True, value=_format_stop_preview_text(total_steps, s1_v, s2_v))

            def _toggle_stop_preview(enabled, total_steps, s1_v, s2_v):
                return gr.update(visible=enabled), _render_stop_preview(enabled, total_steps, s1_v, s2_v)

            stop_preview_toggle.change(
                _toggle_stop_preview,
                inputs=[stop_preview_toggle, stop_preview_steps, s1, s2],
                outputs=[stop_preview_steps, stop_preview_md],
            )

            for trigger in (stop_preview_steps, s1, s2):
                trigger.change(
                    _render_stop_preview,
                    inputs=[stop_preview_toggle, stop_preview_steps, s1, s2],
                    outputs=[stop_preview_md],
                )

            # 2. Status Indicator
            def _update_status(enabled: bool, mode: str):
                if not enabled: return "🔴 **Отключено**"
                icon = "🟢" if mode == "Enhanced (RU+)" else "🟡"
                return f"{icon} **Активен: {mode}**"

            enable.change(_update_status, [enable, algo_mode], [status_indicator])
            algo_mode.change(_update_status, [enable, algo_mode], [status_indicator])

            # 3. Quick Presets Logic
            def _apply_quick_preset(preset_type: str):
                presets = {
                    "safe": {"s1": 0.15, "s2": 0.25, "d1": 3, "d2": 4, "down": 0.6, "up": 1.8, "se": True, "sl": True, "c": "Линейная", "k": True},
                    "balanced": {"s1": 0.18, "s2": 0.32, "d1": 3, "d2": 5, "down": 0.5, "up": 2.0, "se": True, "sl": True, "c": "Smoothstep", "k": True},
                    "aggressive": {"s1": 0.22, "s2": 0.38, "d1": 4, "d2": 6, "down": 0.4, "up": 2.5, "se": True, "sl": True, "c": "Smoothstep", "k": False},
                }
                p = presets.get(preset_type, presets["balanced"])
                return (p["s1"], p["s2"], p["d1"], p["d2"], p["down"], p["up"], p["se"], p["sl"], p["c"], p["k"])

            btn_quick_safe.click(lambda: _apply_quick_preset("safe"), 
                outputs=[s1, s2, d1, d2, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_curve, keep_unitary_product])
            btn_quick_balanced.click(lambda: _apply_quick_preset("balanced"), 
                outputs=[s1, s2, d1, d2, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_curve, keep_unitary_product])
            btn_quick_aggressive.click(lambda: _apply_quick_preset("aggressive"), 
                outputs=[s1, s2, d1, d2, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_curve, keep_unitary_product])

            # 4. Simple Mode Toggle (Optimized)
            def _toggle_simple_mode(is_simple: bool, mode: str):
                is_enh = (mode == "Enhanced (RU+)")
                return (gr.update(visible=not is_simple), gr.update(visible=is_enh), gr.update(visible=is_enh))

            simple_mode.change(_toggle_simple_mode, inputs=[simple_mode, algo_mode], 
                               outputs=[advanced_group, align_corners_mode, recompute_scale_factor_mode])

            # 5. Algorithm Visibility Toggle
            def _toggle_algo_vis(mode: str):
                is_enh = (mode == "Enhanced (RU+)")
                return (
                    gr.update(visible=is_enh), gr.update(visible=not is_enh), gr.update(visible=is_enh),
                    gr.update(visible=is_enh), gr.update(visible=is_enh), gr.update(visible=is_enh),
                    gr.update(visible=is_enh), gr.update(visible=not is_enh)
                )

            algo_mode.change(_toggle_algo_vis, inputs=[algo_mode],
                outputs=[smooth_scaling_enh, smooth_scaling_legacy, smoothing_curve, keep_unitary_product,
                         align_corners_mode, recompute_scale_factor_mode, only_one_pass_enh, only_one_pass_legacy])

            # 6. Presets Logic
            def _update_preset_list_for_category(cat: str):
                pm.reload()
                return gr.update(choices=pm.names_for_category(cat), value=None)

            preset_category_filter.change(_update_preset_list_for_category, inputs=[preset_category_filter], outputs=[preset_select])

            def _save_preset_cb(name, cat_in, cat_filt, mode, d1_v, d2_v, depth_guard_v, s1_v, s2_v, scl, dw, up, sm_enh, sm_leg, sm_sel, sm_c, eo, one_enh, one_leg, one_sel, k1, al, rc, res, app, ad, ad_p):
                name = (name or "").strip()
                if not name: return gr.update(), gr.update(), "⚠️ Имя?"
                cat = (cat_in or "").strip() or (cat_filt if cat_filt != "Все" else "Общие")
                base = HiresPreset().to_dict()
                base.update({
                    "category": cat, "algo_mode": mode, "d1": int(d1_v), "d2": int(d2_v), "depth_guard": bool(depth_guard_v), "s1": float(s1_v), "s2": float(s2_v),
                    "scaler": str(scl), "downscale": float(dw), "upscale": float(up),
                    "smooth_scaling_enh": bool(sm_enh), "smooth_scaling_legacy": bool(sm_leg), "smoothing_mode": str(sm_sel),
                    "smoothing_curve": str(sm_c), "early_out": bool(eo),
                    "only_one_pass_enh": bool(one_enh), "only_one_pass_legacy": bool(one_leg), "one_pass_mode": str(one_sel),
                    "keep_unitary_product": bool(k1), "align_corners_mode": str(al), "recompute_scale_factor_mode": str(rc),
                    "resolution_choice": str(res), "apply_resolution": bool(app), "adaptive_by_resolution": bool(ad), "adaptive_profile": str(ad_p),
                })
                pm.upsert(name, HiresPreset(**base))
                cats = ["Все"] + pm.categories()
                return gr.update(choices=cats, value=cat), gr.update(choices=pm.names_for_category(cat), value=name), f"✅ Сохранён «{name}»."

            def _load_preset_cb(name):
                name = (name or "").strip()
                pm.reload()
                preset = pm.get(name)
                if not preset: return (*[gr.update()]*26, f"⚠️ Не найден: {name}")
                p = preset.to_dict()
                return (
                    p.get("algo_mode", "Enhanced (RU+)").strip(), int(p.get("d1", 3)), int(p.get("d2", 4)), bool(p.get("depth_guard", True)),
                    float(p.get("s1", 0.15)), float(p.get("s2", 0.30)),
                    str(p.get("scaler", "bicubic")), float(p.get("downscale", 0.5)), float(p.get("upscale", 2.0)),
                    bool(p.get("smooth_scaling_enh", True)), bool(p.get("smooth_scaling_legacy", True)), str(p.get("smoothing_mode", "Авто (по алгоритму)")),
                    str(p.get("smoothing_curve", "Линейная")), bool(p.get("early_out", False)),
                    bool(p.get("only_one_pass_enh", True)), bool(p.get("only_one_pass_legacy", True)), str(p.get("one_pass_mode", "Авто (по алгоритму)")),
                    bool(p.get("keep_unitary_product", False)), str(p.get("align_corners_mode", "False")), str(p.get("recompute_scale_factor_mode", "False")),
                    str(p.get("resolution_choice", RESOLUTION_CHOICES[0])), bool(p.get("apply_resolution", False)),
                    bool(p.get("adaptive_by_resolution", True)), str(p.get("adaptive_profile", "Сбалансированный")),
                    gr.update(value=name), gr.update(value=p.get("category", "Общие")), f"✅ Загружен «{name}»."
                )

            def _delete_preset_cb(name, cat_filt):
                pm.delete(name)
                cats = ["Все"] + pm.categories()
                return gr.update(choices=cats, value=cat_filt), gr.update(choices=pm.names_for_category(cat_filt), value=None), f"🗑️ Удалён «{name}»."

            btn_save.click(_save_preset_cb, inputs=[preset_name, preset_category_input, preset_category_filter, algo_mode, d1, d2, depth_guard, s1, s2, scaler, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select, keep_unitary_product, align_corners_mode, recompute_scale_factor_mode, resolution_choice, apply_resolution, adaptive_by_resolution, adaptive_profile], outputs=[preset_category_filter, preset_select, preset_status])
            btn_load.click(_load_preset_cb, inputs=[preset_select], outputs=[algo_mode, d1, d2, depth_guard, s1, s2, scaler, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select, keep_unitary_product, align_corners_mode, recompute_scale_factor_mode, resolution_choice, apply_resolution, adaptive_by_resolution, adaptive_profile, preset_name, preset_category_input, preset_status])
            btn_delete.click(_delete_preset_cb, inputs=[preset_select, preset_category_filter], outputs=[preset_category_filter, preset_select, preset_status])

            # 7. Preview
            def _preview_cb(res_v, adapt_v, adapt_prof_v, s1_v, s2_v, d1_v, d2_v, down_v, up_v, keep1_v, mode_v):
                wh = parse_resolution_label(res_v)
                w, h = wh if wh else (1024, 1024)
                if adapt_v:
                    try: u_s1, u_s2, u_d1, u_d2, u_down, u_up = _compute_adaptive_params(w, h, adapt_prof_v, s1_v, s2_v, d1_v, d2_v, down_v, up_v, keep1_v)
                    except: u_s1, u_s2, u_d1, u_d2, u_down, u_up = s1_v, s2_v, d1_v, d2_v, down_v, up_v
                else: u_s1, u_s2, u_d1, u_d2, u_down, u_up = s1_v, s2_v, d1_v, d2_v, down_v, up_v
                return f"**Mode:** {mode_v}\n**Res:** {w}x{h}\n**S1:** {u_s1:.2f}, **D1:** {u_d1}\n**S2:** {u_s2:.2f}, **D2:** {u_d2}\n**Down:** {u_down:.2f}, **Up:** {u_up:.2f}"

            btn_preview.click(_preview_cb, inputs=[resolution_choice, adaptive_by_resolution, adaptive_profile, s1, s2, d1, d2, downscale, upscale, keep_unitary_product, algo_mode], outputs=[preview_md])

            # 8. Debug Logic
            def _clear_debug_log():
                self.debug_log.clear()
                return ""
            btn_clear_log.click(_clear_debug_log, outputs=[debug_output])

            # 9. Export/Import Logic
            def _export_all_config(*params):
                config = {
                    "version": "2.4", "enable": params[0], "simple_mode": params[1], "algo_mode": params[2],
                    "only_one_pass_enh": params[3], "only_one_pass_legacy": params[4], "one_pass_mode": params[5],
                    "d1": params[6], "d2": params[7], "depth_guard": params[8], "s1": params[9], "s2": params[10],
                    "stop_preview_enabled": params[11], "stop_preview_steps": params[12], "scaler": params[13],
                    "downscale": params[14], "upscale": params[15],
                    "smooth_scaling_enh": params[16], "smooth_scaling_legacy": params[17], "smoothing_mode": params[18], "smoothing_curve": params[19],
                    "early_out": params[20], "keep_unitary_product": params[21], "align_corners_mode": params[22],
                    "recompute_scale_factor_mode": params[23], "resolution_choice": params[24], "apply_resolution": params[25],
                    "adaptive_by_resolution": params[26], "adaptive_profile": params[27],
                }
                return json.dumps(config, indent=2, ensure_ascii=False)

            def _import_all_config(json_str):
                try:
                    config = json.loads(json_str)
                    return (
                        gr.update(value=config.get("enable", False)), gr.update(value=config.get("simple_mode", True)),
                        gr.update(value=config.get("algo_mode", "Enhanced (RU+)")), gr.update(value=config.get("only_one_pass_enh", True)),
                        gr.update(value=config.get("only_one_pass_legacy", True)), gr.update(value=config.get("one_pass_mode", "Авто (по алгоритму)")), gr.update(value=config.get("d1", 3)),
                        gr.update(value=config.get("d2", 4)), gr.update(value=config.get("depth_guard", True)), gr.update(value=config.get("s1", 0.15)), gr.update(value=config.get("s2", 0.30)),
                        gr.update(value=config.get("stop_preview_enabled", False)), gr.update(value=int(config.get("stop_preview_steps", 30))),
                        gr.update(value=config.get("scaler", "bicubic")), gr.update(value=config.get("downscale", 0.5)),
                        gr.update(value=config.get("upscale", 2.0)), gr.update(value=config.get("smooth_scaling_enh", True)),
                        gr.update(value=config.get("smooth_scaling_legacy", True)), gr.update(value=config.get("smoothing_mode", "Авто (по алгоритму)")), gr.update(value=config.get("smoothing_curve", "Линейная")),
                        gr.update(value=config.get("early_out", False)), gr.update(value=config.get("keep_unitary_product", False)),
                        gr.update(value=config.get("align_corners_mode", "False")), gr.update(value=config.get("recompute_scale_factor_mode", "False")),
                        gr.update(value=config.get("resolution_choice", RESOLUTION_CHOICES[0])), gr.update(value=config.get("apply_resolution", False)),
                        gr.update(value=config.get("adaptive_by_resolution", True)), gr.update(value=config.get("adaptive_profile", "Сбалансированный")),
                        "✅ Настройки импортированы"
                    )
                except Exception as e: return (*[gr.update()]*28, f"❌ Ошибка: {e}")

            all_params_list = [
                enable, simple_mode, algo_mode, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select,
                d1, d2, depth_guard, s1, s2, stop_preview_toggle, stop_preview_steps, scaler, downscale, upscale,
                smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out, keep_unitary_product, align_corners_mode,
                recompute_scale_factor_mode, resolution_choice, apply_resolution, adaptive_by_resolution, adaptive_profile
            ]
            btn_export_config.click(_export_all_config, inputs=all_params_list, outputs=[config_json])
            btn_import_config.click(_import_all_config, inputs=[config_json], outputs=all_params_list + [import_status])

        # Infotext setup
        self.infotext_fields.append((enable, lambda d: d.get("DSHF_s1", False)))
        for k, el in {
            "DSHF_mode": algo_mode, "DSHF_s1": s1, "DSHF_d1": d1, "DSHF_s2": s2, "DSHF_d2": d2,
            "DSHF_scaler": scaler, "DSHF_down": downscale, "DSHF_up": upscale, "DSHF_smooth_enh": smooth_scaling_enh,
            "DSHF_smooth_legacy": smooth_scaling_legacy, "DSHF_early": early_out, "DSHF_one_enh": only_one_pass_enh, "DSHF_one_legacy": only_one_pass_legacy
        }.items():
            self.infotext_fields.append((el, k))

        return all_params_list + [debug_mode]

    def process(
        self, p, enable, simple, algo_mode, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select,
        d1, d2, depth_guard, s1, s2, stop_preview_enabled, stop_preview_steps, scaler, downscale, upscale,
        smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out,
        keep_unitary_product, align_ui, recompute_ui, res_choice, apply_res,
        adapt, adapt_prof, debug_mode_val
    ):
        self.step_limit = 0
        self.debug_mode = debug_mode_val
        self.config = DictConfig({
            "algo_mode": algo_mode, "simple_mode": simple, "s1": s1, "s2": s2, "d1": d1, "d2": d2,
            "depth_guard": depth_guard,
            "stop_preview_enabled": stop_preview_enabled, "stop_preview_steps": stop_preview_steps,
            "scaler": scaler, "downscale": downscale, "upscale": upscale,
            "smooth_scaling_enh": smooth_scaling_enh, "smooth_scaling_legacy": smooth_scaling_legacy,
            "smoothing_mode": smoothing_mode_select, "smoothing_curve": smoothing_curve, "early_out": early_out,
            "only_one_pass_enh": only_one_pass_enh, "only_one_pass_legacy": only_one_pass_legacy, "one_pass_mode": one_pass_mode_select,
            "keep_unitary_product": keep_unitary_product, "align_corners_mode": align_ui,
            "recompute_scale_factor_mode": recompute_ui, "resolution_choice": res_choice,
            "apply_resolution": apply_res, "adaptive_by_resolution": adapt, "adaptive_profile": adapt_prof,
            "debug_mode": debug_mode_val
        })

        if apply_res:
            try:
                wh = parse_resolution_label(res_choice)
                if wh: p.width, p.height = wh
            except Exception as e: print(f"[KohyaHiresFix] Res Error: {e}")

        if not enable or self.disable:
            try: script_callbacks.remove_current_script_callbacks()
            except: pass
            self._cb_registered = False
            try:
                model_container = getattr(p.sd_model, "model", None)
                if model_container and hasattr(model_container, "diffusion_model"):
                    KohyaHiresFix._unwrap_all(model_container.diffusion_model)
            except: pass
            return

        use_s1, use_s2 = _clamp(float(s1), 0.0, 1.0), _clamp(float(s2), 0.0, 1.0)
        use_d1, use_d2 = int(d1), int(d2)
        use_down, use_up = float(downscale), float(upscale)
        depth_guard = bool(depth_guard)

        if adapt:
            try:
                use_s1, use_s2, use_d1, use_d2, use_down, use_up = _compute_adaptive_params(
                    int(getattr(p, "width", 1024)), int(getattr(p, "height", 1024)), adapt_prof,
                    use_s1, use_s2, d1, d2, downscale, upscale, keep_unitary_product
                )
            except: pass

        if use_s1 > use_s2: use_s2 = use_s1

        model_container = getattr(p.sd_model, "model", None)
        if not model_container or not hasattr(model_container, "diffusion_model"): return
        model = model_container.diffusion_model
        
        inp_list = getattr(model, "input_blocks", [])
        out_list = getattr(model, "output_blocks", [])
        max_inp = len(inp_list) - 1
        
        if max_inp < 1 or len(out_list) < 1:
             print(f"[KohyaHiresFix] Model Error: Invalid blocks (max_inp={max_inp}).")
             return

        d1_idx = int(use_d1) - 1
        d2_idx = int(use_d2) - 1
        scaler_mode = _safe_mode(scaler)

        if algo_mode == "Legacy (Original)":
            align_mode, recompute_mode = "false", "false"
        else:
            align_mode = _norm_mode_choice(align_ui, "auto")
            recompute_mode = _norm_mode_choice(recompute_ui, "auto")

        def _select_cross_mode(choice: str, enh_value: bool, legacy_value: bool) -> bool:
            sel = (choice or "").strip().lower()
            if sel.startswith("использовать legacy"):
                return bool(legacy_value)
            if sel.startswith("использовать enhanced"):
                return bool(enh_value)
            return bool(enh_value) if algo_mode == "Enhanced (RU+)" else bool(legacy_value)

        chosen_smoothing = _select_cross_mode(smoothing_mode_select, smooth_scaling_enh, smooth_scaling_legacy)
        chosen_one_pass = _select_cross_mode(one_pass_mode_select, only_one_pass_enh, only_one_pass_legacy)

        if algo_mode == "Enhanced (RU+)":
            use_smooth_enh = chosen_smoothing
            use_one_enh = chosen_one_pass
            use_smooth_legacy = bool(smooth_scaling_legacy)
            use_one_legacy = bool(only_one_pass_legacy)
        else:
            use_smooth_enh = bool(smooth_scaling_enh)
            use_one_enh = bool(only_one_pass_enh)
            use_smooth_legacy = chosen_smoothing
            use_one_legacy = chosen_one_pass

        mapping_notes: list[str] = [
            f"Доступно входных блоков: {len(inp_list)} | выходных: {len(out_list)}",
            "Автокоррекция глубины: включена" if depth_guard else "Автокоррекция глубины: выключена",
        ]

        def _normalize_depth(idx: int, label: str) -> int:
            clamped = max(0, min(int(idx), max_inp))
            if clamped != int(idx):
                mapping_notes.append(
                    f"{label}: {int(idx) + 1} вне диапазона 1-{max_inp + 1} → использован {clamped + 1}"
                )
            return clamped

        if depth_guard:
            d1_idx = _normalize_depth(d1_idx, "d1")
            d2_idx = _normalize_depth(d2_idx, "d2")
            if d2_idx < d1_idx:
                mapping_notes.append("d1 и d2 упорядочены по возрастанию глубины")
                d1_idx, d2_idx = d2_idx, d1_idx
        else:
            d1_idx = _normalize_depth(d1_idx, "d1")
            d2_idx = _normalize_depth(d2_idx, "d2")

        # Report the real mapping of user-provided depths to model blocks so it is easier
        # to diagnose unexpected behavior (e.g. when a chosen depth is clamped).
        mapped_pairs: list[tuple[int, Optional[int]]] = []

        def _map_with_note(label: str, idx: int) -> Optional[int]:
            if idx != int(idx):
                mapping_notes.append(f"{label}: округлено до {idx}")
            out_idx = KohyaHiresFix._map_output_index(model, idx, early_out)
            if out_idx is None:
                mapping_notes.append(f"{label}: выходной блок не найден (input={idx})")
                return None
            if idx >= len(inp_list):
                mapping_notes.append(f"{label}: input {idx + 1} превышает доступные {len(inp_list)} → использован {inp_list and len(inp_list) or 0}")
            if out_idx >= len(out_list):
                mapping_notes.append(f"{label}: output {out_idx + 1} превышает доступные {len(out_list)} → использован {out_list and len(out_list) or 0}")
            mapped_pairs.append((idx, out_idx))
            return out_idx

        d1_out_idx = _map_with_note("d1", d1_idx)
        d2_out_idx = _map_with_note("d2", d2_idx)
        if d1_out_idx is not None and d2_out_idx is not None and d1_out_idx == d2_out_idx:
            mapping_notes.append(f"d1 и d2 указывают на один и тот же выходной блок ({d1_out_idx + 1})")

        if self.debug_mode:
            mapping_repr = ", ".join([f"in {i + 1} → out {(o or 0) + 1}" for i, o in mapped_pairs]) or "нет"
            self._append_debug(f"Mapping: {mapping_repr}")
            for note in mapping_notes:
                self._append_debug(f"⚠️ {note}")

        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            total = max(1, int(params.total_sampling_steps))
            current = params.sampling_step

            try:
                # Debug logging
                if self.debug_mode:
                    msg = f"Step {current}/{total}: mode={algo_mode}"
                    if algo_mode == "Legacy (Original)" and d1_idx < len(model.input_blocks) and isinstance(model.input_blocks[d1_idx], Scaler):
                         msg += f", d1_scale={model.input_blocks[d1_idx].scale:.3f}"
                    self._append_debug(msg)

                # ========================= LEGACY (ORIGINAL) MODE =========================
                if algo_mode == "Legacy (Original)":
                    if use_one_legacy and self.step_limit: return

                    legacy_pairs = [(use_s1, d1_idx), (use_s2, d2_idx)]
                    combined_legacy: dict[int, float] = {}
                    for s_stop, d_idx in legacy_pairs:
                        if s_stop > 0:
                            combined_legacy[d_idx] = max(combined_legacy.get(d_idx, 0.0), s_stop)
                    n_out = len(model.output_blocks)

                    for d_idx, s_stop in combined_legacy.items():
                        if s_stop <= 0: continue

                        # Added explicit validation for legacy loop
                        if d_idx < 0 or d_idx >= len(model.input_blocks): continue

                        if early_out: out_idx = d_idx
                        else: out_idx = n_out - 1 - d_idx
                        if out_idx < 0 or out_idx >= n_out: continue

                        if current < total * s_stop:
                            if not isinstance(model.input_blocks[d_idx], Scaler):
                                model.input_blocks[d_idx] = Scaler(use_down, model.input_blocks[d_idx], scaler_mode, align_mode, recompute_mode)
                                model.output_blocks[out_idx] = Scaler(use_up, model.output_blocks[out_idx], scaler_mode, align_mode, recompute_mode)
                            elif use_smooth_legacy:
                                scale_ratio = current / (total * s_stop)
                                cur_down = min((1.0 - use_down) * scale_ratio + use_down, 1.0)
                                model.input_blocks[d_idx].scale = cur_down
                                model.output_blocks[out_idx].scale = use_up * (use_down / max(1e-6, cur_down))
                            return
                        elif isinstance(model.input_blocks[d_idx], Scaler):
                            # Unwrapping with safety checks
                            model.input_blocks[d_idx] = model.input_blocks[d_idx].block
                            if isinstance(model.output_blocks[out_idx], Scaler):
                                model.output_blocks[out_idx] = model.output_blocks[out_idx].block

                    if use_one_legacy and current > 0 and current >= total * max(use_s1, use_s2):
                        self.step_limit = 1

                # ========================= ENHANCED (RU+) MODE =========================
                else:
                    if use_one_enh and self.step_limit: return
                    combined = {}
                    for s_stop, d_i in ((float(use_s1), d1_idx), (float(use_s2), d2_idx)):
                        if s_stop > 0: combined[d_i] = max(combined.get(d_i, 0.0), s_stop)
                    max_stop = max(combined.values()) if combined else 0.0

                    for d_i, s_stop in combined.items():
                        out_i = KohyaHiresFix._map_output_index(model, d_i, early_out)
                        if out_i is None: continue

                        if current < total * s_stop:
                            if not isinstance(model.input_blocks[d_i], Scaler):
                                model.input_blocks[d_i] = Scaler(use_down, model.input_blocks[d_i], scaler_mode, align_mode, recompute_mode)
                                model.output_blocks[out_i] = Scaler(use_up, model.output_blocks[out_i], scaler_mode, align_mode, recompute_mode)

                            if use_smooth_enh:
                                ratio = float(max(0.0, min(1.0, current / (total * s_stop))))
                                if (smoothing_curve or "").lower().startswith("smooth"): ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                                cur_down = min((1.0 - use_down) * ratio + use_down, 1.0)
                                model.input_blocks[d_i].scale = cur_down
                                if keep_unitary_product: cur_up = 1.0 / max(1e-6, cur_down)
                                else: cur_up = _clamp(use_up * (use_down / max(1e-6, cur_down)), 1.0, 4.0)
                                model.output_blocks[out_i].scale = cur_up
                            else:
                                model.input_blocks[d_i].scale = use_down
                                model.output_blocks[out_i].scale = use_up
                        elif isinstance(model.input_blocks[d_i], Scaler):
                            model.input_blocks[d_i] = model.input_blocks[d_i].block
                            if isinstance(model.output_blocks[out_i], Scaler):
                                model.output_blocks[out_i] = model.output_blocks[out_i].block

                    if use_one_enh and max_stop > 0 and current >= total * max_stop:
                        self.step_limit = 1

            except Exception as e:
                try: KohyaHiresFix._unwrap_all(model)
                except: pass
                try: script_callbacks.remove_current_script_callbacks()
                except: pass
                self._cb_registered = False
                self.disable = True
                print(f"[KohyaHiresFix] Error: {e}")

        if self._cb_registered:
            try: script_callbacks.remove_current_script_callbacks()
            except: pass
        script_callbacks.on_cfg_denoiser(denoiser_callback)
        self._cb_registered = True

        p.extra_generation_params.update({
            "DSHF_mode": algo_mode, "DSHF_s1": use_s1, "DSHF_s2": use_s2, "DSHF_down": use_down, "DSHF_up": use_up, "DSHF_depth_guard": depth_guard
        })

    def postprocess(self, p, processed, *args):
        try:
            model_container = getattr(p.sd_model, "model", None)
            if model_container and hasattr(model_container, "diffusion_model"):
                KohyaHiresFix._unwrap_all(model_container.diffusion_model)
            if self.debug_mode and self.debug_log:
                print(f"[KohyaHiresFix] Log:\n" + "\n".join(self.debug_log[-20:]))
        finally:
            try: _atomic_save_yaml(CONFIG_PATH, OmegaConf.to_container(self.config, resolve=True) or {})
            except: pass
            self._cb_registered = False

    def process_batch(self, p, *args, **kwargs):
        self.step_limit = 0
