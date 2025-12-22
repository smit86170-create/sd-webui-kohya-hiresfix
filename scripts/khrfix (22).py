# kohya_hires_fix_unified_v2.5.3_ultimate.py
# Версия: 2.5.3 (Ultimate: v19 Features + Old Math Fix)
# Совместимость: A1111 / modules.scripts API, PyTorch >= 1.12
#
# Изменения:
# - В основе лежит код версии 2.5.1 (v19) со всеми проверками безопасности и логами.
# - Внедрена логика "Старой математики" (float vs int) для резкости.
# - Внедрена логика "Старого одного прохода" (step vs flag).
# - Добавлены настройки совместимости в UI.

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
    except Exception as e:
        print(f"[KohyaHiresFix] Failed to save config: {e}")


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
        up = min(10.0, 1.0 / max(0.1, down))
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
        self.depth_clamp: bool = True
        self.depth_pair_mode: str = "Авто (по алгоритму)"  # 🆕 режим обработки пар d1/d2

        self.keep_unitary_product: bool = False
        self.align_corners_mode: str = "False"
        self.recompute_scale_factor_mode: str = "False"
        self.smoothing_mode: str = "Авто (по алгоритму)"
        self.one_pass_mode: str = "Авто (по алгоритму)"

        self.resolution_choice: str = RESOLUTION_CHOICES[0]
        self.apply_resolution: bool = False
        self.adaptive_by_resolution: bool = True
        self.adaptive_profile: str = "Сбалансированный"

        # 🆕 НОВЫЕ ФЛАГИ (из v2.5.2)
        self.use_old_float_math: bool = True
        self.use_old_onepass_logic: bool = True
        self.legacy_force_align_false: bool = True
        self.enhanced_safe_upscale_clamp: bool = True

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
            "depth_clamp": self.depth_clamp,  
            "depth_pair_mode": self.depth_pair_mode,
            "keep_unitary_product": self.keep_unitary_product,
            "align_corners_mode": self.align_corners_mode,
            "recompute_scale_factor_mode": self.recompute_scale_factor_mode,
            "smoothing_mode": self.smoothing_mode,
            "one_pass_mode": self.one_pass_mode,
            "resolution_choice": self.resolution_choice,
            "apply_resolution": self.apply_resolution,
            "adaptive_by_resolution": self.adaptive_by_resolution,
            "adaptive_profile": self.adaptive_profile,
            "use_old_float_math": self.use_old_float_math,
            "use_old_onepass_logic": self.use_old_onepass_logic,
            "legacy_force_align_false": self.legacy_force_align_false,
            "enhanced_safe_upscale_clamp": self.enhanced_safe_upscale_clamp,
        }

# КОНЕЦ ЧАСТИ 1

# НАЧАЛО ЧАСТИ 2

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
        "use_old_float_math": True,
        "use_old_onepass_logic": True,
        "legacy_force_align_false": True,
        "enhanced_safe_upscale_clamp": True,
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
        "use_old_float_math": True,
        "use_old_onepass_logic": True,
        "legacy_force_align_false": True,
        "enhanced_safe_upscale_clamp": True,
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
        use_old_float_math: bool = True,  # 🆕 ФЛАГ
    ) -> None:
        super().__init__()
        self.scale: float = float(scale)
        self.block: torch.nn.Module = block
        self.scaler: str = _safe_mode(scaler)
        self.align_mode: str = _norm_mode_choice(align_mode, "false")
        self.recompute_mode: str = _norm_mode_choice(recompute_mode, "false")
        self.use_old_float_math: bool = use_old_float_math # 🆕

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.scale == 1.0:
            return self.block(x, *args, **kwargs)

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

        # 🆕 ГИБРИДНАЯ ЛОГИКА
        if self.use_old_float_math:
            # ✅ OLD (Legacy) - Прямая передача float, без округления размеров
            x_scaled = F.interpolate(
                x,
                scale_factor=self.scale,
                mode=self.scaler,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
            )
        else:
            # ✅ NEW (Safe) - Округление до целых пикселей
            h, w = x.shape[-2:]
            new_h = max(1, int(h * self.scale))
            new_w = max(1, int(w * self.scale))
            scale_factor = (new_h / h, new_w / w)
            
            x_scaled = F.interpolate(
                x,
                scale_factor=scale_factor,
                mode=self.scaler,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
            )

        out = self.block(x_scaled, *args, **kwargs)
        return out

# КОНЕЦ ЧАСТИ 2

# НАЧАЛО ЧАСТИ 3

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
        return "Kohya Hires.fix · Unified (Ult)"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def _append_debug(self, msg: str) -> None:
        self.debug_log.append(str(msg))
        if len(self.debug_log) > self.debug_log_max_size:
            self.debug_log = self.debug_log[-self.debug_log_max_size :]
            
    def _save_config(self) -> None:
        """Автосохранение последней конфигурации в YAML."""
        try:
            data = OmegaConf.to_container(self.config, resolve=True)
            _atomic_save_yaml(CONFIG_PATH, data)
        except Exception as e:
            print(f"[KohyaHiresFix] Failed to autosave YAML config: {e}")

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

        # Load defaults (с поддержкой новых флагов)
        last_algo_mode = cfg.get("algo_mode", "Enhanced (RU+)")
        last_resolution_choice = cfg.get("resolution_choice", RESOLUTION_CHOICES[0])
        last_apply_resolution = cfg.get("apply_resolution", False)
        last_adaptive_by_resolution = cfg.get("adaptive_by_resolution", True)
        last_adaptive_profile = cfg.get("adaptive_profile", "Сбалансированный")
        
        last_s1 = float(cfg.get("s1", 0.15))
        last_s2 = float(cfg.get("s2", 0.30))
        last_d1 = int(cfg.get("d1", 3))
        last_d2 = int(cfg.get("d2", 4))
        last_scaler = cfg.get("scaler", "bicubic")
        last_downscale = float(cfg.get("downscale", 0.5))
        last_upscale = float(cfg.get("upscale", 2.0))
        
        # ФЛАГИ ПО УМОЛЧАНИЮ = True (OLD Logic)
        last_use_old_float_math = cfg.get("use_old_float_math", True) 
        last_use_old_onepass_logic = cfg.get("use_old_onepass_logic", True) 
        last_legacy_force_align_false = cfg.get("legacy_force_align_false", True)
        last_enh_safe_clamp = cfg.get("enhanced_safe_upscale_clamp", True)

        # Legacy fallback logic
        legacy_smooth = cfg.get("smooth_scaling", None)
        last_smooth_enh = bool(legacy_smooth) if legacy_smooth is not None else cfg.get("smooth_scaling_enh", True)
        last_smooth_leg = bool(legacy_smooth) if legacy_smooth is not None else cfg.get("smooth_scaling_legacy", True)
        
        legacy_one = cfg.get("only_one_pass", None)
        last_only_enh = bool(legacy_one) if legacy_one is not None else cfg.get("only_one_pass_enh", True)
        last_only_leg = bool(legacy_one) if legacy_one is not None else cfg.get("only_one_pass_legacy", True)
        
        last_depth_guard = cfg.get("depth_guard", True)
        last_depth_clamp = cfg.get("depth_clamp", True)                  # 🆕
        last_depth_pair_mode = cfg.get("depth_pair_mode", "Авто (по алгоритму)")  # 🆕
        last_smoothing_curve = cfg.get("smoothing_curve", "Линейная")
        last_early_out = cfg.get("early_out", False)
        last_keep1 = cfg.get("keep_unitary_product", False)
        last_align_mode = cfg.get("align_corners_mode", "False")
        last_recompute_mode = cfg.get("recompute_scale_factor_mode", "False")
        last_smoothing_mode = cfg.get("smoothing_mode", "Авто (по алгоритму)")
        last_one_pass_mode = cfg.get("one_pass_mode", "Авто (по алгоритму)")
        last_simple_mode = cfg.get("simple_mode", True)
        last_stop_preview_enabled = cfg.get("stop_preview_enabled", False)
        last_stop_preview_steps = int(cfg.get("stop_preview_steps", 30))

        is_enhanced = (last_algo_mode == "Enhanced (RU+)")

        def _format_stop_preview_text(total_steps: int, s1_v: float, s2_v: float) -> str:
            total = max(1, int(total_steps))
            lines = [f"Всего шагов: **{total}**"]
            def _line(label: str, ratio: float) -> str:
                safe_ratio = max(0.0, float(ratio))
                if safe_ratio <= 0: return f"{label}: выкл"
                stop_step = max(0, math.ceil(total * safe_ratio))
                return f"{label}: стоп на шаге **{min(total, stop_step)}** (s={safe_ratio:.2f})"
            lines.append(_line("Пара 1", s1_v))
            lines.append(_line("Пара 2", s2_v))
            return "\n".join(lines)

        with gr.Accordion(label="Kohya Hires.fix", open=False):
            with gr.Row():
                enable = gr.Checkbox(label="Включить расширение", value=False)
                algo_mode = gr.Radio(choices=["Enhanced (RU+)", "Legacy (Original)"], value=last_algo_mode, label="Алгоритм")
                status_indicator = gr.Markdown("🔴 **Отключено**", elem_classes=["status-indicator"])

            with gr.Row():
                gr.Markdown("**⚡ Быстрые пресеты:**")
                btn_quick_safe = gr.Button("🛡️ Безопасный", size="sm", variant="secondary")
                btn_quick_balanced = gr.Button("⚖️ Сбалансированный", size="sm", variant="secondary")
                btn_quick_aggressive = gr.Button("🔥 Агрессивный", size="sm", variant="secondary")

            with gr.Row():
                simple_mode = gr.Checkbox(label="Простой режим", value=last_simple_mode)

            with gr.Group():
                gr.Markdown("**Базовые параметры**")
                with gr.Row():
                    s1 = gr.Slider(0.0, 1.0, step=0.01, label="Остановить (s1)", value=last_s1)
                    d1 = gr.Slider(1, 10, step=1, label="Глубина (d1)", value=last_d1)
                with gr.Row():
                    s2 = gr.Slider(0.0, 1.0, step=0.01, label="Остановить (s2)", value=last_s2)
                    d2 = gr.Slider(1, 10, step=1, label="Глубина (d2)", value=last_d2)
                with gr.Row():
                    stop_preview_toggle = gr.Checkbox(label="Визуализация остановки", value=last_stop_preview_enabled)
                    stop_preview_steps = gr.Slider(1, 200, step=1, label="Шаги семплера", value=last_stop_preview_steps, visible=last_stop_preview_enabled)
                stop_preview_md = gr.Markdown(value=_format_stop_preview_text(last_stop_preview_steps, last_s1, last_s2) if last_stop_preview_enabled else "", visible=last_stop_preview_enabled)

                with gr.Row():
                    depth_guard = gr.Checkbox(
                        label="Автокоррекция глубины (swap d1/d2)",
                        value=last_depth_guard,
                        info="Если включено — при d2 < d1 они меняются местами, чтобы не ломать карту блоков."
                    )
                    depth_clamp = gr.Checkbox(
                        label="Жёсткий clamp глубины",
                        value=last_depth_clamp,
                        info="ВКЛ: d1/d2 принудительно зажимаются в [0..max]. ВЫКЛ: если индекс вне диапазона — этот depth просто пропускается."
                    )

                # 🆕 ГРУППА СОВМЕСТИМОСТИ
                with gr.Group():
                    gr.Markdown("### 🛠️ Отладка и совместимость (Влияет на математику)")
                    with gr.Row():
                        use_old_float_math = gr.Checkbox(
                            label="🛠️ Использовать \"Старую математику\" (Float)",
                            value=last_use_old_float_math,
                            info="ВКЛ: передает scale_factor напрямую (OLD). ВЫКЛ: с округлением int() (NEW)"
                        )
                        use_old_onepass_logic = gr.Checkbox(
                            label="🛠️ Строгий режим \"Один проход\" (Old Logic)",
                            value=last_use_old_onepass_logic,
                            info="ВКЛ: запоминает номер шага (OLD). ВЫКЛ: использует флаг (NEW)"
                        )
                with gr.Row():
                    legacy_force_align_false = gr.Checkbox(
                        label="Legacy: align_corners/recompute = False",
                        value=last_legacy_force_align_false,
                        info="При включении в Legacy всегда использует align_corners=False и recompute_scale_factor=False (как в старом скрипте)."
                    )
                    enhanced_safe_clamp = gr.Checkbox(
                        label="Enhanced: безопасный clamp Upscale",
                        value=last_enh_safe_clamp,
                        info="При включении ограничивает Upscale (1–4). При выключении ведёт себя как Legacy/olds."
                    )

                with gr.Row():
                    scaler = gr.Dropdown(choices=["bicubic", "bilinear", "nearest", "nearest-exact"], label="Интерполяция", value=last_scaler)
                    downscale = gr.Slider(0.1, 1.0, step=0.05, label="Downscale", value=last_downscale)
                    upscale = gr.Slider(1.0, 4.0, step=0.1, label="Upscale", value=last_upscale)

                with gr.Row():
                    early_out = gr.Checkbox(label="Early Out", value=last_early_out)
                    only_one_pass_enh = gr.Checkbox(
                        label="Только один проход (Enh)",
                        value=last_only_enh,
                        visible=is_enhanced,
                        info="Enhanced: применяет масштаб только до одного стоп-интервала; дальше denoiser идёт как обычно."
                    )
                    only_one_pass_legacy = gr.Checkbox(
                        label="Только один проход (Leg)",
                        value=last_only_leg,
                        visible=not is_enhanced,
                        info="Legacy: строгая логика как в old.py — запоминает шаг и после него больше не трогает блоки."
                    )
                    one_pass_mode_select = gr.Dropdown(
                        choices=["Авто (по алгоритму)", "Использовать Enhanced", "Использовать Legacy old"],
                        value=last_one_pass_mode,
                        label="Логика One Pass",
                        info="Авто: берёт only_one_pass_enh/legacy в зависимости от алгоритма. Остальные режимы принудительно используют соответствующую логику даже в другом алгоритме."
                    )

                with gr.Row():
                    param_warnings = gr.Markdown("", elem_classes=["warning-box"])

# КОНЕЦ ЧАСТИ 3

# НАЧАЛО ЧАСТИ 4 (ИСПРАВЛЕННАЯ ПОЛНАЯ ВЕРСИЯ)

            with gr.Group(visible=not last_simple_mode) as advanced_group:
                with gr.Group():
                    gr.Markdown("**Параметры сглаживания и разрешения**")
                    with gr.Row():
                        smooth_scaling_enh = gr.Checkbox(
                            label="Плавное изменение (Enh)",
                            value=last_smooth_enh,
                            visible=is_enhanced,
                            info="Для Enhanced: плавно двигает Downscale/Upscale от начального значения до стоп-шага."
                        )
                        smooth_scaling_legacy = gr.Checkbox(
                            label="Плавное изменение (Leg)",
                            value=last_smooth_leg,
                            visible=not is_enhanced,
                            info="Для Legacy: аналогично, но повторяет поведение старого скрипта."
                        )
                        smoothing_mode_select = gr.Dropdown(
                            choices=["Авто (по алгоритму)", "Использовать Enhanced", "Использовать Legacy old"],
                            value=last_smoothing_mode,
                            label="Логика сглаживания",
                            info="Авто: берёт Enh/Leg в зависимости от выбранного алгоритма. Остальные пункты принудительно включают соответствующий флаг независимо от алгоритма."
                        )
                        smoothing_curve = gr.Dropdown(choices=["Линейная", "Smoothstep"], value=last_smoothing_curve, label="Кривая", visible=is_enhanced)
                        keep_unitary_product = gr.Checkbox(label="Сохранять масштаб=1", value=last_keep1, visible=is_enhanced)
                    with gr.Row():
                        resolution_choice = gr.Dropdown(choices=RESOLUTION_CHOICES, value=last_resolution_choice, label="Разрешение")
                        apply_resolution = gr.Checkbox(label="Применить к W/H", value=last_apply_resolution)
                        adaptive_by_resolution = gr.Checkbox(
                            label="Адаптация",
                            value=last_adaptive_by_resolution,
                            info="Если включено — автоматически подстраивает s1/s2/d1/d2 и Downscale/Upscale под конечное разрешение (только для Enhanced)."
                        )
                        adaptive_profile = gr.Dropdown(
                            choices=["Консервативный", "Сбалансированный", "Агрессивный"],
                            value=last_adaptive_profile,
                            label="Профиль",
                            visible=last_adaptive_by_resolution,
                        )
                        depth_pair_mode = gr.Dropdown(              # 🆕 новый дропдаун
                        choices=[
                            "Авто (по алгоритму)",
                            "Последовательно (как olds.py)",
                            "Обе глубины одновременно",
                            "Только d1",
                            "Только d2",
                        ],
                        value=last_depth_pair_mode,
                        label="Режим обработки depth-пар",
                        info="Авто: Legacy → как в olds.py (по очереди и return), Enhanced → оба depth одновременно (как сейчас). "
                             "Остальные режимы принудительно задают поведение независимо от алгоритма."
                    )

                with gr.Group():
                    gr.Markdown("**Интерполяция (Advanced)**")
                    with gr.Row():
                        align_corners_mode = gr.Dropdown(choices=["False", "True", "Авто"], value=last_align_mode, label="align_corners", visible=is_enhanced)
                        recompute_scale_factor_mode = gr.Dropdown(choices=["False", "True", "Авто"], value=last_recompute_mode, label="recompute_scale_factor", visible=is_enhanced)
                
                with gr.Group():
                    gr.Markdown("**Пресеты / Импорт / Экспорт / Логи**")
                    with gr.Row():
                        preset_category_filter = gr.Dropdown(choices=["Все"] + pm.categories(), value="Все", label="Категория")
                        preset_select = gr.Dropdown(choices=pm.names_for_category(None), value=None, label="Выбрать пресет")
                        btn_save = gr.Button("Сохранить", variant="primary")
                        btn_load = gr.Button("Загрузить")
                        btn_delete = gr.Button("Удалить", variant="stop")
                    with gr.Row():
                         preset_name = gr.Textbox(placeholder="Имя пресета...", show_label=False)
                         preset_category_input = gr.Textbox(placeholder="Категория...", show_label=False)
                    preset_status = gr.Markdown("")
                    with gr.Row():
                        btn_export_config = gr.Button("Экспорт JSON")
                        btn_import_config = gr.Button("Импорт JSON")
                    config_json = gr.Textbox(label="JSON конфигурация", lines=3)
                    import_status = gr.Markdown("")
                    with gr.Row():
                        debug_mode = gr.Checkbox(label="Debug Mode (Log Steps)", value=False)
                        btn_clear_log = gr.Button("Clear Log")
                    debug_output = gr.Textbox(label="Debug Log", interactive=False, lines=5)

            # --- Logic Connectors ---
            def _validate_params(mode, d1_v, d2_v, s1_v, s2_v, down_v, up_v, keep1):
                msgs = []

                # s1 / s2
                if s1_v <= 0 and s2_v <= 0:
                    msgs.append("⚠️ s1 и s2 = 0 — остановки по шагу не будут применяться (эффект минимум).")
                elif s1_v > s2_v:
                    msgs.append("⚠️ s1 > s2 — вторая пара фактически не сработает (будет обрезана до s1).")

                # d1 / d2
                if d1_v < 1 or d2_v < 1:
                    msgs.append("⚠️ d1 и d2 должны быть ≥ 1.")
                if d1_v > 10 or d2_v > 10:
                    msgs.append("ℹ️ d1/d2 > 10 — очень глубокий захват блоков, возможен перерасчёт лишних слоёв.")

                # Downscale / Upscale
                if down_v < 0.2:
                    msgs.append("⚠️ Downscale < 0.2 — слишком агрессивное уменьшение, возможны артефакты.")
                if down_v > 0.9:
                    msgs.append("⚠️ Downscale > 0.9 — эффект hires.fix будет минимальным.")
                if up_v > 3.0:
                    msgs.append("⚠️ Upscale > 3.0 — очень агрессивное увеличение, возможны мыло и шум.")

                # Масштаб = 1
                if keep1:
                    eff = float(down_v) * float(up_v)
                    if abs(eff - 1.0) > 0.15:
                        msgs.append(f"ℹ️ \"Сохранять масштаб=1\" включён, но down*up ≈ {eff:.2f} (отклонение от 1.0).")

                mode_s = (mode or "").strip()
                if "Legacy" in mode_s and up_v > 2.5:
                    msgs.append("ℹ️ В Legacy-режиме Upscale > 2.5 иногда даёт более жёсткий шум.")

                if not msgs:
                    return "✅ Параметры корректны"
                return "\n".join(msgs)

            for p in [d1, d2, s1, s2, downscale, upscale, keep_unitary_product]:
                p.change(
                    _validate_params,
                    inputs=[algo_mode, d1, d2, s1, s2, downscale, upscale, keep_unitary_product],
                    outputs=[param_warnings]
                )

            def _toggle_simple_mode(is_simple, mode):
                is_enh = (mode == "Enhanced (RU+)")
                return gr.update(visible=not is_simple), gr.update(visible=is_enh), gr.update(visible=is_enh)
            simple_mode.change(_toggle_simple_mode, inputs=[simple_mode, algo_mode], outputs=[advanced_group, align_corners_mode, recompute_scale_factor_mode])

            def _toggle_algo_vis(mode):
                is_enh = (mode == "Enhanced (RU+)")
                return (gr.update(visible=is_enh), gr.update(visible=not is_enh), gr.update(visible=is_enh),
                        gr.update(visible=is_enh), gr.update(visible=is_enh), gr.update(visible=is_enh))
            algo_mode.change(_toggle_algo_vis, inputs=[algo_mode], outputs=[smooth_scaling_enh, smooth_scaling_legacy, smoothing_curve, keep_unitary_product, only_one_pass_enh, only_one_pass_legacy])

            def _toggle_adaptive(adapt_enabled):
                # Профиль имеет смысл только когда адаптация включена
                return gr.update(visible=adapt_enabled)

            adaptive_by_resolution.change(
                _toggle_adaptive,
                inputs=[adaptive_by_resolution],
                outputs=[adaptive_profile]
            )

            def _update_status(enabled, mode):
                return f"🟢 **Активен: {mode}**" if enabled else "🔴 **Отключено**"
            enable.change(_update_status, [enable, algo_mode], [status_indicator])
            algo_mode.change(_update_status, [enable, algo_mode], [status_indicator])
            
            # Helper for presets list update
            def _update_preset_list_for_category(cat: str):
                pm.reload()
                return gr.update(choices=pm.names_for_category(cat), value=None)
            preset_category_filter.change(_update_preset_list_for_category, inputs=[preset_category_filter], outputs=[preset_select])

            stop_preview_toggle.change(lambda e,t,s1,s2: (gr.update(visible=e), _format_stop_preview_text(t,s1,s2) if e else ""), inputs=[stop_preview_toggle, stop_preview_steps, s1, s2], outputs=[stop_preview_steps, stop_preview_md])
            
            # --- Quick Presets Logic ---
            def _apply_quick_preset(preset_type: str):
                presets = {
                    "safe": {"s1": 0.15, "s2": 0.25, "d1": 3, "d2": 4, "down": 0.6, "up": 1.8},
                    "balanced": {"s1": 0.18, "s2": 0.32, "d1": 3, "d2": 5, "down": 0.5, "up": 2.0},
                    "aggressive": {"s1": 0.22, "s2": 0.38, "d1": 4, "d2": 6, "down": 0.4, "up": 2.5},
                }
                p = presets.get(preset_type, presets["balanced"])
                return (p["s1"], p["s2"], p["d1"], p["d2"], p["down"], p["up"])
            
            btn_quick_safe.click(lambda: _apply_quick_preset("safe"), outputs=[s1, s2, d1, d2, downscale, upscale])
            btn_quick_balanced.click(lambda: _apply_quick_preset("balanced"), outputs=[s1, s2, d1, d2, downscale, upscale])
            btn_quick_aggressive.click(lambda: _apply_quick_preset("aggressive"), outputs=[s1, s2, d1, d2, downscale, upscale])

            # --- Save/Load Presets (FULL LOGIC RESTORED) ---
            def _save_preset_cb(
                name, cat_in, cat_filt,
                mode, d1_v, d2_v, depth_guard_v, depth_clamp_v, depth_pair_mode_v,
                s1_v, s2_v, scl, dw, up,
                sm_enh, sm_leg, sm_sel, sm_c, eo,
                one_enh, one_leg, one_sel,
                k1, al, rc,
                res, app, ad, ad_p,
                legacy_align_false, enh_safe_clamp,
                old_math, old_one
            ):
                name = (name or "").strip()
                if not name:
                    return gr.update(), gr.update(), "⚠️ Имя?"

                cat = (cat_in or "").strip() or (cat_filt if cat_filt != "Все" else "Общие")

                base = HiresPreset().to_dict()
                base.update({
                    "category": cat,
                    "algo_mode": mode,
                    "d1": int(d1_v),
                    "d2": int(d2_v),
                    "depth_guard": bool(depth_guard_v),
                    "depth_clamp": bool(depth_clamp_v),
                    "depth_pair_mode": str(depth_pair_mode_v),
                    "s1": float(s1_v),
                    "s2": float(s2_v),
                    "scaler": str(scl),
                    "downscale": float(dw),
                    "upscale": float(up),
                    "smooth_scaling_enh": bool(sm_enh),
                    "smooth_scaling_legacy": bool(sm_leg),
                    "smoothing_mode": str(sm_sel),
                    "smoothing_curve": str(sm_c),
                    "early_out": bool(eo),
                    "only_one_pass_enh": bool(one_enh),
                    "only_one_pass_legacy": bool(one_leg),
                    "one_pass_mode": str(one_sel),
                    "keep_unitary_product": bool(k1),
                    "align_corners_mode": str(al),
                    "recompute_scale_factor_mode": str(rc),
                    "resolution_choice": str(res),
                    "apply_resolution": bool(app),
                    "adaptive_by_resolution": bool(ad),
                    "adaptive_profile": str(ad_p),
                    "legacy_force_align_false": bool(legacy_align_false),
                    "enhanced_safe_upscale_clamp": bool(enh_safe_clamp),
                    "use_old_float_math": bool(old_math),
                    "use_old_onepass_logic": bool(old_one),
                })
                pm.upsert(name, HiresPreset(**base))
                cats = ["Все"] + pm.categories()
                return (
                    gr.update(choices=cats, value=cat),
                    gr.update(choices=pm.names_for_category(cat), value=name),
                    f"✅ Сохранён «{name}»."
                )
            
            btn_save.click(_save_preset_cb, inputs=[preset_name, preset_category_input, preset_category_filter, algo_mode, d1, d2, depth_guard, depth_clamp, depth_pair_mode, s1, s2, scaler, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select, keep_unitary_product, align_corners_mode, recompute_scale_factor_mode, resolution_choice, apply_resolution, adaptive_by_resolution, adaptive_profile, legacy_force_align_false, enhanced_safe_clamp, use_old_float_math, use_old_onepass_logic], outputs=[preset_category_filter, preset_select, preset_status])
            
            def _load_preset_cb(name):
                p = pm.get(name)
                if not p:
                    # 31 gr.update() для всех контролов + строка в preset_status
                    return (*[gr.update()]*31, "❌ Пресет не найден")

                pd = p.to_dict()
                return (
                    gr.update(value=pd.get("algo_mode", "Enhanced (RU+)")),      # algo_mode
                    gr.update(value=pd.get("d1", 3)),                            # d1
                    gr.update(value=pd.get("d2", 4)),                            # d2
                    gr.update(value=pd.get("depth_guard", True)),                # depth_guard
                    gr.update(value=pd.get("depth_clamp", True)),                # depth_clamp
                    gr.update(value=pd.get("depth_pair_mode", "Авто (по алгоритму)")),
                    gr.update(value=pd.get("s1", 0.15)),                         # s1
                    gr.update(value=pd.get("s2", 0.30)),                         # s2
                    gr.update(value=pd.get("scaler", "bicubic")),                # scaler
                    gr.update(value=pd.get("downscale", 0.5)),                   # downscale
                    gr.update(value=pd.get("upscale", 2.0)),                     # upscale
                    gr.update(value=pd.get("smooth_scaling_enh", True)),
                    gr.update(value=pd.get("smooth_scaling_legacy", True)),
                    gr.update(value=pd.get("smoothing_mode", "Авто (по алгоритму)")),
                    gr.update(value=pd.get("smoothing_curve", "Линейная")),
                    gr.update(value=pd.get("early_out", False)),
                    gr.update(value=pd.get("only_one_pass_enh", True)),
                    gr.update(value=pd.get("only_one_pass_legacy", True)),
                    gr.update(value=pd.get("one_pass_mode", "Авто (по алгоритму)")),
                    gr.update(value=pd.get("keep_unitary_product", False)),
                    gr.update(value=pd.get("align_corners_mode", "False")),
                    gr.update(value=pd.get("recompute_scale_factor_mode", "False")),
                    gr.update(value=pd.get("resolution_choice", RESOLUTION_CHOICES[0])),
                    gr.update(value=pd.get("apply_resolution", False)),
                    gr.update(value=pd.get("adaptive_by_resolution", True)),
                    gr.update(value=pd.get("adaptive_profile", "Сбалансированный")),
                    gr.update(value=pd.get("legacy_force_align_false", True)),
                    gr.update(value=pd.get("enhanced_safe_upscale_clamp", True)),
                    gr.update(value=pd.get("use_old_float_math", True)),
                    gr.update(value=pd.get("use_old_onepass_logic", True)),
                    gr.update(value=name),                       # preset_name
                    gr.update(value="✅ Loaded"),                # preset_status
                )
            
            btn_load.click(_load_preset_cb, inputs=[preset_select], outputs=[algo_mode, d1, d2, depth_guard, depth_clamp, depth_pair_mode, s1, s2, scaler, downscale, upscale, smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select, keep_unitary_product, align_corners_mode, recompute_scale_factor_mode, resolution_choice, apply_resolution, adaptive_by_resolution, adaptive_profile, legacy_force_align_false, enhanced_safe_clamp, use_old_float_math, use_old_onepass_logic, preset_name, preset_status])
            
            def _delete_preset_cb(name, cat_filt):
                pm.delete(name)
                cats = ["Все"] + pm.categories()
                return gr.update(choices=cats, value=cat_filt), gr.update(choices=pm.names_for_category(cat_filt), value=None), f"🗑️ Удалён «{name}»."
            btn_delete.click(_delete_preset_cb, inputs=[preset_select, preset_category_filter], outputs=[preset_category_filter, preset_select, preset_status])

            def _clear_log():
                self.debug_log.clear()
                return ""

            btn_clear_log.click(_clear_log, outputs=[debug_output])

            # --- Export/Import Logic (FULL LOGIC RESTORED) ---
            def _export_all_config(*params):
                config = {
                    "version": "2.5.3",
                    "enable": params[0],
                    "simple_mode": params[1],
                    "algo_mode": params[2],
                    "only_one_pass_enh": params[3],
                    "only_one_pass_legacy": params[4],
                    "one_pass_mode": params[5],
                    "d1": params[6],
                    "d2": params[7],
                    "depth_guard": params[8],
                    "depth_clamp": params[9],
                    "depth_pair_mode": params[10],
                    "s1": params[11],
                    "s2": params[12],
                    "stop_preview_enabled": params[13],
                    "stop_preview_steps": params[14],
                    "scaler": params[15],
                    "downscale": params[16],
                    "upscale": params[17],
                    "smooth_scaling_enh": params[18],
                    "smooth_scaling_legacy": params[19],
                    "smoothing_mode": params[20],
                    "smoothing_curve": params[21],
                    "early_out": params[22],
                    "keep_unitary_product": params[23],
                    "align_corners_mode": params[24],
                    "recompute_scale_factor_mode": params[25],
                    "resolution_choice": params[26],
                    "apply_resolution": params[27],
                    "adaptive_by_resolution": params[28],
                    "adaptive_profile": params[29],
                    "legacy_force_align_false": params[30],
                    "enhanced_safe_upscale_clamp": params[31],
                    "use_old_float_math": params[32],
                    "use_old_onepass_logic": params[33],
                }
                return json.dumps(config, indent=2, ensure_ascii=False)

            def _import_all_config(json_str):
                try:
                    config = json.loads(json_str)

                    return (
                        gr.update(value=config.get("enable", False)),                      # 0 enable
                        gr.update(value=config.get("simple_mode", True)),                 # 1 simple_mode
                        gr.update(value=config.get("algo_mode", "Enhanced (RU+)")),       # 2 algo_mode
                        gr.update(value=config.get("only_one_pass_enh", True)),           # 3 only_one_pass_enh
                        gr.update(value=config.get("only_one_pass_legacy", True)),        # 4 only_one_pass_legacy
                        gr.update(value=config.get("one_pass_mode", "Авто (по алгоритму)")),  # 5 one_pass_mode_select
                        gr.update(value=config.get("d1", 3)),                             # 6 d1
                        gr.update(value=config.get("d2", 4)),                             # 7 d2
                        gr.update(value=config.get("depth_guard", True)),                 # 8 depth_guard
                        gr.update(value=config.get("depth_clamp", True)),                 # 9 depth_clamp
                        gr.update(value=config.get("depth_pair_mode", "Авто (по алгоритму)")),  # 10 depth_pair_mode
                        gr.update(value=config.get("s1", 0.15)),                          # 11 s1
                        gr.update(value=config.get("s2", 0.30)),                          # 12 s2
                        gr.update(value=config.get("stop_preview_enabled", False)),       # 13 stop_preview_enabled
                        gr.update(value=int(config.get("stop_preview_steps", 30))),       # 14 stop_preview_steps
                        gr.update(value=config.get("scaler", "bicubic")),                 # 15 scaler
                        gr.update(value=config.get("downscale", 0.5)),                    # 16 downscale
                        gr.update(value=config.get("upscale", 2.0)),                      # 17 upscale
                        gr.update(value=config.get("smooth_scaling_enh", True)),          # 18 smooth_scaling_enh
                        gr.update(value=config.get("smooth_scaling_legacy", True)),       # 19 smooth_scaling_legacy
                        gr.update(value=config.get("smoothing_mode", "Авто (по алгоритму)")),  # 20 smoothing_mode_select
                        gr.update(value=config.get("smoothing_curve", "Линейная")),       # 21 smoothing_curve
                        gr.update(value=config.get("early_out", False)),                  # 22 early_out
                        gr.update(value=config.get("keep_unitary_product", False)),       # 23 keep_unitary_product
                        gr.update(value=config.get("align_corners_mode", "False")),       # 24 align_corners_mode
                        gr.update(value=config.get("recompute_scale_factor_mode", "False")), # 25 recompute_scale_factor_mode
                        gr.update(value=config.get("resolution_choice", RESOLUTION_CHOICES[0])), # 26 resolution_choice
                        gr.update(value=config.get("apply_resolution", False)),           # 27 apply_resolution
                        gr.update(value=config.get("adaptive_by_resolution", True)),      # 28 adaptive_by_resolution
                        gr.update(value=config.get("adaptive_profile", "Сбалансированный")), # 29 adaptive_profile
                        gr.update(value=config.get("legacy_force_align_false", True)),    # 30 legacy_force_align_false
                        gr.update(value=config.get("enhanced_safe_upscale_clamp", True)), # 31 enhanced_safe_clamp
                        gr.update(value=config.get("use_old_float_math", True)),          # 32 use_old_float_math
                        gr.update(value=config.get("use_old_onepass_logic", True)),       # 33 use_old_onepass_logic
                        "✅ Настройки импортированы",                                     # 34 import_status
                    )
                except Exception as e:
                    return (
                        *[gr.update()]*34,
                        f"❌ Ошибка: {e}",
                    )

            all_params_list = [
                enable, simple_mode, algo_mode, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select,
                d1, d2, depth_guard, depth_clamp, depth_pair_mode, s1, s2, stop_preview_toggle, stop_preview_steps, scaler, downscale, upscale,
                smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out,
                keep_unitary_product, align_corners_mode, recompute_scale_factor_mode, resolution_choice,
                apply_resolution, adaptive_by_resolution, adaptive_profile, legacy_force_align_false, enhanced_safe_clamp,
                use_old_float_math, use_old_onepass_logic
            ]

            btn_export_config.click(_export_all_config, inputs=all_params_list, outputs=[config_json])
            btn_import_config.click(_import_all_config, inputs=[config_json], outputs=all_params_list + [import_status])

        self.infotext_fields.append((enable, "DSHF_enabled"))
        for k, el in {
            "DSHF_mode": algo_mode, "DSHF_s1": s1, "DSHF_d1": d1, "DSHF_s2": s2, "DSHF_d2": d2,
            "DSHF_scaler": scaler, "DSHF_down": downscale, "DSHF_up": upscale, "DSHF_old_float": use_old_float_math
        }.items():
            self.infotext_fields.append((el, k))

        return all_params_list + [debug_mode]

    def process(
        self, p, enable, simple, algo_mode, only_one_pass_enh, only_one_pass_legacy, one_pass_mode_select,
        d1, d2, depth_guard, depth_clamp, depth_pair_mode, s1, s2, stop_preview_enabled, stop_preview_steps, scaler, downscale, upscale,
        smooth_scaling_enh, smooth_scaling_legacy, smoothing_mode_select, smoothing_curve, early_out,
        keep_unitary_product, align_ui, recompute_ui, res_choice, apply_res,
        adapt, adapt_prof, legacy_force_align_false, enhanced_safe_clamp,
        use_old_float_math, use_old_onepass_logic, # 🆕 HYBRID FLAGS
        debug_mode_val
    ):
        self.step_limit = 0
        self.debug_mode = debug_mode_val
        self.disable = False  # ⬅︎ добавить
        
        # ✅ ИСПРАВЛЕНИЕ 1: ПОЛНЫЙ CONFIG (Как в версии 19)
        self.config = DictConfig({
            "algo_mode": algo_mode,
            "simple_mode": simple,
            "s1": s1, "s2": s2,
            "d1": d1, "d2": d2,
            "depth_guard": depth_guard,
            "depth_clamp": depth_clamp,               # 🆕
            "depth_pair_mode": depth_pair_mode, 
            "stop_preview_enabled": stop_preview_enabled,
            "stop_preview_steps": stop_preview_steps,
            "scaler": scaler,
            "downscale": downscale,
            "upscale": upscale,
            "smooth_scaling_enh": smooth_scaling_enh,
            "smooth_scaling_legacy": smooth_scaling_legacy,
            "smoothing_mode": smoothing_mode_select,
            "smoothing_curve": smoothing_curve,
            "early_out": early_out,
            "only_one_pass_enh": only_one_pass_enh,
            "only_one_pass_legacy": only_one_pass_legacy,
            "one_pass_mode": one_pass_mode_select,
            "keep_unitary_product": keep_unitary_product,
            "align_corners_mode": align_ui,
            "recompute_scale_factor_mode": recompute_ui,
            "resolution_choice": res_choice,
            "apply_resolution": apply_res,
            "adaptive_by_resolution": adapt,
            "adaptive_profile": adapt_prof,
            "use_old_float_math": use_old_float_math,       # 🆕
            "use_old_onepass_logic": use_old_onepass_logic, # 🆕
            "legacy_force_align_false": legacy_force_align_false,
            "enhanced_safe_upscale_clamp": enhanced_safe_clamp,
            "debug_mode": debug_mode_val
        })

        self._save_config()
        
        if not enable or self.disable:
            try: script_callbacks.remove_current_script_callbacks()
            except: pass
            self._cb_registered = False
            return

        if apply_res:
             try:
                 wh = parse_resolution_label(res_choice)
                 if wh: p.width, p.height = wh
             except: pass

        use_s1, use_s2 = _clamp(float(s1), 0.0, 1.0), _clamp(float(s2), 0.0, 1.0)
        use_d1, use_d2 = int(d1), int(d2)
        use_down, use_up = float(downscale), float(upscale)
        
        # ⚙️ Адаптация по разрешению: ТОЛЬКО для Enhanced, в Legacy ведём себя как khrfix_olds
        if adapt and algo_mode != "Legacy (Original)":
            try:
                use_s1, use_s2, use_d1, use_d2, use_down, use_up = _compute_adaptive_params(
                    int(getattr(p, "width", 1024)), int(getattr(p, "height", 1024)), adapt_prof,
                    use_s1, use_s2, d1, d2, downscale, upscale, keep_unitary_product
                )
            except:
                pass

        if use_s1 > use_s2: use_s2 = use_s1

        model_container = getattr(p.sd_model, "model", None)
        if not model_container: return
        model = model_container.diffusion_model
        
        inp_list = getattr(model, "input_blocks", [])
        out_list = getattr(model, "output_blocks", [])
        max_inp = len(inp_list) - 1

        d1_idx = int(use_d1) - 1
        d2_idx = int(use_d2) - 1
        scaler_mode = _safe_mode(scaler)

        if algo_mode == "Legacy (Original)" and legacy_force_align_false:
            # Строгий Legacy: как в старом скрипте — всегда False/False
            align_mode, recompute_mode = "false", "false"
        else:
            # Во всех остальных случаях — читаем из UI
            align_mode = _norm_mode_choice(align_ui, "auto")
            recompute_mode = _norm_mode_choice(recompute_ui, "auto")

        # Select logic
        def _select_cross_mode(choice: str, enh_value: bool, legacy_value: bool) -> bool:
            sel = (choice or "").strip().lower()
            if sel.startswith("использовать legacy"): return bool(legacy_value)
            if sel.startswith("использовать enhanced"): return bool(enh_value)
            return bool(enh_value) if algo_mode == "Enhanced (RU+)" else bool(legacy_value)

        use_smooth = _select_cross_mode(smoothing_mode_select, smooth_scaling_enh, smooth_scaling_legacy)
        use_one = _select_cross_mode(one_pass_mode_select, only_one_pass_enh, only_one_pass_legacy)

        # 🛡️ DEPTH GUARD & MAPPING LOGIC
        mapping_notes = []

        def _normalize_depth(idx, label):
            # Если clamp выключен — индексы вне диапазона просто выкидываем
            if not depth_clamp:
                if idx < 0 or idx > max_inp:
                    mapping_notes.append(f"{label} out of range → skip (idx={idx}, max={max_inp})")
                    return None
                return int(idx)

            # Старое поведение: жёсткий clamp в [0..max_inp]
            clamped = max(0, min(int(idx), max_inp))
            if clamped != int(idx):
                mapping_notes.append(f"{label} clamped {idx} → {clamped}")
            return clamped

        d1_idx = _normalize_depth(d1_idx, "d1")
        d2_idx = _normalize_depth(d2_idx, "d2")

        # Если оба ушли в None — делать вообще нечего
        if d1_idx is None and d2_idx is None:
            if self.debug_mode:
                self._append_debug("Both d1 and d2 are out of range, skip callback.")
            return

        # Если один из depth отвалился — работаем только со вторым
        if depth_guard and d1_idx is not None and d2_idx is not None and d2_idx < d1_idx:
            d1_idx, d2_idx = d2_idx, d1_idx

        def _resolve_pair_mode(mode_str: str) -> str:
            ms = (mode_str or "").strip().lower()
            if ms.startswith("авто"):
                # Авто: Legacy → как olds.py, Enhanced → обе глубины
                return "sequential" if algo_mode == "Legacy (Original)" else "both"
            if "послед" in ms:
                return "sequential"
            if "оба" in ms or "обе" in ms:
                return "both"
            if "только d1" in ms:
                return "d1"
            if "только d2" in ms:
                return "d2"
            return "both"

        resolved_pair_mode = _resolve_pair_mode(depth_pair_mode)
        
        def _map_with_note(idx, tag: str):
            if idx is None:
                mapping_notes.append(f"{tag}: skipped (idx=None)")
                return None
            out_idx = KohyaHiresFix._map_output_index(model, idx, early_out)
            if out_idx is None:
                mapping_notes.append(f"{tag}: {idx} -> No Out")
            else:
                mapping_notes.append(f"{tag}: {idx} -> {out_idx}")
            return out_idx

        _map_with_note(d1_idx, "d1")
        _map_with_note(d2_idx, "d2")

        if self.debug_mode:
            self._append_debug(f"Mapping notes: {', '.join(mapping_notes)}")

        def _apply_depth_pair(d_i: int, s_stop: float, current: int, total: int):
            if d_i is None or s_stop <= 0:
                return False  # ничего не делали

            if d_i >= len(model.input_blocks):
                return False
            out_i = KohyaHiresFix._map_output_index(model, d_i, early_out)
            if out_i is None or out_i >= len(model.output_blocks):
                return False

            if model.input_blocks[d_i] is None or model.output_blocks[out_i] is None:
                return False

            stop_step = total * s_stop

            if current < stop_step:
                # Вешаем/обновляем Scaler
                if not isinstance(model.input_blocks[d_i], Scaler):
                    model.input_blocks[d_i] = Scaler(
                        use_down, model.input_blocks[d_i], scaler_mode,
                        align_mode, recompute_mode,
                        use_old_float_math=use_old_float_math
                    )
                    model.output_blocks[out_i] = Scaler(
                        use_up, model.output_blocks[out_i], scaler_mode,
                        align_mode, recompute_mode,
                        use_old_float_math=use_old_float_math
                    )

                if use_smooth:
                    ratio = float(max(0.0, min(1.0, current / stop_step)))
                    if algo_mode == "Enhanced (RU+)" and smoothing_curve == "Smoothstep":
                        ratio = ratio * ratio * (3.0 - 2.0 * ratio)

                    cur_down = min((1.0 - use_down) * ratio + use_down, 1.0)
                    model.input_blocks[d_i].scale = cur_down

                    if algo_mode == "Enhanced (RU+)" and keep_unitary_product:
                        cur_up = 1.0 / max(1e-6, cur_down)
                    else:
                        cur_up = use_up * (use_down / max(1e-6, cur_down))
                        if algo_mode == "Enhanced (RU+)" and enhanced_safe_clamp:
                            cur_up = _clamp(cur_up, 1.0, 4.0)

                    model.output_blocks[out_i].scale = cur_up

                return True  # что-то делали

            else:
                # Вышли за stop_step — снимаем Scaler (если он есть)
                if isinstance(model.input_blocks[d_i], Scaler):
                    model.input_blocks[d_i] = model.input_blocks[d_i].block
                    if isinstance(model.output_blocks[out_i], Scaler):
                        model.output_blocks[out_i] = model.output_blocks[out_i].block
                return False

        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if self.disable:
                return
            try:
                total = max(1, int(params.total_sampling_steps))
                current = params.sampling_step
                
                # ✅ Логирование шагов – здесь, внутри колбэка
    
                if self.debug_mode:
                    self._append_debug(f"Step {current}/{total}")

                # ✅ ИСПРАВЛЕНИЕ 2: ПРОВЕРКА В НАЧАЛЕ (Start Check)
                if use_one:
                    if use_old_onepass_logic:
                        # OLD LOGIC: строгое сравнение с записанным шагом
                        if current < self.step_limit:
                            return
                    else:
                        # NEW LOGIC: проверка флага
                        if self.step_limit == 1:
                            return

                # Формируем пары с учётом выбранного режима
                raw_pairs = []
                if resolved_pair_mode in ("both", "sequential", "d1", "d2"):
                    if resolved_pair_mode in ("both", "sequential", "d1") and d1_idx is not None:
                        raw_pairs.append(("p1", use_s1, d1_idx))
                    if resolved_pair_mode in ("both", "sequential", "d2") and d2_idx is not None:
                        raw_pairs.append(("p2", use_s2, d2_idx))
                else:
                    # fallback — как "both"
                    if d1_idx is not None:
                        raw_pairs.append(("p1", use_s1, d1_idx))
                    if d2_idx is not None:
                        raw_pairs.append(("p2", use_s2, d2_idx))

                # Для расчёта max_stop_s (нужно для new onepass logic)
                max_stop_s = max((s for _, s, _ in raw_pairs if s > 0), default=0.0)

                if resolved_pair_mode == "sequential":
                    # 🧠 Режим "как olds.py": обрабатываем пары по очереди, после первой успешной — return
                    for tag, s_stop, d_i in raw_pairs:
                        if s_stop <= 0:
                            continue

                        did_work = _apply_depth_pair(d_i, s_stop, current, total)

                        # olds.py по факту тоже «останавливается» на первой сработавшей паре
                        if did_work:
                            # После первой активной пары — выходим из callback
                            return

                    # Если до сюда дошли — ни одна пара не активна, но могли снять Scaler.
                    # Ниже сработает onepass-логика.

                else:
                    # Режим "оба depth одновременно" (и прочие, кроме sequential)
                    combined: Dict[int, float] = {}
                    for _, s_stop, d_i in raw_pairs:
                        if s_stop > 0 and d_i is not None:
                            combined[d_i] = max(combined.get(d_i, 0.0), s_stop)

                    for d_i, s_stop in combined.items():
                        _apply_depth_pair(d_i, s_stop, current, total)

                # ✅ ИСПРАВЛЕНИЕ 3: УСТАНОВКА ЛИМИТА В КОНЦЕ (End Update)
                if use_one:
                    if use_old_onepass_logic:
                        # OLD WAY: обновляем ВСЕГДА, чтобы step_limit рос вместе с current
                        # Это в точности копирует логику khrfix_olds
                        self.step_limit = current
                    else:
                        # NEW WAY: ставим флаг "1" только когда эффект закончился
                        if max_stop_s > 0 and current >= total * max_stop_s:
                            self.step_limit = 1

            except Exception as e:
                try:
                    KohyaHiresFix._unwrap_all(model)
                except Exception:
                    pass

                print(f"[KohyaHiresFix] Error in callback: {e}")
                self.disable = True

                # сразу отключаем все колбэки этого скрипта, чтобы не спамить ошибками
                try:
                    script_callbacks.remove_current_script_callbacks()
                except Exception:
                    pass

                self._cb_registered = False
        
        # 🔗 РЕГИСТРАЦИЯ CALLBACK'А (главное, чего не хватало)
        if not self._cb_registered:
            script_callbacks.on_cfg_denoiser(denoiser_callback)
            self._cb_registered = True

    def postprocess(self, p, processed, *args):
        try:
            model_container = getattr(p.sd_model, "model", None)
            if model_container:
                KohyaHiresFix._unwrap_all(model_container.diffusion_model)
        finally:
            try:
                script_callbacks.remove_current_script_callbacks()
            except Exception:
                pass
            self._cb_registered = False

    def process_batch(self, p, *args, **kwargs):
        self.step_limit = 0

# КОНЕЦ ЧАСТИ 4 (ФИНАЛ)
