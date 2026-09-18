"""Mamdani Fuzzy Inference Engine (FUS-03).

Triển khai động cơ suy luận mờ Mamdani hoàn chỉnh theo đặc tả §8 của DriverSafe-IoT:
- 6 biến đầu vào (0-100): perclos, cles, yawn, head_drop, alco, cabin
- 1 biến đầu ra: risk (0-100) với 4 tập thuộc (safe, warn, alarm, critical)
- 12 luật mờ (ngân hàng luật cấu hình qua host/fusion/rules.yaml)
- Giải mờ Mamdani min-implication và defuzzification bằng trọng tâm (Centroid)

Chạy kiểm tra độc lập:
    python host/fusion/fuzzy.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"


def trimf(x: np.ndarray | float, params: list[float] | tuple[float, ...]) -> np.ndarray | float:
    """Hàm thuộc tam giác: trimf(x, [a, b, c])."""
    a, b, c = params
    is_scalar = np.isscalar(x) or isinstance(x, (int, float))
    x_arr = np.asarray(x, dtype=float)

    term1 = (x_arr - a) / (b - a) if b > a else np.where(x_arr >= a, 1.0, 0.0)
    term2 = (c - x_arr) / (c - b) if c > b else np.where(x_arr <= c, 1.0, 0.0)
    y = np.maximum(0.0, np.minimum(term1, term2))
    return float(y.item()) if is_scalar else y


def trapmf(x: np.ndarray | float, params: list[float] | tuple[float, ...]) -> np.ndarray | float:
    """Hàm thuộc hình thang: trapmf(x, [a, b, c, d])."""
    a, b, c, d = params
    is_scalar = np.isscalar(x) or isinstance(x, (int, float))
    x_arr = np.asarray(x, dtype=float)

    term1 = (x_arr - a) / (b - a) if b > a else np.where(x_arr >= a, 1.0, 0.0)
    term2 = (d - x_arr) / (d - c) if d > c else np.where(x_arr <= d, 1.0, 0.0)
    y = np.maximum(0.0, np.minimum(np.minimum(term1, 1.0), term2))
    return float(y.item()) if is_scalar else y


def compute_membership(x: np.ndarray | float, mf_def: dict[str, Any]) -> np.ndarray | float:
    """Tính độ thuộc của x dựa trên định nghĩa hàm thuộc mf_def."""
    mf_type = mf_def.get("type", "trimf")
    params = mf_def.get("params", [])
    if mf_type == "trimf":
        return trimf(x, params)
    elif mf_type == "trapmf":
        return trapmf(x, params)
    raise ValueError(f"Khong ho tro loai ham thuoc '{mf_type}'")


@dataclass
class FiredRule:
    """Thông tin một luật mờ đã được kích hoạt."""

    id: str
    name: str
    weight: float
    consequent: str


class MamdaniEngine:
    """Động cơ suy luận mờ Mamdani hỗ trợ 6 inputs, giải mờ trọng tâm (Centroid)."""

    def __init__(self, rules_path: str | Path | None = None, config_dict: dict | None = None) -> None:
        if config_dict is not None:
            self.cfg = config_dict
        else:
            path = Path(rules_path or DEFAULT_RULES_PATH)
            if not path.is_file():
                raise FileNotFoundError(f"Khong tim thay file cau hinh rules tai: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        self.inputs_cfg: dict[str, Any] = self.cfg.get("inputs", {})
        self.output_cfg: dict[str, Any] = self.cfg.get("output", {}).get("risk", {})
        self.rules: list[dict[str, Any]] = self.cfg.get("rules", [])

        # Khởi tạo lưới tọa độ đầu ra y phục vụ giải mờ Centroid
        out_range = self.output_cfg.get("range", [0, 100])
        step = float(self.output_cfg.get("step", 0.5))
        self.y_grid = np.arange(out_range[0], out_range[1] + step, step)

        # Tiền tính toán các hàm thuộc của biến đầu ra trên lưới y
        self.output_mfs: dict[str, np.ndarray] = {}
        for mf_name, mf_def in self.output_cfg.get("mfs", {}).items():
            self.output_mfs[mf_name] = compute_membership(self.y_grid, mf_def)  # type: ignore

    def eval_condition(self, var_name: str, mf_target: str | list[str], val: float) -> float:
        """Tính giá trị độ thuộc cho một điều kiện đơn lẻ."""
        var_spec = self.inputs_cfg.get(var_name)
        if not var_spec:
            return 0.0

        val_clamped = max(0.0, min(100.0, float(val)))
        mfs_dict = var_spec.get("mfs", {})

        if isinstance(mf_target, list):
            # Kết hợp OR giữa các mức thuộc tính của cùng 1 biến (vd: [med, high])
            scores = [
                float(compute_membership(val_clamped, mfs_dict[mf]))
                for mf in mf_target
                if mf in mfs_dict
            ]
            return max(scores) if scores else 0.0

        if mf_target in mfs_dict:
            return float(compute_membership(val_clamped, mfs_dict[mf_target]))
        return 0.0

    def eval_antecedent(self, ant: dict[str, Any], inputs: dict[str, float]) -> float:
        """Đánh giá phần tiền đề (antecedent) của luật mờ (hỗ trợ AND / OR)."""
        op = ant.get("op", "and").lower()
        conditions = ant.get("conditions", [])
        if not conditions:
            return 0.0

        cond_values: list[float] = []
        for cond in conditions:
            var_name = cond.get("var", "")
            mf_target = cond.get("mf", "")
            val = inputs.get(var_name, 0.0)
            cond_values.append(self.eval_condition(var_name, mf_target, val))

        if op == "or":
            return max(cond_values) if cond_values else 0.0
        # Mặc định toán tử AND (min)
        return min(cond_values) if cond_values else 0.0

    def infer(self, inputs: dict[str, float]) -> tuple[float, list[FiredRule]]:
        """Thực hiện chu trình suy luận Mamdani: Fuzzify -> Infer -> Aggregation -> Centroid.

        Args:
            inputs: Dict 6 biến đầu vào đã scale 0-100:
                    perclos, cles, yawn, head_drop, alco, cabin

        Returns:
            (risk_score, fired_rules):
                risk_score: Điểm rủi ro 0-100 được làm tròn 1 chữ số thập phân
                fired_rules: Danh sách các luật đã kích hoạt kèm độ mạnh
        """
        # Mảng chứa đường bao tổng hợp độ thuộc đầu ra mu_agg(y)
        mu_agg = np.zeros_like(self.y_grid)
        fired_rules: list[FiredRule] = []

        for r in self.rules:
            r_id = r.get("id", "")
            r_name = r.get("name", "")
            antecedent = r.get("antecedent", {})
            consequent = r.get("consequent", {})

            # 1. Đánh giá firing strength của antecedent
            strength = self.eval_antecedent(antecedent, inputs)

            if strength > 1e-4:
                cons_mf_name = consequent.get("mf", "")
                fired_rules.append(
                    FiredRule(
                        id=r_id,
                        name=r_name,
                        weight=round(strength, 3),
                        consequent=cons_mf_name,
                    )
                )

                # 2. Mamdani Min-Implication: cắt đỉnh hàm thuộc đầu ra tương ứng
                if cons_mf_name in self.output_mfs:
                    cons_mu = self.output_mfs[cons_mf_name]
                    mu_cut = np.minimum(strength, cons_mu)

                    # 3. Max-Aggregation: gộp vào đường bao tổng hợp
                    mu_agg = np.maximum(mu_agg, mu_cut)

        # 4. Giải mờ trọng tâm (Centroid Defuzzification)
        denom = np.sum(mu_agg)
        if denom <= 1e-6:
            # Fallback an toàn nếu không luật nào kích hoạt
            risk_score = 0.0
        else:
            centroid = float(np.sum(self.y_grid * mu_agg) / denom)
            risk_score = round(max(0.0, min(100.0, centroid)), 1)

        # Sắp xếp fired_rules theo độ mạnh giảm dần
        fired_rules.sort(key=lambda item: item.weight, reverse=True)
        return risk_score, fired_rules


def normalize_inputs(
    perclos_pct: float = 0.0,
    cles_dur_ms: float = 0.0,
    yawn_per_min: float = 0.0,
    head_drop: bool = False,
    alco_level: int = 0,
    ntc_temp_c: float = 28.0,
    ldr_lux: float = 300.0,
) -> dict[str, float]:
    """Ánh xạ các chỉ số thô từ Vision và ESP32 sang miền chuẩn hóa 0-100 cho Fuzzy Engine.

    Quy tắc ánh xạ:
      - perclos: 0% -> 0, 40% -> 100 (tuyến tính)
      - cles: 0ms -> 0, 1500ms -> 100 (1500ms là ngưỡng microsleep nguy hiểm)
      - yawn: 0 yawn/min -> 0, 4.0 yawn/min -> 100
      - head_drop: True -> 100.0, False -> 0.0
      - alco: mức 0 -> 0.0, mức 1 -> 50.0, mức 2 -> 100.0
      - cabin: kết hợp nhiệt độ cabin NTC (20°C..42°C -> 0..100) và độ tối mắt (<50 lux).
    """
    # 1. PERCLOS: 0 -> 40% ánh xạ 0 -> 100 (spec §8)
    perclos_norm = min(100.0, max(0.0, (perclos_pct / 0.40) * 100.0))

    # 2. CLES: 0 -> 1500ms ánh xạ 0 -> 100
    cles_norm = min(100.0, max(0.0, (cles_dur_ms / 1500.0) * 100.0))

    # 3. Yawn: 0 -> 4 l/phút ánh xạ 0 -> 100
    yawn_norm = min(100.0, max(0.0, (yawn_per_min / 4.0) * 100.0))

    # 4. Head drop: boolean -> 0 hoặc 100
    head_drop_norm = 100.0 if head_drop else 0.0

    # 5. Alcohol level: 0/1/2 -> 0/50/100
    alco_norm = 0.0 if alco_level == 0 else (50.0 if alco_level == 1 else 100.0)

    # 6. Cabin index: Nhiệt độ nóng > 35°C tạo cảm giác buồn ngủ mệt mỏi
    # Baseline: 22°C = 0, 42°C = 100
    temp_score = min(100.0, max(0.0, (ntc_temp_c - 22.0) / (42.0 - 22.0) * 100.0))
    # Yếu tố ánh sáng: quá tối (<30 lux) làm mắt dễ sụp
    lux_penalty = 15.0 if ldr_lux < 30.0 else 0.0
    cabin_norm = min(100.0, temp_score + lux_penalty)

    return {
        "perclos": round(perclos_norm, 1),
        "cles": round(cles_norm, 1),
        "yawn": round(yawn_norm, 1),
        "head_drop": round(head_drop_norm, 1),
        "alco": round(alco_norm, 1),
        "cabin": round(cabin_norm, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mamdani Fuzzy Engine CLI Demo")
    parser.add_argument("--rules", default=None, help="Duong dan rules.yaml")
    args = parser.parse_args()

    engine = MamdaniEngine(args.rules)
    print("=== Mamdani Fuzzy Inference Engine (FUS-03) Demo ===\n")

    test_cases = [
        ("Tinh tao, tat ca binh thuong", {"perclos": 0.0, "cles": 0.0, "yawn": 0.0, "head_drop": 0.0, "alco": 0.0, "cabin": 10.0}),
        ("Cabin nong b bite, ngap nhe", {"perclos": 20.0, "cles": 10.0, "yawn": 45.0, "head_drop": 0.0, "alco": 0.0, "cabin": 85.0}),
        ("Buon ngu trung binh (PERCLOS + CLES)", {"perclos": 45.0, "cles": 50.0, "yawn": 30.0, "head_drop": 0.0, "alco": 0.0, "cabin": 30.0}),
        ("Microsleep chop mat dai 1.8s", {"perclos": 35.0, "cles": 95.0, "yawn": 10.0, "head_drop": 0.0, "alco": 0.0, "cabin": 20.0}),
        ("Guc dau nguy hiem (Head drop)", {"perclos": 20.0, "cles": 20.0, "yawn": 0.0, "head_drop": 100.0, "alco": 0.0, "cabin": 20.0}),
        ("Nong do con cao (ALCO=2)", {"perclos": 10.0, "cles": 0.0, "yawn": 0.0, "head_drop": 0.0, "alco": 100.0, "cabin": 20.0}),
        ("Con cao ket hop buon ngu nang", {"perclos": 80.0, "cles": 70.0, "yawn": 60.0, "head_drop": 0.0, "alco": 100.0, "cabin": 80.0}),
    ]

    for label, inp in test_cases:
        score, drivers = engine.infer(inp)
        rule_strs = [f"{r.id}(w={r.weight}->{r.consequent})" for r in drivers[:3]]
        print(f"[*] {label:<35} -> RISK: {score:4.1f} | Rules: {', '.join(rule_strs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
