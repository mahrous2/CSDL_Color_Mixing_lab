import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import uuid

@dataclass
class ColorMixingResult:
    red_volume: float
    yellow_volume: float
    blue_volume: float
    well_position: str
    sensor_data: Dict[str, int]
    experiment_id: str
    timestamp: str

class VirtualColorMixingLab:
    """Pure software simulation of the color mixing lab.

    Inputs: R/Y/B volumes (µL), 1–300 each, and R+Y+B <= 300.
    Output: 8-channel sensor readings: ch410, ch440, ch470, ch510, ch550, ch583, ch620, ch670.

    Noise model:
    - Well-position systematic bias (~systematic_noise_level): deterministic per well within a session.
    - Random measurement noise per read (~random_noise_level).
    - Volume-dependent pipetting error.
    """

    def __init__(self,
                 plate_rows: int = 8,
                 plate_cols: int = 12,
                 random_seed: Optional[int] = None,
                 systematic_noise_level: float = 0.10,
                 random_noise_level: float = 0.02):
        self.plate_rows = plate_rows
        self.plate_cols = plate_cols
        self.systematic_noise_level = float(systematic_noise_level)
        self.random_noise_level = float(random_noise_level)

        self._rng = np.random.default_rng(random_seed)

        self.sensor_channels = ['ch410', 'ch440', 'ch470', 'ch510', 'ch550', 'ch583', 'ch620', 'ch670']
        self.results_log: List[ColorMixingResult] = []
        self.experiment_counter = 0

        self._generate_systematic_well_map()

    def _row_col_to_well(self, row: int, col: int) -> str:
        return f"{chr(65 + row)}{col + 1}"

    def _well_to_row_col(self, well: str) -> Tuple[int, int]:
        row = ord(well[0].upper()) - 65
        col = int(well[1:]) - 1
        return row, col

    def validate_volumes(self, r: float, y: float, b: float) -> Optional[str]:
        if not all(1 <= v <= 300 for v in (r, y, b)):
            return "Each volume must be between 1 and 300 µL"
        total = r + y + b
        if total > 300:
            return f"Total volume ({total}µL) exceeds 300µL limit"
        return None

    def _generate_systematic_well_map(self) -> None:
        # Structured field with mild radial trend + correlated noise, normalized to mean 1.
        row_lin = np.linspace(-1, 1, self.plate_rows)
        col_lin = np.linspace(-1, 1, self.plate_cols)
        rr, cc = np.meshgrid(row_lin, col_lin, indexing='ij')
        dist = np.sqrt(rr**2 + cc**2)
        dist = dist / (dist.max() if dist.max() else 1.0)
        base = 1.0 + (self.systematic_noise_level * 0.6) * dist

        field = self._rng.normal(0.0, 1.0, size=(self.plate_rows, self.plate_cols))
        for _ in range(3):
            field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5.0
        field = (field - field.mean()) / (field.std() if field.std() else 1.0)
        structured = 1.0 + (self.systematic_noise_level * 0.4) * field

        wf = base * structured
        self.well_systematic_factor = wf / wf.mean()

    def _pipetting_error(self, target: float) -> float:
        if target < 10:
            cv = 0.15
        elif target < 50:
            cv = 0.05
        else:
            cv = 0.02
        actual = target + self._rng.normal(0.0, cv * target)
        return float(max(0.5, actual))

    def _calculate_sensor_readings(self, r_vol: float, y_vol: float, b_vol: float, well: str) -> Dict[str, int]:
        total = r_vol + y_vol + b_vol
        r = r_vol / total
        y = y_vol / total
        b = b_vol / total

        baseline = {
            'ch410': 300,
            'ch440': 1000,
            'ch470': 1400,
            'ch510': 2300,
            'ch550': 3100,
            'ch583': 3600,
            'ch620': 4800,
            'ch670': 3800
        }

        readings = {}
        for ch, base in baseline.items():
            val = float(base)

            # Red dye effect
            if ch in ['ch410', 'ch440', 'ch470', 'ch510']:
                val *= (1.0 - 0.7 * r)
            elif ch in ['ch550', 'ch583']:
                val *= (1.0 - 0.3 * r)
            else:
                val *= (1.0 + 0.3 * r)

            # Yellow dye effect
            if ch in ['ch410', 'ch440', 'ch470']:
                val *= (1.0 - 0.8 * y)
            elif ch in ['ch510', 'ch550', 'ch583']:
                val *= (1.0 + 0.4 * y)
            else:
                val *= (1.0 + 0.2 * y)

            # Blue dye effect
            if ch in ['ch410', 'ch440', 'ch470', 'ch510']:
                val *= (1.0 + 0.4 * b)
            elif ch in ['ch550', 'ch583']:
                val *= (1.0 - 0.4 * b)
            else:
                val *= (1.0 - 0.7 * b)

            readings[ch] = val

        row, col = self._well_to_row_col(well)
        sys_factor = float(self.well_systematic_factor[row, col])

        out = {}
        for ch, val in readings.items():
            noisy = val * sys_factor
            noisy *= float(self._rng.normal(1.0, self.random_noise_level))
            out[ch] = int(np.clip(noisy, 0, 65535))
        return out

    def mix_colors(self, R: float, Y: float, B: float, well_position: Optional[str] = None) -> ColorMixingResult:
        msg = self.validate_volumes(R, Y, B)
        if msg:
            raise ValueError(msg)

        if well_position is None:
            row = (self.experiment_counter // self.plate_cols) % self.plate_rows
            col = self.experiment_counter % self.plate_cols
            well_position = self._row_col_to_well(row, col)

        r_a = self._pipetting_error(R)
        y_a = self._pipetting_error(Y)
        b_a = self._pipetting_error(B)

        sensor = self._calculate_sensor_readings(r_a, y_a, b_a, well_position)
        exp_id = str(uuid.uuid4())[:8]

        res = ColorMixingResult(
            red_volume=float(R),
            yellow_volume=float(Y),
            blue_volume=float(B),
            well_position=well_position,
            sensor_data=sensor,
            experiment_id=exp_id,
            timestamp=pd.Timestamp.now().isoformat()
        )
        self.results_log.append(res)
        self.experiment_counter += 1
        return res

    def run_experiment_batch(self, experiment_design: List[Dict]) -> pd.DataFrame:
        rows = []
        for i, exp in enumerate(experiment_design):
            r = exp.get('R', exp.get('Red', exp.get('red')))
            y = exp.get('Y', exp.get('Yellow', exp.get('yellow')))
            b = exp.get('B', exp.get('Blue', exp.get('blue')))
            well = exp.get('well', exp.get('Well'))

            res = self.mix_colors(r, y, b, well_position=well)
            rows.append({'Red': res.red_volume, 'Yellow': res.yellow_volume, 'Blue': res.blue_volume, 'well': res.well_position, **res.sensor_data})

        return pd.DataFrame(rows)

    def export_results_df(self) -> pd.DataFrame:
        out = []
        for r in self.results_log:
            out.append({'Red': r.red_volume, 'Yellow': r.yellow_volume, 'Blue': r.blue_volume, 'well': r.well_position, **r.sensor_data,
                        'timestamp': r.timestamp, 'experiment_id': r.experiment_id})
        return pd.DataFrame(out)
