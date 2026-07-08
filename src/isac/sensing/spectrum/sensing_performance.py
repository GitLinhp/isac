"""
感知性能计算模块

提供延迟、距离、多普勒和速度分辨率的计算功能。
"""

from scipy.constants import speed_of_light as c
import numpy as np
from tabulate import tabulate
from typing import Any

from sionna.phy.ofdm import ResourceGrid


class SensingPerformance:
    """感知性能计算类

    用于计算ISAC系统的感知性能参数，包括延迟、距离、多普勒和速度分辨率。
    单基地与双基地相关量分别以 ``_monostatic`` / ``_bistatic`` 后缀区分。
    """

    def __init__(
        self,
        resource_grid: Any,
        carrier_frequency: float = 6e9,  # 默认6 GHz
    ):
        """初始化感知性能计算类

        参数:
        -------
            - resource_grid: Sionna ResourceGrid 或兼容对象
            - carrier_frequency (float): 载波频率 (Hz), 默认6 GHz
        """
        self.rg: ResourceGrid = resource_grid
        self.carrier_frequency: float = carrier_frequency
        self._validate_parameters()

    def _validate_parameters(self):
        """验证输入参数的合法性"""
        assert isinstance(self.rg, ResourceGrid), "资源网格必须是ResourceGrid对象"
        assert isinstance(self.carrier_frequency, (int, float)), "载波频率必须是数值"
        assert self.carrier_frequency > 0, "载波频率必须大于0"

    # ==================== 分辨率计算（模式无关） ====================
    @property
    def delay_resolution(self) -> float:
        """时间分辨率 (秒)

        基于总带宽的倒数（用于完整的资源网格）
        """
        return 1 / self.rg.bandwidth

    @property
    def doppler_resolution(self) -> float:
        """多普勒频率分辨率 (Hz)"""
        return 1 / (self.rg.ofdm_symbol_duration * self.rg.num_ofdm_symbols)

    # ==================== 单基地分辨率 ====================
    @property
    def range_resolution_monostatic(self) -> float:
        """单基地径向距离分辨率 (m)：``Δr = c · Δτ / 2``（往返）。"""
        return c * self.delay_resolution / 2

    @property
    def velocity_resolution_monostatic(self) -> float:
        """单基地速度分辨率 (m/s)：``Δv = Δf_d · c / (2 f_c)``。"""
        return self.doppler_resolution * c / (2 * self.carrier_frequency)

    # ==================== 双基地分辨率 ====================
    @property
    def range_resolution_bistatic(self) -> float:
        """双基地折叠路径长度分辨率 (m/bin)：``ΔL = c · Δτ``（单程，无 /2）。"""
        return c * self.delay_resolution

    @property
    def velocity_resolution_bistatic(self) -> float:
        """双基地速度分辨率 (m/s)：``Δv = Δf_d · c / f_c``。"""
        return self.doppler_resolution * c / self.carrier_frequency

    # ==================== 感知轴（模式无关） ====================
    @property
    def delay_bins(self) -> np.ndarray:
        """时间范围 (纳秒)"""
        return np.arange(0, self.rg.fft_size) * self.delay_resolution / 1e-9

    @property
    def doppler_bins(self) -> np.ndarray:
        """多普勒频率范围 (Hz)"""
        return np.arange(
            -self.rg.num_ofdm_symbols / 2 * self.doppler_resolution,
            self.rg.num_ofdm_symbols / 2 * self.doppler_resolution,
            self.doppler_resolution,
        )

    # ==================== 单基地轴 ====================
    @property
    def range_bins_monostatic(self) -> np.ndarray:
        """单基地径向距离轴 (m)，与 ``delay_to_range(..., sens_mode='monostatic')`` 一致。"""
        k = np.arange(self.rg.fft_size, dtype=np.float64)
        return k * self.range_resolution_monostatic

    @property
    def velocity_bins_monostatic(self) -> np.ndarray:
        """单基地速度轴 (m/s)。"""
        fd_hz = self.doppler_bins
        return -(fd_hz * c) / (2.0 * self.carrier_frequency)

    # ==================== 双基地轴 ====================
    @property
    def range_bins_bistatic(self) -> np.ndarray:
        """双基地折叠路径长度轴 (m)，与 ``delay_to_range(..., sens_mode='bistatic')`` 一致。"""
        k = np.arange(self.rg.fft_size, dtype=np.float64)
        return k * self.range_resolution_bistatic

    @property
    def velocity_bins_bistatic(self) -> np.ndarray:
        """双基地速度轴 (m/s)。"""
        fd_hz = self.doppler_bins
        return -(fd_hz * c) / self.carrier_frequency

    # ==================== 最大探测范围 ====================
    @property
    def max_range_monostatic(self) -> float:
        """单基地最大探测径向距离 (m)。"""
        return (self.rg.fft_size - 1) * self.range_resolution_monostatic

    @property
    def max_velocity_monostatic(self) -> float:
        """单基地最大探测速度 (m/s)。"""
        num_doppler_bins = self.rg.num_ofdm_symbols
        return (num_doppler_bins // 2) * self.velocity_resolution_monostatic

    @property
    def max_range_bistatic(self) -> float:
        """双基地最大折叠路径长度 (m)。"""
        return (self.rg.fft_size - 1) * self.range_resolution_bistatic

    @property
    def max_velocity_bistatic(self) -> float:
        """双基地最大探测速度 (m/s)。"""
        num_doppler_bins = self.rg.num_ofdm_symbols
        return (num_doppler_bins // 2) * self.velocity_resolution_bistatic

    # ==================== 打印性能参数 ====================
    def __call__(self) -> None:
        """打印感知性能参数（单基地 + 双基地）。"""
        table = [
            ["时间分辨率", f"{self.delay_resolution * 1e9:.2f}", "ns"],
            ["多普勒分辨率", f"{self.doppler_resolution:.2f}", "Hz"],
            ["距离分辨率(单基地)", f"{self.range_resolution_monostatic:.2f}", "m"],
            ["距离分辨率(双基地)", f"{self.range_resolution_bistatic:.2f}", "m"],
            ["速度分辨率(单基地)", f"{self.velocity_resolution_monostatic:.2f}", "m/s"],
            ["速度分辨率(双基地)", f"{self.velocity_resolution_bistatic:.2f}", "m/s"],
            ["最大探测距离(单基地)", f"{self.max_range_monostatic:.2f}", "m"],
            ["最大探测路径长度(双基地)", f"{self.max_range_bistatic:.2f}", "m"],
            ["最大探测速度(单基地)", f"{self.max_velocity_monostatic:.2f}", "m/s"],
            ["最大探测速度(双基地)", f"{self.max_velocity_bistatic:.2f}", "m/s"],
        ]
        print("感知性能参数:")
        print(tabulate(table, headers=["参数", "数值", "单位"], tablefmt="simple_grid") + "\n")
