"""
感知性能计算模块

提供延迟、距离、多普勒和速度分辨率的计算功能。
"""

import math

from scipy.constants import speed_of_light as c
import numpy as np
from tabulate import tabulate
from typing import Any

from sionna.phy.ofdm import ResourceGrid


class SensingPerformance:
    """感知性能计算类

    用于计算ISAC系统的感知性能参数，包括延迟、距离、多普勒和速度分辨率。
    """

    def __init__(
        self,
        resource_grid: Any,
        carrier_frequency: float = 6e9,  # 默认6 GHz
    ):
        """初始化感知性能计算类

        参数:
        -------
            - ofdm: OFDM对象（可以是OFDM类或兼容的包装对象）
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

    # ==================== 分辨率计算 ====================
    @property
    def delay_resolution(self) -> float:
        """时间分辨率 (秒)

        基于总带宽的倒数（用于完整的资源网格）

        返回值:
        -------
            - float: 时间分辨率，单位为秒
        """
        return 1 / self.rg.bandwidth

    @property
    def range_resolution(self) -> float:
        """距离分辨率 (米)

        根据时间分辨率计算，考虑往返时间

        返回值:
        -------
            - float: 距离分辨率，单位为米
        """
        return c * self.delay_resolution / 2

    @property
    def doppler_resolution(self) -> float:
        """多普勒频率分辨率 (Hz)

        根据OFDM符号持续时间和符号数计算

        返回值:
        -------
            - float: 多普勒频率分辨率，单位为Hz
        """
        return 1 / (self.rg.ofdm_symbol_duration * self.rg.num_ofdm_symbols)

    @property
    def velocity_resolution(self) -> float:
        """速度分辨率 (m/s)

        根据多普勒频率分辨率计算

        返回值:
        -------
            - float: 速度分辨率，单位为m/s
        """
        return self.doppler_resolution * c / (2 * self.carrier_frequency)

    # ==================== 感知单元 ====================
    @property
    def delay_bins(self) -> np.ndarray:
        """时间范围 (纳秒)

        基于总子载波数的时间分辨率（用于完整的资源网格）

        返回值:
        -------
            - np.ndarray: 时间范围数组，单位为纳秒
        """
        return np.arange(0, self.rg.fft_size) * self.delay_resolution / 1e-9

    @property
    def range_bins(self) -> np.ndarray:
        """距离范围 (米)

        基于有效子载波数的距离分辨率

        返回值:
        -------
            - np.ndarray: 距离范围数组，单位为米
        """
        return np.arange(0, self.rg.fft_size) * self.range_resolution

    @property
    def bistatic_range_resolution(self) -> float:
        """双基地折叠路径长度分辨率 (m/bin)：``ΔL = c · Δτ``（单程，无 /2）。"""
        return c * self.delay_resolution

    def range_bins_for(self, sens_mode: str = "monostatic") -> np.ndarray:
        """与 ``delay_to_range(bin·Δτ, …, sens_mode)`` 一致的逐 bin 距离轴 (m)。"""
        n = self.rg.fft_size
        k = np.arange(n, dtype=np.float64)
        if sens_mode == "monostatic":
            return k * self.range_resolution
        if sens_mode == "bistatic":
            return k * self.bistatic_range_resolution
        raise ValueError(f"不支持的 sens_mode: {sens_mode!r}")

    def near_delay_guard_bins(
        self, guard_range_m: float, sens_mode: str = "monostatic"
    ) -> int:
        """将近距保护物理距离 (m) 换算为跳过的时延 bin 数（``ceil(guard/res)``）。

        与 ``range_bins_for`` 一致：单基地用 ``range_resolution``，双基地用
        ``bistatic_range_resolution``。``guard_range_m <= 0`` 时不跳过任何 bin。
        """
        if guard_range_m <= 0:
            return 0
        if sens_mode == "monostatic":
            res = self.range_resolution
        elif sens_mode == "bistatic":
            res = self.bistatic_range_resolution
        else:
            raise ValueError(f"不支持的 sens_mode: {sens_mode!r}")
        if res <= 0:
            raise ValueError("距离分辨率须为正数")
        return int(math.ceil(guard_range_m / res))

    def velocity_bins_for(self, sens_mode: str = "monostatic") -> np.ndarray:
        """与 ``doppler_to_velocity`` 一致的逐 bin 速度轴 (m/s)。"""
        fd_hz = self.doppler_bins
        if sens_mode == "monostatic":
            return -(fd_hz * c) / (2.0 * self.carrier_frequency)
        if sens_mode == "bistatic":
            return -(fd_hz * c) / self.carrier_frequency
        raise ValueError(f"不支持的 sens_mode: {sens_mode!r}")

    @property
    def doppler_bins(self) -> np.ndarray:
        """多普勒频率范围 (Hz)

        基于OFDM符号数的频率分辨率

        返回值:
        -------
            - np.ndarray: 多普勒频率范围数组，单位为Hz
        """
        return np.arange(
            -self.rg.num_ofdm_symbols / 2 * self.doppler_resolution,
            self.rg.num_ofdm_symbols / 2 * self.doppler_resolution,
            self.doppler_resolution,
        )

    @property
    def velocity_bins(self) -> np.ndarray:
        """速度范围 (m/s)，单基地尺度；双基地请用 ``velocity_bins_for('bistatic')``。"""
        return self.velocity_bins_for("monostatic")

    # ==================== 最大探测范围 ====================
    @property
    def max_range(self) -> float:
        """最大探测距离 (米)

        返回值:
        -------
            - float: 最大探测距离，单位为米
        """
        return (self.rg.fft_size - 1) * self.range_resolution

    @property
    def max_velocity(self) -> float:
        """最大探测速度 (m/s)

        返回值:
        -------
            - float: 最大探测速度，单位为m/s
        """
        num_doppler_bins = self.rg.num_ofdm_symbols
        return (num_doppler_bins // 2) * self.velocity_resolution

    # ==================== 打印性能参数 ====================
    def display_performance(self):
        """打印感知性能参数"""
        table = [
            ["时间分辨率", f"{self.delay_resolution * 1e9:.2f}", "ns"],
            ["距离分辨率", f"{self.range_resolution:.2f}", "m"],
            ["多普勒分辨率", f"{self.doppler_resolution:.2f}", "Hz"],
            ["速度分辨率", f"{self.velocity_resolution:.2f}", "m/s"],
            ["最大探测距离", f"{self.max_range:.2f}", "m"],
            ["最大探测速度", f"{self.max_velocity:.2f}", "m/s"],
        ]
        print("感知性能参数:")
        print(tabulate(table, headers=["参数", "数值", "单位"], tablefmt="simple_grid"))
