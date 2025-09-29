"""
ODE模块适配器包

这个包提供了各种输入输出适配器，让ODE模块能够处理不同格式的数据。
"""

from .input_adapters import (
    TimeSeriesAdapter,
    ImageSequenceAdapter,
    PointCloudAdapter,
    MultiModalAdapter,
    create_adapter
)

from .output_adapters import (
    TimeSeriesOutputAdapter,
    ImageSequenceOutputAdapter,
    PointCloudOutputAdapter,
    SegmentationOutputAdapter,
    create_output_adapter
)

from .wrapper_examples import (
    TimeSeriesODEWrapper,
    VideoODEWrapper,
    LidarODEWrapper,
    SegmentationODEWrapper,
    CustomODEWrapper,
    create_ode_wrapper
)

__all__ = [
    # 输入适配器
    'TimeSeriesAdapter',
    'ImageSequenceAdapter',
    'PointCloudAdapter',
    'MultiModalAdapter',
    'create_adapter',

    # 输出适配器
    'TimeSeriesOutputAdapter',
    'ImageSequenceOutputAdapter',
    'PointCloudOutputAdapter',
    'SegmentationOutputAdapter',
    'create_output_adapter',

    # 完整包装器
    'TimeSeriesODEWrapper',
    'VideoODEWrapper',
    'LidarODEWrapper',
    'SegmentationODEWrapper',
    'CustomODEWrapper',
    'create_ode_wrapper'
]