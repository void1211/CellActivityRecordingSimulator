#!/usr/bin/env python3
"""
GPU最適化と新しい信号生成方式のテストスクリプト
"""

import sys
import numpy as np
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cellactivityrecodingsimulator.gpu_utils import gpu_manager, generate_unit_spike_signal_gpu
from cellactivityrecodingsimulator.Unit import Unit
from cellactivityrecodingsimulator.Contact import Contact
from cellactivityrecodingsimulator.Settings import Settings

def test_gpu_manager():
    """GPUマネージャーのテスト"""
    print("=== GPU Manager Test ===")
    device_info = gpu_manager.get_device_info()
    print(f"Device: {device_info}")
    print(f"GPU Available: {gpu_manager.is_gpu_available()}")
    print()

def test_unit_spike_signal_generation():
    """Unitのスパイク信号生成テスト"""
    print("=== Unit Spike Signal Generation Test ===")
    
    # テスト用の設定（既存設定ファイルを参考）
    settings = {
        "baseSettings": {
            "duration": 1.0,  # 1秒
            "fs": 30000,      # 30kHz
            "random_seed": 42
        },
        "spikeSettings": {
            "spikeType": "exponential",  # 指数テンプレートを使用
            "rate": 10.0,     # 10 Hz
            "refractoryPeriod": 2.0,  # 2ms
            "amplitudeMin": 50.0,
            "amplitudeMax": 100.0,
            "attenTime": 50.0,
            "exponential": {
                "randType": "list",
                "ms_before": [0.4, 0.5, 0.6],
                "ms_after": [1.0, 1.2, 1.5],
                "negative_amplitude": [-0.8, -0.9, -1.0],
                "positive_amplitude": [0.2, 0.3, 0.4],
                "depolarization_ms": [0.1, 0.15, 0.2],
                "repolarization_ms": [0.3, 0.4, 0.5],
                "recovery_ms": [0.5, 0.6, 0.7],
                "smooth_ms": [0.05, 0.1, 0.15]
            }
        }
    }
    
    # Unitを作成
    unit = Unit.generate(id=0, group=0, x=0.0, y=0.0, z=0.0)
    unit.set_spike_time(settings=settings)
    unit.set_amplitudes(settings=settings)
    unit.set_templateObject(settings=settings)
    
    print(f"Unit {unit.id}: {len(unit.spikeTimeList)} spikes")
    print(f"Spike times: {unit.spikeTimeList[:5]}...")  # 最初の5個
    print(f"Amplitudes: {unit.spikeAmpList[:5]}...")    # 最初の5個
    
    # スパイク信号を生成
    spike_signal = unit.generate_spike_signal(settings)
    print(f"Generated signal length: {len(spike_signal)} samples")
    print(f"Signal range: [{np.min(spike_signal):.2f}, {np.max(spike_signal):.2f}]")
    print(f"Non-zero samples: {np.count_nonzero(spike_signal)}")
    print()

def test_contact_signal_generation():
    """Contactの信号生成テスト"""
    print("=== Contact Signal Generation Test ===")
    
    # テスト用の設定（既存設定ファイルを参考）
    settings = {
        "baseSettings": {
            "duration": 1.0,  # 1秒
            "fs": 30000,      # 30kHz
            "random_seed": 42
        },
        "spikeSettings": {
            "spikeType": "exponential",  # 指数テンプレートを使用
            "rate": 10.0,     # 10 Hz
            "refractoryPeriod": 2.0,  # 2ms
            "amplitudeMin": 50.0,
            "amplitudeMax": 100.0,
            "attenTime": 50.0,
            "exponential": {
                "randType": "list",
                "ms_before": [0.4, 0.5, 0.6],
                "ms_after": [1.0, 1.2, 1.5],
                "negative_amplitude": [-0.8, -0.9, -1.0],
                "positive_amplitude": [0.2, 0.3, 0.4],
                "depolarization_ms": [0.1, 0.15, 0.2],
                "repolarization_ms": [0.3, 0.4, 0.5],
                "recovery_ms": [0.5, 0.6, 0.7],
                "smooth_ms": [0.05, 0.1, 0.15]
            }
        }
    }
    
    # Unitを作成
    unit = Unit.generate(id=0, group=0, x=0.0, y=0.0, z=0.0)
    unit.set_spike_time(settings=settings)
    unit.set_amplitudes(settings=settings)
    unit.set_templateObject(settings=settings)
    unit.generate_spike_signal(settings)
    
    # Contactを作成
    contact = Contact.generate(settings, id=0, x=10.0, y=10.0, z=0.0)
    
    print(f"Unit position: [{unit.x}, {unit.y}, {unit.z}]")
    print(f"Contact position: [{contact.x}, {contact.y}, {contact.z}]")
    
    # ContactにUnitのスパイク信号を追加
    contact.add_unit_spike_signal(unit, settings)
    
    contact_signal = contact.get_signal("spike")
    print(f"Contact signal length: {len(contact_signal)} samples")
    print(f"Contact signal range: [{np.min(contact_signal):.2f}, {np.max(contact_signal):.2f}]")
    print(f"Contact non-zero samples: {np.count_nonzero(contact_signal)}")
    print()

def test_multiple_units():
    """複数Unitのテスト"""
    print("=== Multiple Units Test ===")
    
    # テスト用の設定（既存設定ファイルを参考）
    settings = {
        "baseSettings": {
            "duration": 1.0,  # 1秒
            "fs": 30000,      # 30kHz
            "random_seed": 42
        },
        "spikeSettings": {
            "spikeType": "exponential",  # 指数テンプレートを使用
            "rate": 10.0,     # 10 Hz
            "refractoryPeriod": 2.0,  # 2ms
            "amplitudeMin": 50.0,
            "amplitudeMax": 100.0,
            "attenTime": 50.0,
            "exponential": {
                "randType": "list",
                "ms_before": [0.4, 0.5, 0.6],
                "ms_after": [1.0, 1.2, 1.5],
                "negative_amplitude": [-0.8, -0.9, -1.0],
                "positive_amplitude": [0.2, 0.3, 0.4],
                "depolarization_ms": [0.1, 0.15, 0.2],
                "repolarization_ms": [0.3, 0.4, 0.5],
                "recovery_ms": [0.5, 0.6, 0.7],
                "smooth_ms": [0.05, 0.1, 0.15]
            }
        }
    }
    
    # 複数のUnitを作成
    units = []
    for i in range(3):
        unit = Unit.generate(id=i, group=0, x=i*20.0, y=0.0, z=0.0)
        unit.set_spike_time(settings=settings)
        unit.set_amplitudes(settings=settings)
        unit.set_templateObject(settings=settings)
        unit.generate_spike_signal(settings)
        units.append(unit)
    
    # Contactを作成
    contact = Contact.generate(settings, id=0, x=10.0, y=0.0, z=0.0)
    
    # 各Unitのスパイク信号をContactに追加
    for unit in units:
        print(f"Adding Unit {unit.id} at position [{unit.x}, {unit.y}, {unit.z}]")
        contact.add_unit_spike_signal(unit, settings)
    
    contact_signal = contact.get_signal("spike")
    print(f"Final contact signal range: [{np.min(contact_signal):.2f}, {np.max(contact_signal):.2f}]")
    print(f"Final contact non-zero samples: {np.count_nonzero(contact_signal)}")
    print()

def main():
    """メイン関数"""
    print("GPU Optimization and New Signal Generation Test")
    print("=" * 50)
    
    try:
        test_gpu_manager()
        test_unit_spike_signal_generation()
        test_contact_signal_generation()
        test_multiple_units()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
