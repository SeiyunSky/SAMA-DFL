"""
Logging utilities for tracking experiments
"""
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 轻量任务 log：每个 (method, attack, param) 任务一个文件，带完整头信息
# 适合在 spawn 子进程中直接 open/write，无需 logging 模块
# ──────────────────────────────────────────────────────────────────────────────

def make_run_log_dir(exp_name: str, run_timestamp: str, base_dir: str = None) -> Path:
    """
    返回本次实验运行的 log 目录（已创建）。
    路径：{base_dir}/logs/{run_timestamp}_{exp_name}/
    run_timestamp 由调用方统一生成（主进程），子进程复用，保证同一次运行落同一文件夹。
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'results'
    log_dir = Path(base_dir) / 'logs' / f"{run_timestamp}_{exp_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def open_task_log(log_dir: Path, filename: str, meta: dict) -> Path:
    """
    创建单个任务的 log 文件，写入结构化头信息，返回文件路径。

    meta 字段（全部可选，建议包含）：
        exp_name, method, attack, 以及实验特有参数（byz_ratio / noniid_alpha / n_clients 等）、
        num_rounds, lr, batch_size, num_clients, topology 等

    调用方后续用 append_task_log() 按轮写入 acc。
    """
    log_path = log_dir / filename
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("# " + "=" * 62 + "\n")
        f.write(f"# Experiment  : {meta.get('exp_name', 'unknown')}\n")
        f.write(f"# Started at  : {now}\n")
        f.write("# " + "-" * 62 + "\n")
        for key, val in meta.items():
            if key == 'exp_name':
                continue
            f.write(f"# {key:<14}: {val}\n")
        f.write("# " + "=" * 62 + "\n")
    return log_path


def append_task_log(log_path: Path, round_num: int, num_rounds: int, acc: float):
    """向任务 log 文件追加一行：round=X/Y  acc=ZZ.ZZ%"""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"round={round_num}/{num_rounds}  acc={acc:.2f}%\n")


def finalize_task_log(log_path: Path, final_acc: float, elapsed_sec: float = None):
    """在任务 log 末尾写最终结果行。"""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("# " + "-" * 62 + "\n")
        f.write(f"# final_acc   : {final_acc:.2f}%\n")
        if elapsed_sec is not None:
            f.write(f"# elapsed     : {elapsed_sec:.1f}s\n")
        f.write("# " + "=" * 62 + "\n")



class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, exp_name, log_dir='./results'):
        self.exp_name = exp_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{exp_name}_{timestamp}.log"

        # 配置logger
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)

        # 文件handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # 控制台handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.metrics_history = []

    def log_config(self, config):
        """记录配置"""
        self.logger.info("=" * 60)
        self.logger.info("Experiment Configuration")
        self.logger.info("=" * 60)
        self.logger.info(json.dumps(config, indent=2, ensure_ascii=False))

    def log_round(self, round_num, metrics):
        """
        记录每轮指标

        参数:
            round_num: int
            metrics: dict - 指标字典
        """
        metrics['round'] = round_num
        self.metrics_history.append(metrics)

        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items() if k != 'round'])
        self.logger.info(f"Round {round_num}: {metric_str}")

    def log_final(self, final_metrics):
        """记录最终结果"""
        self.logger.info("=" * 60)
        self.logger.info("Final Results")
        self.logger.info("=" * 60)
        for key, value in final_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")

    def save_metrics(self, filename=None):
        """保存指标历史到JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.exp_name}_metrics_{timestamp}.json"

        save_path = self.log_dir / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics saved to: {save_path}")
        return save_path


def setup_logger(name, log_file, level=logging.INFO):
    """
    配置简单logger

    参数:
        name: logger名称
        log_file: 日志文件路径
        level: 日志级别

    返回:
        logging.Logger
    """
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class MetricsTracker:
    """简单指标跟踪器"""

    def __init__(self):
        self.history = {}

    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get(self, key):
        """获取指标历史"""
        return self.history.get(key, [])

    def get_latest(self, key):
        """获取最新指标"""
        values = self.history.get(key, [])
        return values[-1] if values else None

    def get_mean(self, key, last_n=None):
        """获取指标平均值"""
        values = self.history.get(key, [])
        if not values:
            return None
        if last_n is not None:
            values = values[-last_n:]
        return np.mean(values)

    def summary(self):
        """打印摘要"""
        print("\nMetrics Summary:")
        print("-" * 40)
        for key, values in self.history.items():
            if values:
                print(f"{key:20s}: latest={values[-1]:.4f}, mean={np.mean(values):.4f}")


__all__ = ['ExperimentLogger', 'setup_logger', 'MetricsTracker']
