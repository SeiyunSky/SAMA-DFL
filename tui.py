"""
SAMA-DFL Experiment Dashboard (TUI)
交互式实验管理界面
"""
import sys
import subprocess
import threading
import time
import os
from pathlib import Path

# Windows UTF-8 编码修复（必须在 rich 导入前）
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

from utils import check_all_datasets, download_dataset

DATA_DIR = str(PROJ_DIR / 'data')

console = Console(width=80)

THEORY_EXPERIMENTS = {'lemma41', 'convergence', 'kappa', 'consensus', 'lyapunov'}
PERF_MAX_PARALLEL = 1  # 性能实验内部已用 ProcessPoolExecutor，外层必须严格串行


# ──────────────────────────────────────────────────────────
# GPU 信息
# ──────────────────────────────────────────────────────────

def get_gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(',')
        return {
            'mem_used': int(parts[0].strip()),
            'mem_total': int(parts[1].strip()),
            'util': int(parts[2].strip()),
            'temp': int(parts[3].strip()),
        }
    except Exception:
        return None


def render_gpu_panel():
    info = get_gpu_info()
    if info is None:
        return Panel("[dim]GPU 信息不可用[/dim]", title="GPU", border_style="dim")

    pct = info['mem_used'] / info['mem_total']
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "#" * filled + "." * (bar_len - filled)

    color = "green" if pct < 0.7 else ("yellow" if pct < 0.9 else "red")
    mem_text = Text()
    mem_text.append(f"MEM  [{bar}] ", style="white")
    mem_text.append(f"{info['mem_used']}/{info['mem_total']} MB ({pct:.0%})", style=color)

    util_pct = info['util'] / 100
    u_filled = int(bar_len * util_pct)
    u_bar = "#" * u_filled + "." * (bar_len - u_filled)
    u_color = "green" if util_pct < 0.5 else ("yellow" if util_pct < 0.85 else "red")
    util_text = Text()
    util_text.append(f"UTIL [{u_bar}] ", style="white")
    util_text.append(f"{info['util']}%", style=u_color)
    util_text.append(f"   Temp {info['temp']}C",
                     style="green" if info['temp'] < 75 else "yellow")

    content = Text.assemble(mem_text, "\n", util_text)
    return Panel(content, title="[bold cyan]GPU[/bold cyan]", border_style="cyan")


# ──────────────────────────────────────────────────────────
# 结果扫描
# ──────────────────────────────────────────────────────────

def scan_results():
    results_dir = PROJ_DIR / 'results'
    if not results_dir.exists():
        return []
    files = list(results_dir.glob('*.png'))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def render_results_table():
    files = scan_results()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", expand=True)
    table.add_column("文件名", style="cyan", no_wrap=True)
    table.add_column("大小", justify="right", style="green")
    table.add_column("时间", style="dim")

    if not files:
        table.add_row("[dim]暂无结果文件[/dim]", "", "")
    else:
        for f in files[:8]:
            size = f"{f.stat().st_size / 1024:.1f} KB"
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%m-%d %H:%M')
            table.add_row(f.name, size, mtime)

    return Panel(table, title=f"[bold magenta]已有结果 ({len(files)} 个)[/bold magenta]",
                 border_style="magenta")


# ──────────────────────────────────────────────────────────
# 数据集检测与下载
# ──────────────────────────────────────────────────────────

def render_dataset_panel(status: dict):
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("数据集", style="bold", width=12)
    table.add_column("状态")
    for name, ok in status.items():
        if ok:
            table.add_row(name.upper(), "[green]OK[/green]")
        else:
            table.add_row(name.upper(), "[red]缺失  (未下载)[/red]")
    all_ok = all(status.values())
    border = "green" if all_ok else "yellow"
    title = "[bold green]数据集状态[/bold green]" if all_ok else "[bold yellow]数据集状态  (有缺失)[/bold yellow]"
    return Panel(table, title=title, border_style=border)


def download_with_progress(name: str):
    script = (
        f"import sys; sys.path.insert(0, r'{PROJ_DIR}'); "
        f"from utils import download_dataset; "
        f"download_dataset('{name}', r'{DATA_DIR}')"
    )
    cmd = [sys.executable, '-u', '-c', script]
    console.print(f"\n[bold cyan]正在下载 {name.upper()}...[/bold cyan]")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=str(PROJ_DIR))
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip())
            if len(lines) > 10:
                lines = lines[-10:]
            text = Text()
            for l in lines:
                text.append(l + "\n", style="dim white")
            live.update(Panel(text, title=f"[cyan]下载 {name.upper()}[/cyan]", border_style="cyan"))
        proc.wait()
    if proc.returncode == 0:
        console.print(f"[green]✓ {name.upper()} 下载完成[/green]")
    else:
        console.print(f"[red]✗ {name.upper()} 下载失败 (exit {proc.returncode})[/red]")


def check_and_prompt_datasets():
    status = check_all_datasets(DATA_DIR)
    console.print(render_dataset_panel(status))
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return
    console.print(f"[yellow]缺失数据集: {', '.join(m.upper() for m in missing)}[/yellow]")
    if Confirm.ask("是否现在下载缺失的数据集?", default=True):
        for name in missing:
            download_with_progress(name)
        console.print(render_dataset_panel(check_all_datasets(DATA_DIR)))


# ──────────────────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────────────────

EXPERIMENTS = {
    'A': {
        'label': '理论验证实验',
        'items': [
            ('A1', 'lemma41',    '引理4.1  幅度对齐公式验证'),
            ('A2', 'convergence','收敛速率测量'),
            ('A3', 'kappa',      'κ 值对比 SAMA vs BALANCE'),
            ('A4', 'consensus',  '共识直径 vs 理论上界'),
            ('A5', 'lyapunov',   'Lyapunov 函数单调性'),
        ]
    },
    'B': {
        'label': '性能实验',
        'items': [
            ('B1', 'multi_attack_table',  'MNIST 多攻击汇总（8方法×6攻击）'),
            ('B2', 'cifar10_attack_table','CIFAR-10 多攻击汇总（8方法×6攻击）'),
            ('B3', 'byz_sweep',           '拜占庭比例扫描 (0.1~0.4)'),
            ('B4', 'noniid_sweep',        'Non-IID 程度扫描 (α=0.1/0.2/0.3)'),
            ('B5', 'ablation',            '消融实验（6种变体）'),
            ('B6', 'client_scale',        '客户端数量扩展性（n=20/30/40）'),
        ]
    },
}

PERF_GROUPS = {'B'}
NO_ATTACK_PROMPT = {'multi_attack_table', 'cifar10_attack_table', 'client_scale'}


# ──────────────────────────────────────────────────────────
# 单个实验进程
# ──────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self, exp_key, label, attack=None):
        self.exp_key = exp_key
        self.label = label
        self.attack = attack
        self.running = False
        self.done = False
        self.returncode = None
        self.output_lines = []
        self.start_time = None
        self.proc = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.start_time = datetime.now()
        env = os.environ.copy()
        if self.attack:
            env['ATTACK_TYPE'] = self.attack
        cmd = [sys.executable, '-u', str(PROJ_DIR / 'run_experiments.py'),
               '--experiment', self.exp_key]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(PROJ_DIR),
            bufsize=0,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        # 用底层二进制 read1，读多少返回多少，避免阻塞凑满 buffer
        import io
        raw = self.proc.stdout.buffer if isinstance(self.proc.stdout, io.TextIOBase) else self.proc.stdout
        buf = ""
        while True:
            try:
                chunk_bytes = raw.read1(4096) if hasattr(raw, 'read1') else raw.read(1)
            except Exception:
                break
            if not chunk_bytes:
                break
            chunk = chunk_bytes.decode('utf-8', errors='replace') if isinstance(chunk_bytes, bytes) else chunk_bytes
            buf += chunk
            while '\n' in buf or '\r' in buf:
                nl = buf.find('\n')
                cr = buf.find('\r')
                if nl == -1:
                    idx, is_cr = cr, True
                elif cr == -1:
                    idx, is_cr = nl, False
                else:
                    idx, is_cr = (cr, True) if cr < nl else (nl, False)

                line = buf[:idx].rstrip('\r\n')
                buf = buf[idx+1:]

                if not line:
                    continue
                with self._lock:
                    if is_cr and self.output_lines:
                        # \r 覆写最后一行（tqdm 进度条行为）
                        self.output_lines[-1] = line
                    else:
                        self.output_lines.append(line)
                        if len(self.output_lines) > 200:
                            self.output_lines = self.output_lines[-200:]
        self.proc.wait()
        self.returncode = self.proc.returncode
        self.running = False
        self.done = True

    def stop(self):
        if self.proc and self.running:
            self.proc.terminate()
            self.running = False

    def get_last_lines(self, n=6):
        with self._lock:
            return list(self.output_lines[-n:])

    def elapsed(self):
        if self.start_time is None:
            return ""
        s = int((datetime.now() - self.start_time).total_seconds())
        return f"{s // 60:02d}:{s % 60:02d}"

    def status_str(self):
        if self.done:
            return "[green]完成[/green]" if self.returncode == 0 else f"[red]失败({self.returncode})[/red]"
        if self.running:
            return f"[cyan]运行中 {self.elapsed()}[/cyan]"
        return "[dim]等待[/dim]"


# ──────────────────────────────────────────────────────────
# 并行批量运行（显存动态调度）
# ──────────────────────────────────────────────────────────

def run_parallel(jobs: list):
    """
    理论实验全部并行；性能实验最多同时跑 PERF_MAX_PARALLEL 个。
    理论实验先全部启动，性能实验串行补入。
    """
    theory_jobs = [j for j in jobs if j.exp_key in THEORY_EXPERIMENTS]
    perf_jobs = [j for j in jobs if j.exp_key not in THEORY_EXPERIMENTS]

    pending_perf = list(perf_jobs)
    active_perf = []
    interrupted = False

    # 理论实验全部立即启动
    for j in theory_jobs:
        j.start()

    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                # 移除已完成的性能实验
                active_perf = [j for j in active_perf if not j.done]

                # 补性能实验：未超上限就补
                while pending_perf and len(active_perf) < PERF_MAX_PARALLEL:
                    j = pending_perf.pop(0)
                    j.start()
                    active_perf.append(j)

                all_active = [j for j in theory_jobs if j.running] + active_perf
                live.update(_render_parallel_status(jobs, all_active, pending_perf))
                time.sleep(0.5)

                # 全部完成则退出
                if all(j.done for j in jobs):
                    break

    except KeyboardInterrupt:
        interrupted = True
        for j in theory_jobs + active_perf:
            j.stop()
        console.print("\n[yellow]已中断，正在终止所有子进程...[/yellow]")

    if not interrupted:
        console.print(Rule("[bold green]全部实验完成[/bold green]"))
        for job in jobs:
            mark = "[green]✓[/green]" if job.returncode == 0 else "[red]✗[/red]"
            console.print(f"  {mark} {job.label}  耗时 {job.elapsed()}")


def _render_parallel_status(all_jobs, active, pending):
    gpu = render_gpu_panel()

    # 分组：完成、运行中、等待
    done_jobs    = [j for j in all_jobs if j.done]
    running_jobs = [j for j in all_jobs if j.running]
    waiting_jobs = [j for j in all_jobs if not j.done and not j.running]

    renderables = []

    # ── 运行中：每个实验展开显示最近 4 行输出 ──────────────────
    if running_jobs:
        for job in running_jobs:
            lines = job.get_last_lines(4)
            log_text = Text()
            for line in lines:
                if 'ERROR' in line or 'Error' in line:
                    log_text.append(f"  {line}\n", style="red")
                elif '%' in line or 'accuracy' in line.lower() or 'acc=' in line.lower() or 'done' in line:
                    log_text.append(f"  {line}\n", style="green")
                elif 'round' in line.lower() or 'Round' in line:
                    log_text.append(f"  {line}\n", style="cyan")
                else:
                    log_text.append(f"  {line}\n", style="dim white")
            renderables.append(Panel(
                log_text,
                title=f"[bold cyan]{job.label}  {job.elapsed()}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))

    # ── 完成 / 等待：紧凑表格 ──────────────────────────────────
    if done_jobs or waiting_jobs:
        table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        table.add_column("实验", style="white")
        table.add_column("状态", width=18)
        table.add_column("最新输出", style="dim", max_width=55)

        for job in done_jobs:
            last = job.get_last_lines(1)
            last_line = last[-1][:52] + "..." if last and len(last[-1]) > 52 else (last[-1] if last else "")
            table.add_row(job.label, job.status_str(), last_line)

        for job in waiting_jobs:
            table.add_row(job.label, "[dim]等待[/dim]", "")

        renderables.append(Panel(table, border_style="dim", padding=(0, 1)))

    slots_text = f"运行: {len(running_jobs)}  完成: {len(done_jobs)}  排队: {len(waiting_jobs)}"
    jobs_panel = Panel(
        Group(*renderables) if renderables else Text(""),
        title=f"[bold cyan]实验进度  {slots_text}[/bold cyan]",
        border_style="cyan",
    )

    layout = Layout()
    layout.split_column(
        Layout(gpu, size=4),
        Layout(jobs_panel),
    )
    return layout


# ──────────────────────────────────────────────────────────
# 主菜单渲染
# ──────────────────────────────────────────────────────────

def render_menu():
    table = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    table.add_column("组", style="bold yellow", width=4)
    table.add_column("编号", style="bold cyan", width=4)
    table.add_column("描述", style="white")

    for group_key, group in EXPERIMENTS.items():
        table.add_row(f"[{group_key}]", "", f"[bold]{group['label']}[/bold]")
        for idx_key, exp_key, label in group['items']:
            table.add_row("", idx_key, label)

    table.add_row("", "", "")
    table.add_row("[*]", "all",  "[dim]并行运行全部实验[/dim]")
    table.add_row("[Q]", "quit", "[dim]退出[/dim]")

    return Panel(table, title="[bold white]SAMA-DFL Experiment Dashboard[/bold white]",
                 border_style="blue")


# ──────────────────────────────────────────────────────────
# 单个实验（单独运行时仍保留 Live 输出）
# ──────────────────────────────────────────────────────────

def run_single(exp_key, label, attack=None):
    job = ExperimentRunner(exp_key, label, attack)
    job.start()

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while job.running:
                lines = job.get_last_lines(14)
                log_text = Text()
                for line in lines:
                    if 'ERROR' in line or 'Error' in line or 'FAILED' in line:
                        log_text.append(line + "\n", style="red")
                    elif 'acc=' in line.lower() or 'accuracy' in line.lower() or '%' in line:
                        log_text.append(line + "\n", style="green")
                    elif 'Round' in line or 'round' in line or 'Epoch' in line:
                        log_text.append(line + "\n", style="cyan")
                    elif line.startswith('[') or '===' in line:
                        log_text.append(line + "\n", style="yellow")
                    else:
                        log_text.append(line + "\n", style="dim white")

                layout = Layout()
                layout.split_column(
                    Layout(render_gpu_panel(), size=4),
                    Layout(Panel(log_text,
                                 title=f"[bold cyan]{label}  {job.elapsed()}[/bold cyan]",
                                 border_style="cyan")),
                    Layout(Panel(
                        f"[dim]攻击: [yellow]{attack or '默认'}[/yellow]    按 Ctrl+C 中断[/dim]",
                        border_style="dim"
                    ), size=3),
                )
                live.update(layout)
                time.sleep(0.25)
    except KeyboardInterrupt:
        job.stop()
        console.print("\n[yellow]已中断[/yellow]")
        return

    if job.returncode == 0:
        console.print(Rule(f"[green]✓ {label} 完成  耗时 {job.elapsed()}[/green]"))
    else:
        console.print(Rule(f"[red]✗ {label} 失败 (exit {job.returncode})[/red]"))
    for line in job.get_last_lines(6):
        console.print(f"  [dim]{line}[/dim]")


# ──────────────────────────────────────────────────────────
# 攻击类型选择
# ──────────────────────────────────────────────────────────

def choose_attack():
    console.print()
    console.print("[bold]选择攻击类型[/bold]（留空使用配置文件默认值）")
    console.print("  [cyan]1[/cyan] Gaussian        [cyan]2[/cyan] Label Flipping")
    console.print("  [cyan]3[/cyan] Omniscient      [cyan]4[/cyan] Krum Attack")
    console.print("  [cyan]5[/cyan] Trim Attack")
    choice = Prompt.ask("攻击", default="").strip()
    mapping = {
        '1': 'gaussian', '2': 'label_flipping',
        '3': 'omniscient', '4': 'krum_attack', '5': 'trim_attack',
        'gaussian': 'gaussian', 'label_flipping': 'label_flipping',
        'omniscient': 'omniscient', 'krum_attack': 'krum_attack',
        'trim_attack': 'trim_attack',
    }
    return mapping.get(choice, None)


# ──────────────────────────────────────────────────────────
# 查找实验
# ──────────────────────────────────────────────────────────

def resolve_experiment(choice):
    choice = choice.strip().upper()
    for group_key, group in EXPERIMENTS.items():
        for idx_key, exp_key, label in group['items']:
            if choice in (idx_key.upper(), exp_key.upper(), exp_key.lower()):
                needs_attack = group_key in PERF_GROUPS and exp_key not in NO_ATTACK_PROMPT
                return exp_key, label, needs_attack
    return None


# ──────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────

def main():
    console.clear()
    console.print(render_gpu_panel())
    check_and_prompt_datasets()
    console.print(render_results_table())

    while True:
        console.print()
        console.print(render_menu())
        console.print()

        choice = Prompt.ask("[bold white]输入编号或命令[/bold white]",
                            default="").strip().lower()

        if choice in ('q', 'quit', 'exit'):
            console.print("\n[dim]Bye.[/dim]")
            break

        if choice == 'all':
            jobs = []
            for group_key, group in EXPERIMENTS.items():
                for idx_key, exp_key, label in group['items']:
                    jobs.append(ExperimentRunner(exp_key, label, None))
            run_parallel(jobs)
            console.print()
            console.print(render_results_table())
            continue

        # 整组并行运行
        if choice.upper() in EXPERIMENTS:
            group_key = choice.upper()
            attack = choose_attack() if group_key in PERF_GROUPS else None
            group = EXPERIMENTS[group_key]
            jobs = [
                ExperimentRunner(exp_key, label,
                                 None if exp_key in NO_ATTACK_PROMPT else attack)
                for _, exp_key, label in group['items']
            ]
            run_parallel(jobs)
            console.print()
            console.print(render_results_table())
            continue

        # 单个实验
        result = resolve_experiment(choice)
        if result is None:
            console.print(f"[red]未知实验: {choice}[/red]")
            continue

        exp_key, label, needs_attack = result
        attack = choose_attack() if needs_attack else None
        run_single(exp_key, label, attack)

        console.print()
        console.print(render_results_table())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]退出[/dim]")
