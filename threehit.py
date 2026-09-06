# 是否保存 DEBUG 图片和搜索记录；图片由后台线程写入。
DEBUG_SAVE_IMAGES = True
# 是否输出每轮搜索的分阶段耗时。
PRINT_SEARCH_BENCHMARK = True
# 是否只截棋盘区域；点击仍使用屏幕绝对坐标。
CAPTURE_BOARD_ONLY = True

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
import ctypes
import math
import os
import queue
import random
import statistics
import threading
import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui


# 棋盘与识别配置：颜色面积用于判级，连通域和中心距离用于筛选区域。
ROWS = 3
COLS = 9
REFERENCE_BOARD_WIDTH = 862
REFERENCE_BOARD_HEIGHT = 288
MIN_BOARD_WIDTH = REFERENCE_BOARD_WIDTH / 4
MIN_BOARD_HEIGHT = REFERENCE_BOARD_HEIGHT / 4
CENTER_RATIO = 0.82
BACKGROUND_DISTANCE = 35
MIN_COLOR_RATIO = 0.005
# 颜色像素面积 / 中央识别区域面积；12、23 分别为等级分界。
LEVEL_THRESHOLDS = {
    "f": {"12": 0.165, "23": 0.260},
    "r": {"12": 0.285, "23": 0.410},
    "w": {"12": 0.315, "23": 0.555},
}
MASK_OPEN_KERNEL = np.ones((3, 3), np.uint8)
MASK_CLOSE_KERNEL = np.ones((5, 5), np.uint8)


# 搜索配置：候选数量和样本数量决定主要计算量。
FAST_FILTER_LIMIT = 12
# 当前连锁优先展开概率分支；超预算后只采样未完成分支。
CURRENT_CHAIN_NODE_LIMIT = 6144
CURRENT_CHAIN_SAMPLE_COUNT = 54
# 前瞻只抽取这些补球后局面，不随当前连锁的备用样本数增加。
LOOKAHEAD_SAMPLE_COUNT = 27
NEXT_SWAP_CANDIDATE_LIMIT = 3
NEXT_CHAIN_NODE_LIMIT = 48
NEXT_CHAIN_SAMPLE_COUNT = 9
# 收益接近时放宽边界竞争；最多轻量评估 11 个，最终仍选 12 个。
FILTER_PROTECTED_COUNT = 9
FILTER_MIN_PROTECTED_COUNT = 7
FILTER_EXTRA_CANDIDATES = 4
FILTER_MAX_EXTRA_CANDIDATES = 6
CANDIDATE_NEAR_GAIN = 2.0
FILTER_CHAIN_NODE_LIMIT = 96
FILTER_CHAIN_SAMPLE_COUNT = 9
EVALUATION_WORKERS = 8
LOOKAHEAD_SEED_BASE = 0x5A17C9
# 20 步及以后仅评估当前收益，不强制限制交换次数。
TOTAL_GAME_MOVES = 20
TYPE_ORDER = ("f", "r", "w")
BALL_ENERGY = {
    "f": {1: 5.0, 2: 20.0, 3: 80.0},
    "r": {1: 4.0, 2: 16.0, 3: 64.0},
    "w": {1: 3.0, 2: 12.0, 3: 48.0},
}
# 完整棋盘每合成一次净增两个空位；等概率补一级球，首次补球期望为 8。
# 筛选与最终评估均直接使用首次补球期望，归入交换收益。
EXPECTED_REFILL_ENERGY = sum(BALL_ENERGY[color][1] for color in TYPE_ORDER) / len(TYPE_ORDER)
EXPECTED_REFILL_GAIN_PER_MERGE = 2 * EXPECTED_REFILL_ENERGY
# 概率分支中的 -1 表示尚未展开颜色的一级球，0 表示真实空位。
BRANCH_BALLS = (None,) + tuple(
    (color, level) for level in (1, 2, 3) for color in TYPE_ORDER
)
BRANCH_BALL_CODES = {ball: index for index, ball in enumerate(BRANCH_BALLS)}
COLOR_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
# 9 个样本的四个取色方向：a、b、a+b、a+2b（模 3）。
# 任意连续四个空位中，每对空位都覆盖全部 9 种颜色配对。
BALANCED_SAMPLE_COLUMNS = tuple(
    tuple((a * (i // 3) + b * (i % 3)) % 3 for i in range(9))
    for a, b in ((1, 0), (0, 1), (1, 1), (1, 2))
)
ALL_SWAPS = tuple(
    ((r1, c1), (r2, c2))
    for r1 in range(ROWS)
    for c1 in range(COLS)
    for r2 in range(r1, ROWS)
    for c2 in range(c1 + 1 if r2 == r1 else 0, COLS)
)
# 横向 21 个、竖向 9 个三格窗口，使用展开后的格子编号。
TRIPLE_WINDOWS = tuple(
    (r * COLS + c, r * COLS + c + 1, r * COLS + c + 2)
    for r in range(ROWS)
    for c in range(COLS - 2)
) + tuple((c, COLS + c, 2 * COLS + c) for c in range(COLS))


# 点击与等待配置。
AUTO_CLICK_ENABLED = True
AUTO_CLICK_DELAY = 0.12
MOUSE_MOVE_OUT_X = 10
MOUSE_MOVE_OUT_Y = 10
MIN_WAIT_AFTER_SWAP = 0.20
FIRST_STEP_MIN_WAIT_AFTER_SWAP = 0.80
INITIAL_BOARD_STABLE_TIME = 0.80
BOARD_CHECK_INTERVAL = 0.08
BOARD_STABLE_TIME = 0.25
MAX_WAIT_AFTER_SWAP = 3.00
# 异常运行保护上限；当前配置为 1200 秒，不是游戏的 81 秒倒计时。
GAME_MAX_SECONDS = 1200.0


# DEBUG 路径与 BGR 标注颜色。
DEBUG_DIR = "debug"
BOARD_DEBUG_DIR = os.path.join(DEBUG_DIR, "board")
SEARCH_DEBUG_DIR = os.path.join(DEBUG_DIR, "search")
DRAW_COLOR = {
    "f": (0, 0, 255),
    "r": (255, 0, 255),
    "w": (255, 255, 0),
    "?": (0, 255, 255),
}


# 运行状态：热键只更新状态，后台图片线程只处理队列任务。
AUTO_PAUSED = False
AUTO_STOP_REQUESTED = False
HOTKEYS_REGISTERED = False
RESELECT_REQUESTED = threading.Event()
RESELECT_RESULT = object()
RETRY_RESULT = object()
SELECTION_OVERLAY = None
DEBUG_WRITE_QUEUE = queue.Queue()
DEBUG_WRITER_THREAD = None
DEBUG_WRITER_STOP = object()


# 区域截图与网格校准


def get_board_capture_monitor(x1, y1, dx, dy, full_monitor):
    """根据校准结果生成棋盘截图区域。"""
    if not CAPTURE_BOARD_ONLY:
        return full_monitor

    left_center = x1
    right_center = x1 + (COLS - 1) * dx
    top_center = y1
    bottom_center = y1 + (ROWS - 1) * dy

    margin_x = abs(dx) * 0.50
    margin_y = abs(dy) * 0.50

    left = math.floor(min(left_center, right_center) - margin_x)
    top = math.floor(min(top_center, bottom_center) - margin_y)
    right = math.ceil(max(left_center, right_center) + margin_x)
    bottom = math.ceil(max(top_center, bottom_center) + margin_y)

    screen_left = full_monitor["left"]
    screen_top = full_monitor["top"]
    screen_right = screen_left + full_monitor["width"]
    screen_bottom = screen_top + full_monitor["height"]

    left = max(left, screen_left)
    top = max(top, screen_top)
    right = min(right, screen_right)
    bottom = min(bottom, screen_bottom)

    return {
        "left": left,
        "top": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
    }


def get_capture_board_origin(x1, y1, monitor):
    """将屏幕绝对坐标转换为当前截图区域内的坐标。"""
    return (x1 - monitor["left"], y1 - monitor["top"])


def _projection_runs(values, threshold):
    runs = []
    start = None

    for index, value in enumerate(values):
        if value >= threshold:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None

    if start is not None:
        runs.append((start, len(values) - 1))

    return runs


def refine_grid_geometry(screenshot, x1, y1, dx, dy):
    """根据格子边框的颜色投影修正 27 个格子的中心位置。"""
    # 颜色通道原为 uint8，直接加阈值会在超过 255 时回绕。
    b, g, r = cv2.split(screenshot.astype(np.int16))
    frame_mask = (r >= 150) & (g >= 90) & (b >= 60) & (r >= g + 25) & (g >= b + 15)

    x_projection = frame_mask.sum(axis=0)
    y_projection = frame_mask.sum(axis=1)

    x_runs = _projection_runs(x_projection, max(8, int(frame_mask.shape[0] * 0.08)))
    y_runs = _projection_runs(y_projection, max(8, int(frame_mask.shape[1] * 0.08)))

    def find_centers(runs, expected_start, expected_step, count):
        pitch = abs(expected_step)
        if pitch <= 1:
            return None

        candidates = [
            run for run in runs if pitch * 0.45 <= run[1] - run[0] + 1 <= pitch * 1.20
        ]

        centers = []
        used = set()

        for index in range(count):
            expected = expected_start + index * expected_step
            available = [
                run
                for run_index, run in enumerate(candidates)
                if run_index not in used
                and abs((run[0] + run[1]) / 2 - expected) <= pitch * 0.55
            ]

            if not available:
                return None

            run = min(
                available, key=lambda item: abs((item[0] + item[1]) / 2 - expected)
            )
            used.add(candidates.index(run))
            centers.append((run[0] + run[1]) / 2)

        return centers

    x_centers = find_centers(x_runs, x1, dx, COLS)
    y_centers = find_centers(y_runs, y1, dy, ROWS)

    if x_centers is None or y_centers is None:
        return None

    def fit_axis(centers, expected_step):
        # 使用全部格子拟合间距，避免首尾边框的一点偏差传递给整行。
        indices = np.arange(len(centers), dtype=float)
        centered = indices - indices.mean()
        step = float(np.dot(centered, centers) / np.dot(centered, centered))
        origin = float(np.mean(centers) - step * indices.mean())
        residual = np.max(np.abs(np.asarray(centers) - (origin + indices * step)))
        if not 0.70 * expected_step <= step <= 1.30 * expected_step:
            return None
        if residual > max(2.0, abs(step) * 0.10):
            return None
        return origin, step

    horizontal = fit_axis(x_centers, dx)
    vertical = fit_axis(y_centers, dy)
    if horizontal is None or vertical is None:
        return None
    return horizontal[0], vertical[0], horizontal[1], vertical[1]


def is_left_mouse_button_down():
    """读取 Windows 当前左键状态。"""
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)


def set_overlay_click_through(overlay):
    """让已完成的校准方框不拦截游戏鼠标点击。"""
    hwnd = overlay.winfo_id()
    user32 = ctypes.windll.user32
    get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)

    style = get_window_long(hwnd, -20)
    style |= 0x00000020  # WS_EX_TRANSPARENT
    style |= 0x08000000  # WS_EX_NOACTIVATE
    set_window_long(hwnd, -20, style)


def close_selection_overlay():
    """关闭当前校准方框并释放窗口资源。"""
    global SELECTION_OVERLAY

    if SELECTION_OVERLAY is not None:
        try:
            for border in SELECTION_OVERLAY["borders"]:
                border.destroy()
            SELECTION_OVERLAY["root"].destroy()
        except Exception:
            pass

    SELECTION_OVERLAY = None


def update_selection_overlay(overlay, left, top, right, bottom):
    """更新选区四条边框的位置，不覆盖中间的游戏画面。"""
    width = max(1, right - left)
    height = max(1, bottom - top)
    thickness = 3

    def geometry_offset(value):
        return f"+{value}" if value >= 0 else str(value)

    geometries = (
        (left, top, width, thickness),
        (left, bottom - thickness, width, thickness),
        (left, top, thickness, height),
        (right - thickness, top, thickness, height),
    )

    for border, (x, y, border_width, border_height) in zip(
        overlay["borders"], geometries
    ):
        border.geometry(
            f"{border_width}x{border_height}{geometry_offset(x)}{geometry_offset(y)}"
        )

    overlay["root"].update_idletasks()
    overlay["root"].update()


def select_board_region():
    """等待用户拖动鼠标框选棋盘，并返回四个边界坐标。"""
    global SELECTION_OVERLAY

    import tkinter as tk

    close_selection_overlay()

    print()
    print("按住鼠标左键，拖动鼠标, 确保所有格子都包括在内后，松开鼠标完成框选。")

    overlay = None

    try:
        root = tk.Tk()
        root.withdraw()

        borders = []
        for _ in range(4):
            border = tk.Toplevel(root)
            border.overrideredirect(True)
            border.attributes("-topmost", True)
            border.configure(bg="#00ff00")
            border.geometry("1x1+0+0")
            border.withdraw()
            borders.append(border)

        overlay = {"root": root, "borders": borders}
        SELECTION_OVERLAY = overlay
        root.update_idletasks()
        root.update()

        for border in borders:
            set_overlay_click_through(border)

    except Exception as exc:
        if overlay is not None:
            close_selection_overlay()
        overlay = None
        print(f"提示：无法显示框选方框，将继续记录鼠标区域（{exc}）。")

    selection_completed = False

    try:
        while not is_left_mouse_button_down():
            if AUTO_STOP_REQUESTED:
                return None
            if RESELECT_REQUESTED.is_set():
                return RESELECT_RESULT
            wait_while_paused()
            if AUTO_STOP_REQUESTED:
                return None
            if RESELECT_REQUESTED.is_set():
                return RESELECT_RESULT
            if overlay is not None:
                overlay["root"].update_idletasks()
                overlay["root"].update()
            time.sleep(0.01)

        start = pyautogui.position()
        start_x = int(start.x)
        start_y = int(start.y)

        rectangle_visible = False
        if overlay is not None:
            for border in overlay["borders"]:
                border.deiconify()
                border.lift()
            rectangle_visible = True

        while is_left_mouse_button_down():
            if AUTO_STOP_REQUESTED:
                return None
            if RESELECT_REQUESTED.is_set():
                return RESELECT_RESULT
            if AUTO_PAUSED:
                return RETRY_RESULT

            current = pyautogui.position()
            current_x = int(current.x)
            current_y = int(current.y)

            if rectangle_visible:
                update_selection_overlay(
                    overlay,
                    min(start_x, current_x),
                    min(start_y, current_y),
                    max(start_x, current_x),
                    max(start_y, current_y),
                )

            time.sleep(0.01)

        end = pyautogui.position()
        selection_completed = True

    finally:
        if not selection_completed and overlay is not None:
            close_selection_overlay()

    left = min(int(start.x), int(end.x))
    top = min(int(start.y), int(end.y))
    right = max(int(start.x), int(end.x))
    bottom = max(int(start.y), int(end.y))

    return left, top, right, bottom


def calibrate(wait_for_enter=True):
    """通过鼠标拖动框选棋盘，计算截图区域与网格间距。"""

    print()
    print("=" * 80)
    print("框选区域")
    print("=" * 80)

    if wait_for_enter:
        print("确保程序切到前台，且所有格子完整显示后，按 Enter 开始框选。")
        if not wait_for_start_key():
            return None

    while True:
        wait_while_paused()
        if AUTO_STOP_REQUESTED:
            return None
        # 当前即将开始新框选，消费此前的重选请求。
        RESELECT_REQUESTED.clear()
        selected_region = select_board_region()

        if selected_region is None:
            return None
        if selected_region is RESELECT_RESULT or selected_region is RETRY_RESULT:
            continue

        grid_left, grid_top, grid_right, grid_bottom = selected_region
        grid_width = grid_right - grid_left
        grid_height = grid_bottom - grid_top

        if grid_width >= MIN_BOARD_WIDTH and grid_height >= MIN_BOARD_HEIGHT:
            break

        print("框选区域过小，请重新框选。(按F9可退出程序)")

    dx = grid_width / COLS
    dy = grid_height / ROWS

    # x1、y1 表示第 1 行第 1 列球的中心，点击坐标仍然准确。
    x1 = grid_left + dx / 2
    y1 = grid_top + dy / 2

    return x1, y1, dx, dy


# 单格识别与棋盘状态
def crop_cell(screenshot, cx, cy, cell_w, cell_h):
    """按中心点和尺寸截取格子，越界区域会自动裁剪。"""
    h, w = screenshot.shape[:2]
    x1 = max(0, int(round(cx - cell_w / 2)))
    y1 = max(0, int(round(cy - cell_h / 2)))
    x2 = min(w, int(round(cx + cell_w / 2)))
    y2 = min(h, int(round(cy + cell_h / 2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return screenshot[y1:y2, x1:x2]


def get_center_region(cell):
    """截取格子中央区域，减少边框和背景对识别的干扰。"""
    h, w = cell.shape[:2]
    cw = int(w * CENTER_RATIO)
    ch = int(h * CENTER_RATIO)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    return cell[y1 : y1 + ch, x1 : x1 + cw]


def estimate_background(img):
    """用格子四角的像素中位数估计当前格子的背景颜色。"""
    h, w = img.shape[:2]
    patch = max(3, int(min(h, w) * 0.12))
    patches = (
        img[0:patch, 0:patch],
        img[0:patch, w - patch : w],
        img[h - patch : h, 0:patch],
        img[h - patch : h, w - patch : w],
    )
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(pixels, axis=0).astype(np.float32)


def get_background_mask(img, background):
    """返回与估计背景颜色差异足够大的像素掩码。"""

    diff = img.astype(np.float32) - background.reshape(1, 1, 3)

    distance = np.sqrt(np.sum(diff * diff, axis=2))

    return distance > BACKGROUND_DISTANCE


def make_color_masks(img):
    """按颜色特征生成火、雷、风三种候选区域掩码。"""

    b = img[:, :, 0].astype(np.int16)

    g = img[:, :, 1].astype(np.int16)

    r = img[:, :, 2].astype(np.int16)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.int16)

    s = hsv[:, :, 1].astype(np.int16)

    v = hsv[:, :, 2].astype(np.int16)

    background = estimate_background(img)

    not_background = get_background_mask(img, background)

    # 火是红橙色，不能只用 R > G，因为棋盘背景也偏棕橙色。

    fire_hue = (h <= 15) | (h >= 170)

    fire_mask = (
        fire_hue
        & (s >= 135)
        & (v >= 100)
        & (r >= g + 35)
        & (r >= b + 60)
        & not_background
    )

    # 雷是紫色。

    thunder_hue = (h >= 125) & (h <= 165)

    thunder_mask = (
        thunder_hue
        & (s >= 80)
        & (v >= 80)
        & (r >= g + 20)
        & (b >= g + 20)
        & not_background
    )

    # 风是灰蓝色。

    wind_mask = (s <= 105) & (v >= 75) & (b >= r + 3) & (b >= g - 8) & not_background

    # 开运算去除小噪声。

    kernel = MASK_OPEN_KERNEL

    fire_mask = cv2.morphologyEx(fire_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    thunder_mask = cv2.morphologyEx(
        thunder_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )

    wind_mask = cv2.morphologyEx(wind_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 闭运算连接同一能量球中被高光分开的区域。

    kernel2 = MASK_CLOSE_KERNEL

    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel2)

    thunder_mask = cv2.morphologyEx(thunder_mask, cv2.MORPH_CLOSE, kernel2)

    wind_mask = cv2.morphologyEx(wind_mask, cv2.MORPH_CLOSE, kernel2)

    return (fire_mask * 255, thunder_mask * 255, wind_mask * 255, background)


def get_best_component(mask, *, include_image=True):
    """从掩码中选择最可能代表能量球的连通区域。"""

    h, w = mask.shape

    center_x = w / 2
    center_y = h / 2

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    best_label = -1
    best_score = -1

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 5:
            continue

        cx = centroids[i][0]
        cy = centroids[i][1]

        distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

        center_factor = 1.0 / (1.0 + distance)

        score = area * (0.5 + center_factor * 2.0)

        if score > best_score:
            best_score = score

            best_label = i

    if best_label < 0:
        return (np.zeros_like(mask) if include_image else None, 0)

    component = None
    if include_image:
        component = np.zeros_like(mask)
        component[labels == best_label] = 255

    # 连通域分析已经统计面积，直接复用，避免再次遍历整张掩码。
    area = int(stats[best_label, cv2.CC_STAT_AREA])

    return (component, area)


def estimate_level(color_type, ratio):
    """按颜色对应的面积阈值将能量球判为 1、2 或 3 级。"""

    if ratio <= 0:
        return 0

    thresholds = LEVEL_THRESHOLDS[color_type]

    if ratio < thresholds["12"]:
        return 1

    if ratio < thresholds["23"]:
        return 2

    return 3


def recognize_cell(cell, row, col, debug_dir, *, include_diagnostics=True):
    """识别单个格子的颜色、等级和置信度。"""

    unknown_result = {"type": "?", "level": 0, "area_ratio": 0, "confidence": 0}

    if cell is None or cell.size == 0:
        return unknown_result, {}

    center = get_center_region(cell)

    if center.size == 0:
        return unknown_result, {}

    h, w = center.shape[:2]

    total_pixels = h * w

    (fire_mask, thunder_mask, wind_mask, background) = make_color_masks(center)

    masks = {"f": fire_mask, "r": thunder_mask, "w": wind_mask}

    include_diagnostics = include_diagnostics or (
        DEBUG_SAVE_IMAGES and debug_dir is not None
    )
    data = {}
    ratios = {}

    for color_type, mask in masks.items():
        component, pixel_area = get_best_component(mask, include_image=include_diagnostics)

        ratio = pixel_area / total_pixels
        ratios[color_type] = ratio
        if include_diagnostics:
            data[color_type] = {
                "mask": mask,
                "component": component,
                "pixel_area": pixel_area,
                "ratio": ratio,
            }

    valid = {}

    for color_type in ("f", "r", "w"):
        ratio = ratios[color_type]

        if ratio >= MIN_COLOR_RATIO:
            valid[color_type] = ratio

    if not valid:
        result = {"type": "?", "level": 0, "area_ratio": 0, "confidence": 0}

        return (result, data)

    best_type = max(valid, key=valid.get)

    best_ratio = valid[best_type]

    sorted_values = sorted(valid.values(), reverse=True)

    if len(sorted_values) >= 2:
        second_ratio = sorted_values[1]

    else:
        second_ratio = 0

    if best_ratio > 0:
        color_gap = (best_ratio - second_ratio) / best_ratio

    else:
        color_gap = 0

    level = estimate_level(best_type, best_ratio)

    # 面积特别小时判定为1级
    if best_ratio < 0.08:
        level = 1

    thresholds = LEVEL_THRESHOLDS[best_type]

    if level == 1:
        target = thresholds["12"] * 0.65

    elif level == 2:
        target = (thresholds["12"] + thresholds["23"]) / 2

    else:
        target = thresholds["23"] + thresholds["23"] * 0.45

    distance = abs(best_ratio - target)

    level_confidence = max(0, 1.0 - distance / 0.20)

    confidence = color_gap * 0.50 + level_confidence * 0.50

    result = {
        "type": best_type,
        "level": level,
        "area_ratio": best_ratio,
        "confidence": confidence,
    }

    # 主线程仅识别；轮廓和标签由后台 DEBUG 保存时生成。
    if not DEBUG_SAVE_IMAGES or debug_dir is None:
        return result, data

    # 前面的提前返回已确认启用 DEBUG 且保存目录有效。
    base = os.path.join(debug_dir, f"r{row + 1}c{col + 1}")
    cv2.imwrite(base + "_center.png", center)
    cv2.imwrite(base + "_fire.png", fire_mask)
    cv2.imwrite(base + "_thunder.png", thunder_mask)
    cv2.imwrite(base + "_wind.png", wind_mask)
    cv2.imwrite(base + "_component.png", data[best_type]["component"])

    # 生成带轮廓、颜色、等级和面积比例的识别图。
    debug_img = center.copy()

    contours, _ = cv2.findContours(
        data[best_type]["component"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(debug_img, contours, -1, DRAW_COLOR[best_type], 2)

    label = f"{level}{best_type} {best_ratio:.3f}"

    cv2.putText(
        debug_img,
        label,
        (3, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(base + "_result.png", debug_img)

    return (result, data)


def get_cell_center_from_board(r, c, x1, y1, dx, dy):
    return (int(round(x1 + c * dx)), int(round(y1 + r * dy)))


def recognize_board_snapshot(screenshot, x1, y1, dx, dy):
    """
    识别整张 3×9 棋盘，并返回棋盘状态。

    棋盘状态只包含“颜色 + 等级”，不包含面积比例等易波动数据。
    因此稳定判断不会因为动画中的轻微亮度变化而误判为新棋盘。
    """
    cell_w = abs(dx) * 0.90
    cell_h = abs(dy) * 0.90
    board = []
    unknown_count = 0

    for row in range(ROWS):
        board_row = []

        for col in range(COLS):
            cx = x1 + col * dx
            cy = y1 + row * dy
            cell = crop_cell(screenshot, cx, cy, cell_w, cell_h)

            if cell is None:
                result = {"type": "?", "level": 0, "area_ratio": 0, "confidence": 0}
            else:
                result, _ = recognize_cell(
                    cell, row, col, None, include_diagnostics=False
                )

            if result["type"] == "?":
                unknown_count += 1

            board_row.append(result)

        board.append(board_row)

    if unknown_count > 0:
        return board, unknown_count, None

    board_state = make_board_state(board)
    return board, 0, board_state


# 热键、等待与交换控制
def request_auto_stop(message="收到停止指令"):
    """执行统一的停止流程，供 F9 和自动结束条件共同调用。"""
    global AUTO_STOP_REQUESTED

    if AUTO_STOP_REQUESTED:
        return

    AUTO_STOP_REQUESTED = True
    print()
    print("=" * 90)
    print(message)
    print("=" * 90)


def request_reselect():
    """记录 F7 重新框选请求，等待当前操作安全结束后执行。"""
    global AUTO_PAUSED

    if AUTO_STOP_REQUESTED:
        return

    AUTO_PAUSED = False
    RESELECT_REQUESTED.set()
    print()
    print("已收到 F7，当前操作完成后重新框选。")


def setup_hotkeys():
    """注册全局热键；重复调用时不重复注册。"""
    global HOTKEYS_REGISTERED

    if HOTKEYS_REGISTERED:
        return

    def toggle_pause():
        global AUTO_PAUSED
        AUTO_PAUSED = not AUTO_PAUSED

        if AUTO_PAUSED:
            print()
            print("=" * 90)
            print("已暂停")
            print("按 F8 继续，F9 退出")
            print("=" * 90)
        else:
            print()
            print("=" * 90)
            print("已继续")
            print("=" * 90)

    def stop_program():
        request_auto_stop()

    keyboard.add_hotkey("f8", toggle_pause)
    keyboard.add_hotkey("f9", stop_program)
    keyboard.add_hotkey("f7", request_reselect)
    HOTKEYS_REGISTERED = True


def wait_while_paused():
    """
    暂停状态下保持程序运行。
    F8 恢复，F9 退出。
    """
    while AUTO_PAUSED and not AUTO_STOP_REQUESTED and not RESELECT_REQUESTED.is_set():
        time.sleep(0.10)


def wait_for_start_key():
    """等待控制台 Enter，同时响应暂停、重选和退出。"""
    import msvcrt

    while not AUTO_STOP_REQUESTED:
        if RESELECT_REQUESTED.is_set():
            return True
        if AUTO_PAUSED:
            wait_while_paused()
            continue
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == "\x03":
                raise KeyboardInterrupt
            if key in ("\r", "\n"):
                return True
        time.sleep(0.05)
    return False


def pause_for_no_move():
    """
    当前没有立即可合成交换时：
    不退出程序，而是自动暂停。
    用户手动处理后按 F8 恢复。
    """
    global AUTO_PAUSED

    AUTO_PAUSED = True

    print()
    print("=" * 90)
    print("当前没有发现能够立即形成三连的交换。")
    print("=" * 90)
    print()
    print("自动操作已暂停。")
    print("你可以手动完成一次交换，然后按 F8。")
    print("F7 = 重新框选")
    print("F8 = 继续")
    print("F9 = 退出")
    print("=" * 90)


def wait_for_initial_board_ready(sct, monitor, x1, y1, dx, dy):
    """等待 27 个球全部识别成功且棋盘状态稳定后返回截图。"""

    print("等待能量球出现...(按F7可重新选择区域)", end="", flush=True)

    capture_x1, capture_y1 = get_capture_board_origin(x1, y1, monitor)

    previous_state = None
    stable_start = None
    refined_geometry = None

    while True:
        if AUTO_STOP_REQUESTED:
            return None

        if RESELECT_REQUESTED.is_set():
            return RESELECT_RESULT

        was_paused = AUTO_PAUSED
        wait_while_paused()
        if was_paused:
            previous_state = None
            stable_start = None

        if AUTO_STOP_REQUESTED:
            return None
        if RESELECT_REQUESTED.is_set():
            return RESELECT_RESULT

        raw = np.array(sct.grab(monitor))

        screenshot = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        # 边框定位不能依赖球已全部识别，否则偏移本身会阻止坐标校正。
        if refined_geometry is None:
            refined_geometry = refine_grid_geometry(
                screenshot, capture_x1, capture_y1, dx, dy
            )

            if refined_geometry is not None:
                capture_x1, capture_y1, dx, dy = refined_geometry

        _, unknown_count, current_state = recognize_board_snapshot(
            screenshot, capture_x1, capture_y1, dx, dy
        )

        if unknown_count > 0:
            stable_start = None
            previous_state = None
            print(
                f"\r等待能量球出现... 已识别 "
                f"{ROWS * COLS - unknown_count}/{ROWS * COLS} 个",
                end="",
                flush=True,
            )
        elif current_state != previous_state:
            previous_state = current_state
            stable_start = time.monotonic()
        elif time.monotonic() - stable_start >= INITIAL_BOARD_STABLE_TIME:
            print("\r已识别到完整棋盘（27个能量球）。        ")
            return (raw, capture_x1, capture_y1, dx, dy)

        time.sleep(BOARD_CHECK_INTERVAL)


def wait_for_calibrated_board(sct, full_monitor, calibration):
    """首次框选和 F7 共用校正流程，统一重建截图区域及屏幕坐标。"""
    while calibration is not None:
        x1, y1, dx, dy = calibration
        monitor = get_board_capture_monitor(x1, y1, dx, dy, full_monitor)
        ready = wait_for_initial_board_ready(sct, monitor, x1, y1, dx, dy)
        if ready is RESELECT_RESULT:
            RESELECT_REQUESTED.clear()
            calibration = calibrate(wait_for_enter=False)
            continue
        if ready is None:
            return None
        _, capture_x1, capture_y1, dx, dy = ready
        x1 = capture_x1 + monitor["left"]
        y1 = capture_y1 + monitor["top"]
        monitor = get_board_capture_monitor(x1, y1, dx, dy, full_monitor)
        return x1, y1, dx, dy, monitor
    return None


def wait_for_board_stable(sct, monitor, x1, y1, dx, dy, minimum_wait=None):
    """
    交换完成后等待棋盘稳定。

    正常情况下等到 27 格连续稳定；热键或超时可提前结束等待。
    """

    if minimum_wait is None:
        minimum_wait = MIN_WAIT_AFTER_SWAP

    start_time = time.monotonic()
    capture_x1, capture_y1 = get_capture_board_origin(x1, y1, monitor)

    previous_state = None
    stable_start = None

    while True:
        if AUTO_STOP_REQUESTED:
            print("\r收到停止指令，停止等待棋盘稳定。")
            return time.monotonic() - start_time

        if RESELECT_REQUESTED.is_set():
            return RESELECT_RESULT
        if AUTO_PAUSED:
            pause_start = time.monotonic()
            wait_while_paused()
            start_time += time.monotonic() - pause_start
            previous_state = None
            stable_start = None
            continue

        elapsed = time.monotonic() - start_time
        # 最低等待也分段检查热键；暂停时间不占用稳定检测的超时预算。
        if elapsed < minimum_wait:
            time.sleep(min(0.05, minimum_wait - elapsed))
            continue

        if elapsed >= minimum_wait + MAX_WAIT_AFTER_SWAP:
            print(f"  棋盘等待达到上限 {MAX_WAIT_AFTER_SWAP:.2f}s，强制进入下一轮。")

            return elapsed

        time.sleep(BOARD_CHECK_INTERVAL)

        raw = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        _, unknown_count, current_state = recognize_board_snapshot(
            screenshot, capture_x1, capture_y1, dx, dy
        )

        if unknown_count > 0:
            previous_state = None
            stable_start = None
            print(
                f"\r等待棋盘稳定... 已识别 "
                f"{ROWS * COLS - unknown_count}/{ROWS * COLS} 个",
                end="",
                flush=True,
            )
            continue

        if current_state != previous_state:
            previous_state = current_state
            stable_start = time.monotonic()
            print("\r棋盘状态发生变化，重新等待稳定...     ", end="", flush=True)
            continue

        stable_elapsed = time.monotonic() - stable_start
        print(f"\r检测棋盘状态稳定：{stable_elapsed:.2f}s", end="", flush=True)

        if stable_elapsed < BOARD_STABLE_TIME:
            continue

        print(f"\r棋盘已稳定，等待 {elapsed:.2f}s。              ")
        return elapsed


def click_best_swap(results, x1, y1, dx, dy, sct, monitor, step=1):
    """
    执行当前一步最优交换，然后等待棋盘稳定。

    点击前的暂停或重选返回对应标记；两次点击完成后记为 True。
    点击后的等待被打断，也不能漏记已经执行的交换。
    """

    if AUTO_STOP_REQUESTED:
        return False
    if RESELECT_REQUESTED.is_set():
        return RESELECT_RESULT
    if AUTO_PAUSED:
        return RETRY_RESULT

    if not results:
        print()
        print("=" * 90)
        print("自动操作")
        print("=" * 90)
        print("没有可以立即合成的交换，本轮停止。")

        return False

    if not AUTO_CLICK_ENABLED:
        print("自动点击已关闭。")

        return False

    best = results[0]

    p1 = best["p1"]
    p2 = best["p2"]

    x1_click, y1_click = get_cell_center_from_board(*p1, x1, y1, dx, dy)

    x2_click, y2_click = get_cell_center_from_board(*p2, x1, y1, dx, dy)

    # 执行交换。鼠标移到屏幕角落时保留 PyAutoGUI 的紧急停止保护。
    pair_completed = False
    try:
        pyautogui.click(x1_click, y1_click)

        time.sleep(AUTO_CLICK_DELAY)

        if AUTO_STOP_REQUESTED:
            return False

        pyautogui.click(x2_click, y2_click)
        pair_completed = True

        # 点击完成后立即移出棋盘，避免鼠标指针遮挡小球
        pyautogui.moveTo(MOUSE_MOVE_OUT_X, MOUSE_MOVE_OUT_Y, duration=0.05)
    except pyautogui.FailSafeException:
        request_auto_stop("检测到鼠标位于屏幕角落，已安全停止自动点击")
        return pair_completed

    print()
    print("交换已执行，等待棋盘稳定...")

    wait_for_board_stable(
        sct,
        monitor,
        x1,
        y1,
        dx,
        dy,
        minimum_wait=(
            FIRST_STEP_MIN_WAIT_AFTER_SWAP if step == 1 else MIN_WAIT_AFTER_SWAP
        ),
    )

    return True


# 棋盘数据与确定性合成
def make_board_state(board):
    state = []
    for row in range(ROWS):
        state_row = []
        for col in range(COLS):
            result = board[row][col]
            t = result["type"]
            level = result["level"]
            if t in ("f", "r", "w") and level in (1, 2, 3):
                state_row.append((t, level))
            else:
                state_row.append(None)
        state.append(state_row)
    return state


def same_ball(a, b):
    return a is not None and b is not None and a[0] == b[0] and a[1] == b[1]


def empty_cells(board):
    return [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] is None]


def copy_board(board):
    return [row.copy() for row in board]


def swap_cells(board, p1, p2):
    new_board = copy_board(board)
    r1, c1 = p1
    r2, c2 = p2
    new_board[r1][c1], new_board[r2][c2] = (new_board[r2][c2], new_board[r1][c1])
    return new_board


def find_horizontal_matches(board):
    matches = []

    for r in range(ROWS):
        c = 0
        while c < COLS:
            ball = board[r][c]

            if ball is None or ball[1] >= 3:
                c += 1
                continue

            start = c
            c += 1

            while c < COLS and ball == board[r][c]:
                c += 1

            if c - start >= 3:
                matches.append([(r, x) for x in range(start, c)])

    return matches


def find_vertical_matches(board):
    """棋盘固定三行，竖向三连直接比较三格，无需逐格寻找连续段。"""
    top, middle, bottom = board
    return [
        [(0, c), (1, c), (2, c)]
        for c, ball in enumerate(top)
        if ball is not None and ball[1] < 3 and ball == middle[c] == bottom[c]
    ]


def find_matches(board):
    return find_horizontal_matches(board) + find_vertical_matches(board)


# 合成模型：同色同级三球合一，三级可交换但不再合成。
# 交叉组合按最下方、最左侧优先选取，同轮组合不得共用格子。
# 长横连每三格一组，剩余球保留；空位原地补一级球，没有下落。


def ball_energy_value(ball):
    """返回一个能量球当前占用的能量价值。"""
    if ball is None:
        return 0.0

    ball_type, level = ball
    return float(BALL_ENERGY.get(ball_type, {}).get(level, 0.0))


def refill_energy_gain(empty_positions, board):
    """计算指定空位补入新球后增加的能量价值。"""
    return sum(
        ball_energy_value(board[r][c])
        for r, c in empty_positions
        if board[r][c] is not None
    )


def match_anchor(group):
    """游戏全局排序：最下方优先，其次最左侧。"""
    return min(group, key=lambda p: (-p[0], p[1]))


def _group_priority_key(group):
    anchor = match_anchor(group)
    return (-anchor[0], anchor[1])


def choose_merge_groups(matches):
    """
    按全局优先级选出本轮所有“不冲突”组合。
    候选组按：
        最下方 -> 最左侧
    排序。
    同轮中已经被选中的位置不能再次使用。
    """
    ordered = sorted(matches, key=_group_priority_key)

    selected = []
    used = set()

    for group in ordered:
        if any(p in used for p in group):
            continue

        selected.append(group)
        used.update(group)

    return selected


def split_long_group(group):
    """
    长组合按连续位置拆成每3个一组。
    排序采用最下方 -> 最左侧的全局位置顺序。

    对横向 5/6/7 连，位置顺序自然对应：
        5 -> [3] + [2]
        6 -> [3] + [3]
        7 -> [3] + [3] + [1]

    3×9 棋盘竖向只有3格，因此不会出现竖向4连。
    """
    ordered = sorted(group, key=lambda p: (-p[0], p[1]))

    return [
        ordered[i : i + 3]
        for i in range(0, len(ordered), 3)
        if len(ordered[i : i + 3]) >= 3
    ]


def expand_matches(matches):
    expanded = []

    for group in matches:
        if len(group) <= 3:
            expanded.append(group)
        else:
            expanded.extend(split_long_group(group))

    return expanded


def _apply_merge_groups_inplace(state, groups):
    """
    原地执行本轮互不冲突的3球组合，只接收当前模拟私有的棋盘。
    每组的生成位置为该组的优先位置。
    合成收益按净能量计算：新球价值减去三个输入球价值。
    """
    gain = 0.0
    merge_count = 0

    for group in groups:
        if len(group) != 3:
            continue

        anchor = match_anchor(group)
        ball = state[anchor[0]][anchor[1]]

        if ball is None or ball[1] >= 3:
            continue

        # 同轮组合已经保证不冲突；但这里再次检查，防御异常。
        if any(state[r][c] is None for r, c in group):
            continue

        ball_type, old_level = ball
        new_level = old_level + 1
        input_energy = sum(ball_energy_value(state[r][c]) for r, c in group)
        new_energy = ball_energy_value((ball_type, new_level))

        for r, c in group:
            state[r][c] = None

        state[anchor[0]][anchor[1]] = (ball_type, new_level)

        gain += new_energy - input_energy
        merge_count += 1

    return merge_count, gain


def apply_merge_groups(board, groups):
    """复制后执行一轮合成，保留调用者传入的棋盘。"""
    state = copy_board(board)
    merge_count, gain = _apply_merge_groups_inplace(state, groups)
    return state, merge_count, gain


def resolve_merges(board, max_rounds=20):
    """连续处理确定性合成，保留空位；补球由 resolve_with_refill 处理。"""
    state = copy_board(board)
    total_gain = 0.0
    total_merges = 0

    for _ in range(max_rounds):
        matches = find_matches(state)

        if not matches:
            break

        expanded = expand_matches(matches)
        selected = choose_merge_groups(expanded)

        if not selected:
            break

        merge_count, gain = _apply_merge_groups_inplace(state, selected)

        if merge_count <= 0:
            break

        total_merges += merge_count
        total_gain += gain

    return state, total_merges, total_gain


def refill_random(board, rng, empty_positions=None):
    state = copy_board(board)

    # 分层 RNG 需要在每轮补球前重置位置编号；普通 RNG 不需要。
    begin_refill = getattr(rng, "begin_refill", None)
    if begin_refill is not None:
        begin_refill()

    if empty_positions is None:
        empty_positions = empty_cells(state)
    for r, c in empty_positions:
        state[r][c] = (rng.choice(TYPE_ORDER), 1)

    return state


def resolve_with_refill(board, rng, max_rounds=20):
    """
    完整真实模拟，严格区分“补球前确定性合成”和“补球后连锁”：

        当前已有组合
        -> 连续执行所有无需补球即可形成的确定性合成
        -> 只有确定性阶段稳定后才补一级球
        -> 补球
        -> 再连续执行所有确定性合成
        -> 若产生新的空位，再补球
        -> ...

    因此，同一次补球之前由升级球立即形成的后续合成，不会被错误
    计入“补球后连锁”之外的阶段。

    返回的 total_gain 使用净能量口径：合成扣除输入球价值，补球加入
    新生成的1级球价值。
    """
    state = copy_board(board)
    total_gain = 0.0
    total_merges = 0
    rounds_used = 0

    while rounds_used < max_rounds:
        # 先把当前状态中所有无需补球即可继续发生的合成全部做完。
        stable_state, merge_count, gain = resolve_merges(
            state, max_rounds=max_rounds - rounds_used
        )

        state = stable_state
        total_gain += float(gain)
        total_merges += int(merge_count)

        if merge_count > 0:
            rounds_used += 1

        # 没有空位，且当前已经稳定 -> 连锁结束。
        empty_positions = empty_cells(state)
        if not empty_positions:
            if not find_matches(state):
                break
            continue

        # 只有确定性合成阶段结束后才进行随机补球，并计入新生成球的价值。
        state = refill_random(state, rng, empty_positions)
        total_gain += refill_energy_gain(empty_positions, state)

        # 补球后下一轮继续先处理全部确定性合成。
        rounds_used += 1

    return state, total_merges, total_gain


def evaluate_swap(board, p1, p2):
    """计算交换后、第一次补球前的确定性合成净收益。"""
    a = board[p1[0]][p1[1]]
    b = board[p2[0]][p2[1]]
    if a is None or b is None or same_ball(a, b):
        return None

    # resolve_merges 自己会检查三连，不再提前扫描一次。
    resolved, merge_count, merge_gain = resolve_merges(swap_cells(board, p1, p2))
    return make_candidate(
        p1=p1, p2=p2, board=resolved, merge_gain=merge_gain, merge_count=merge_count
    )


def _board_key(board):
    """将棋盘转换为不可变键，用于同一轮的重复棋盘去重。"""
    return tuple(tuple(row) for row in board)


def make_candidate(*, p1, p2, board, merge_gain=0.0, merge_count=0):
    """建立快速筛选结果；随机评估字段在最终评估完成后统一补充。"""
    return {
        "p1": p1,
        "p2": p2,
        "board": board,
        "immediate_gain": float(merge_gain),
        "merge_count": int(merge_count),
        "filter_gain": float(merge_gain + EXPECTED_REFILL_GAIN_PER_MERGE * merge_count),
    }


def _future_legal_swaps(board):
    """
    枚举所有合法下一步交换。
    三级球可以交换；不要求相邻；相同球交换无意义。
    """
    for p1, p2 in ALL_SWAPS:
        a = board[p1[0]][p1[1]]
        b = board[p2[0]][p2[1]]

        if a is None or b is None:
            continue

        if same_ball(a, b):
            continue

        yield p1, p2


def _swap_setups(board):
    """枚举三格窗口中的换入位置、目标球及窗口外供球位置。"""
    cells = tuple(ball for row in board for ball in row)
    sources = {}
    for index, ball in enumerate(cells):
        if ball is not None and ball[1] < 3:
            sources.setdefault(ball, []).append(index)

    for i, j, k in TRIPLE_WINDOWS:
        a, b, c = cells[i], cells[j], cells[k]
        if a is None or b is None or c is None:
            continue
        if a == b and a != c:
            target, ball = k, a
        elif a == c and a != b:
            target, ball = j, a
        elif b == c and b != a:
            target, ball = i, b
        else:
            continue
        donors = sources.get(ball, ())
        if len(donors) >= 3:
            yield target, ball, tuple(p for p in donors if p not in (i, j, k))


def _future_merge_swaps(board):
    """稳定局面直接列出可合成交换，按原坐标顺序去重。"""
    if find_matches(board):
        # 模拟达到轮数上限时可能仍有三连，完整枚举保留原有处理方式。
        yield from _future_legal_swaps(board)
        return

    # 稳定局面的新三连必定只换入一个球；交换窗口内两个球不能
    # 把不同颜色/等级变成相同。因此检查全部三格窗口不会漏掉长连或交叉。
    swaps = set()
    for target, _, donors in _swap_setups(board):
        for source in donors:
            swaps.add((min(target, source), max(target, source)))
    for first, second in sorted(swaps):
        yield divmod(first, COLS), divmod(second, COLS)


# 平衡采样与候选评估
@dataclass(frozen=True)
class SamplingPlan:
    """同一步所有候选共用的不可变采样计划。"""

    pass_index: int
    seeds: tuple
    stream_id: int = 0

    @property
    def count(self):
        return len(self.seeds)


def make_sampling_plan(pass_index, count, *, stream_id=0):
    """创建均衡样本；stream_id 区分当前补球与下一步补球的序列。"""
    pass_index = int(pass_index)
    count = int(count)
    stream_id = int(stream_id)

    if count <= 0 or count % 3 != 0:
        raise ValueError("样本数量必须是大于 0 的 3 的倍数")

    seeds = tuple(
        (LOOKAHEAD_SEED_BASE + pass_index * 1000003 + stream_id * 0x51ED270B + i * 9176)
        & ((1 << 63) - 1)
        for i in range(count)
    )

    return SamplingPlan(pass_index=pass_index, seeds=seeds, stream_id=stream_id)


def make_evaluation_plan(current_step=1):
    """为当前步创建统一的前瞻局面采样计划。"""
    return make_sampling_plan(int(current_step), LOOKAHEAD_SAMPLE_COUNT)


@lru_cache(maxsize=2048)
def _balanced_refill_colors(plan_index, sample_count, refill_event_index, draw_index):
    """同一批样本共用不可变颜色表；仅复用取色表，不缓存棋盘评分。"""
    mixed = (
        plan_index * 0xC2B2AE3D
        + sample_count * 0x27D4EB2F
        + refill_event_index * 0x9E3779B1
        + draw_index * 0x85EBCA77
    ) & 0xFFFFFFFF
    colors = list(TYPE_ORDER) * (sample_count // len(TYPE_ORDER))
    random.Random(mixed).shuffle(colors)
    return tuple(colors)


class _BalancedRefillRNG:
    """按样本、补球事件和空位编号取色，跨样本保持各颜色各占 1/3。

    这是有限样本的均衡设计，不保证所有空位之间完全独立。
    非补球序列交给普通 RNG。实际颜色分布由离线测试验证。
    """

    __slots__ = (
        "plan_index",
        "sample_index",
        "draw_index",
        "refill_event_index",
        "fallback",
        "sample_count",
    )

    def __init__(self, seed, sample_index, sample_count=9, plan_index=0, stream_id=0):
        # 不同预测阶段使用不同颜色映射，避免每一步从同一补球序列重头开始。
        self.plan_index = int(plan_index) + int(stream_id) * 0x51ED270B
        self.sample_index = int(sample_index)
        self.draw_index = 0
        self.refill_event_index = 0
        self.fallback = random.Random(seed)
        self.sample_count = int(sample_count)

    def begin_refill(self):
        """进入新一轮补球，重置空位编号并更新颜色映射所用的事件号。"""
        self.refill_event_index += 1
        self.draw_index = 0

    def choice(self, seq):
        """为补球位置提供均衡颜色，其他序列交给后备 RNG。"""
        # 内部补球始终传入 TYPE_ORDER，无需每次创建元组和集合。
        if seq is TYPE_ORDER or (len(seq) == 3 and set(seq) == set(TYPE_ORDER)):
            colors = TYPE_ORDER
            count = self.sample_count

            if count >= 27:
                # 每个空位单独打乱等量的火、雷、风，保持边际平衡，
                # 同时避免固定的样本分组让不同空位产生过强关联。
                balanced_colors = _balanced_refill_colors(
                    self.plan_index, count, self.refill_event_index, self.draw_index
                )
                self.draw_index += 1
                return balanced_colors[self.sample_index]
            elif count == 9:
                digit = BALANCED_SAMPLE_COLUMNS[self.draw_index % 4][self.sample_index]
            elif count == 6:
                block = self.sample_index // 3
                pos = self.sample_index % 3
                digit = (pos + block * self.draw_index) % 3
            else:
                digit = (self.sample_index + self.draw_index) % 3

            mixed = (
                self.plan_index * 0xC2B2AE3D
                + self.sample_count * 0x27D4EB2F
                + self.refill_event_index * 0x9E3779B1
                + self.draw_index * 0x85EBCA77
            ) & 0xFFFFFFFF
            mixed ^= mixed >> 16
            permutation = COLOR_PERMUTATIONS[mixed % len(COLOR_PERMUTATIONS)]

            self.draw_index += 1
            return colors[permutation[digit]]

        return self.fallback.choice(seq)


def _simulate_refill_chain_once(board_after_current_merge, sample_index, sample_plan):
    """返回最终棋盘、首次补球价值、后续连锁净收益；不修改传入棋盘。"""
    rng = _BalancedRefillRNG(
        sample_plan.seeds[sample_index],
        sample_index,
        sample_plan.count,
        sample_plan.pass_index,
        sample_plan.stream_id,
    )
    state = board_after_current_merge
    first_refill_gain = 0.0
    empty_positions = empty_cells(state)
    if empty_positions:
        state = refill_random(state, rng, empty_positions)
        first_refill_gain = refill_energy_gain(empty_positions, state)

    state, _, chain_gain = resolve_with_refill(state, rng)
    return state, float(first_refill_gain), float(chain_gain)


@dataclass(frozen=True)
class ChainExpansion:
    """已计算的加权收益，以及尚未完成的条件局面和累计概率。"""

    gain: float
    branches: tuple
    remaining_probability: float
    nodes: int


def _build_chain_window_tables():
    """预计算三格窗口的分支标记，以及首次补球的同分排序参考值。"""
    # 索引 0～9 对应球编码；未知球 -1 使用元组最后一项。
    codes = (*range(len(BRANCH_BALLS)), -1)

    def classify(a, b, c):
        if a != -1 and b != -1 and c != -1:
            return 8 if a == b == c and 0 < a <= 6 else 0
        if a > 3 or b > 3 or c > 3:
            return 0
        if ((a > 0 and b > 0 and a != b)
                or (a > 0 and c > 0 and a != c)
                or (b > 0 and c > 0 and b != c)):
            return 0
        # 1、2、4 标记仍需展开的未知位置，8 表示确定的三连。
        return (1 if a == -1 else 0) | (2 if b == -1 else 0) | (4 if c == -1 else 0)

    flags = tuple(tuple(tuple(classify(a, b, c) for c in codes) for b in codes) for a in codes)
    merge_gains = (0.0,) + tuple(
        BALL_ENERGY[color][2] - 3 * BALL_ENERGY[color][1]
        + EXPECTED_REFILL_GAIN_PER_MERGE
        for color in TYPE_ORDER
    )

    def refill_hint(a, b, c):
        flag = flags[a][b][c]
        if flag in (0, 8):
            return 0.0
        # 0 是真实空位，不属于尚待随机补入的一级球。
        if 0 in (a, b, c):
            return 0.0
        holes = flag.bit_count()
        if holes == 3:
            return sum(merge_gains) / 27.0
        return merge_gains[max(a, b, c)] / (3 ** holes)

    hints = tuple(tuple(tuple(refill_hint(a, b, c) for c in codes) for b in codes) for a in codes)
    return flags, hints


CHAIN_WINDOW_FLAGS, REFILL_WINDOW_HINTS = _build_chain_window_tables()


def _expand_chain_branches(board, node_limit):
    """按等概率补一级球展开连锁，保留超预算时的全部未完成分支。

    首次补球价值另计。未知球只有仍可能影响三连时才展开颜色；其余球
    保留到后续阶段，不能直接删掉，否则可能漏掉之后才出现的连锁。
    """
    memo = {}
    nodes = 0

    def visit(state):
        nonlocal nodes
        if AUTO_STOP_REQUESTED:
            raise StopIteration
        if state in memo:
            return memo[state]
        if nodes >= node_limit:
            return 0.0, {state: 1.0}
        nodes += 1

        undecided = {}
        known_match = False
        for i, j, k in TRIPLE_WINDOWS:
            flags = CHAIN_WINDOW_FLAGS[state[i]][state[j]][state[k]]
            if not flags:
                continue
            if flags == 8:
                known_match = True
                continue
            if flags & 1:
                undecided[i] = undecided.get(i, 0) + 1
            if flags & 2:
                undecided[j] = undecided.get(j, 0) + 1
            if flags & 4:
                undecided[k] = undecided.get(k, 0) + 1

        if undecided:
            index = max(undecided, key=undecided.get)
            total = 0.0
            remainder = {}
            for color in (1, 2, 3):
                assigned = list(state)
                assigned[index] = color
                gain, unresolved = visit(tuple(assigned))
                total += gain
                for branch, probability in unresolved.items():
                    remainder[branch] = remainder.get(branch, 0.0) + probability / 3
            value = total / 3, remainder
        else:
            # 已排除未知球形成三连，且没有确定的三连，无需再建棋盘扫描。
            if not known_match:
                memo[state] = (0.0, {})
                return memo[state]
            # 所有可能受未知球影响的三连已排除；原模拟器处理确定性合成。
            known_board = [
                [BRANCH_BALLS[max(0, state[r * COLS + c])] for c in range(COLS)]
                for r in range(ROWS)
            ]
            resolved, count, gain = resolve_merges(known_board)
            if not count:
                value = 0.0, {}
            else:
                next_state = []
                holes = 0
                for index, ball in enumerate(ball for row in resolved for ball in row):
                    if state[index] == -1:
                        next_state.append(-1)
                    elif ball is None:
                        next_state.append(-1)
                        holes += 1
                    else:
                        next_state.append(BRANCH_BALL_CODES[ball])
                # 每个新空位的平均价值直接计入；颜色仅在影响后续合成时展开。
                tail_gain, remainder = visit(tuple(next_state))
                value = gain + holes * EXPECTED_REFILL_ENERGY + tail_gain, remainder
        memo[state] = value
        return value

    state = tuple(
        -1 if ball is None else BRANCH_BALL_CODES[ball]
        for row in board for ball in row
    )
    gain, unresolved = visit(state)
    cumulative = 0.0
    branches = []
    for branch, probability in unresolved.items():
        cumulative += probability
        branches.append((branch, cumulative))
    return ChainExpansion(gain, tuple(branches), cumulative, nodes)


def _exact_chain_gain(board, node_limit):
    """供离线核对使用；只有全部分支完成时才返回精确期望。"""
    expansion = _expand_chain_branches(board, node_limit)
    return (None if expansion.branches else expansion.gain), expansion.nodes


def _sample_chain_remainder(expansion, sample_index, sample_plan, *, stratified=False):
    """抽取一个未完成分支，固定已知颜色，独立补齐未知球并继续连锁。"""
    if AUTO_STOP_REQUESTED:
        raise StopIteration
    if not expansion.branches:
        return expansion.gain
    seed = sample_plan.seeds[sample_index]
    selector = random.Random(seed ^ 0xD1B54A32D192ED03).random()
    if stratified:
        # 同一个局面的整批估计可按概率区间分层；单次前瞻打分须均匀抽样。
        selector = (sample_index + selector) / sample_plan.count
    target = selector * expansion.remaining_probability
    state = expansion.branches[-1][0]
    for branch, cumulative in expansion.branches:
        if target < cumulative:
            state = branch
            break
    board = [
        [BRANCH_BALLS[max(0, state[r * COLS + c])] for c in range(COLS)]
        for r in range(ROWS)
    ]
    rng = _BalancedRefillRNG(
        seed,
        sample_index,
        sample_plan.count,
        sample_plan.pass_index,
        sample_plan.stream_id,
    )
    # 这些未知球的补球价值已在父分支或当前交换收益中计过，不能再次加分。
    board = refill_random(board, rng)
    _, _, tail_gain = resolve_with_refill(board, rng)
    return expansion.gain + expansion.remaining_probability * tail_gain


def _estimate_chain_gain(board, sample_plan, node_limit):
    """已完成的分支直接计分，采样预算只分配给剩余概率质量。"""
    expansion = _expand_chain_branches(board, node_limit)
    samples = [
        _sample_chain_remainder(expansion, index, sample_plan, stratified=True)
        for index in range(sample_plan.count)
    ] if expansion.branches else []
    return {
        "gain": statistics.fmean(samples) if samples else expansion.gain,
        "exact": not expansion.branches,
        "samples": samples, "nodes": expansion.nodes, "expansion": expansion,
    }


def _bounded_near_candidates(candidates, base_limit, max_limit, *, cutoff_gain=None):
    """只扩展与边界收益相近的候选，额外计算量受硬上限约束。"""
    selected = list(candidates[:base_limit])
    if not selected:
        return selected
    if cutoff_gain is None:
        cutoff_gain = selected[-1]["filter_gain"]
    for candidate in candidates[len(selected):max_limit]:
        if cutoff_gain - candidate["filter_gain"] > CANDIDATE_NEAR_GAIN:
            break
        selected.append(candidate)
    return selected


def refill_chain_opportunity_score(board):
    """按首次补球形成三连的概率加权，仅供下一步同收益候选排序。

    每个窗口单独计算，重叠窗口可能重复贡献；这不是完整连锁期望，
    不加入最终评分，也不代替后续的真实合成模拟。
    """
    cells = tuple(
        -1 if ball is None else BRANCH_BALL_CODES[ball]
        for row in board for ball in row
    )
    return sum(REFILL_WINDOW_HINTS[cells[i]][cells[j]][cells[k]] for i, j, k in TRIPLE_WINDOWS)


def _select_next_swap(board, selection_plan=None):
    """先筛出少量下一步候选，再按净收益、首次补球及连锁期望选点。"""
    candidates = []
    # 枚举顺序固定；直接收益和补球机会分都相同时，保留先遇到的交换。
    for p1, p2 in _future_merge_swaps(board):
        if AUTO_STOP_REQUESTED:
            raise StopIteration
        resolved, merge_count, gain = resolve_merges(swap_cells(board, p1, p2))
        if merge_count <= 0:
            continue
        candidates.append(make_candidate(
            p1=p1, p2=p2, board=resolved, merge_gain=gain, merge_count=merge_count
        ))
    if not candidates:
        return None
    if selection_plan is None:
        return max(candidates, key=lambda item: item["filter_gain"])

    shortlist = _shortlist_candidates(
        candidates, NEXT_SWAP_CANDIDATE_LIMIT,
        tie_breaker=refill_chain_opportunity_score,
    )

    best = None
    best_gain = -math.inf
    estimates = {}
    for candidate in shortlist:
        key = _board_key(candidate["board"])
        if key not in estimates:
            estimates[key] = _estimate_chain_gain(candidate["board"], selection_plan, NEXT_CHAIN_NODE_LIMIT)
        estimate = estimates[key]
        expected_gain = candidate["filter_gain"] + estimate["gain"]
        if expected_gain > best_gain:
            best_gain = expected_gain
            best = {**candidate, "chain_estimate": estimate}
    return best


def _evaluate_candidate_samples(board_after_current_merge, sample_plan, current_step):
    """当前连锁优先精确求期望，下一步在单独的补球局面样本上评估。"""
    current_plan = make_sampling_plan(
        sample_plan.pass_index, CURRENT_CHAIN_SAMPLE_COUNT, stream_id=sample_plan.stream_id + 2
    )
    current = _estimate_chain_gain(board_after_current_merge, current_plan, CURRENT_CHAIN_NODE_LIMIT)

    result = {
        "current_first_refill_gain": len(empty_cells(board_after_current_merge)) * EXPECTED_REFILL_ENERGY,
        "current_chain_gain": current["gain"],
        "current_chain_samples": current["samples"],
        "current_sample_count": len(current["samples"]),
        "current_chain_exact": current["exact"],
        "current_chain_nodes": current["nodes"],
        "current_chain_exact_part": current["expansion"].gain,
        "current_chain_remaining_probability": current["expansion"].remaining_probability,
        "next_gain": 0.0,
        "next_chain_gain": 0.0,
        "future_gain": 0.0,
        "next_gain_samples": [],
        "next_chain_samples": [],
        "next_sample_swaps": [],
        "next_legal_count": 0,
        "next_chain_exact_count": 0,
        "lookahead_enabled": current_step < TOTAL_GAME_MOVES,
    }
    if not result["lookahead_enabled"]:
        return result

    next_gain_samples = []
    next_chain_samples = []
    future_gain_samples = []
    next_sample_swaps = []
    next_sample_plan = make_sampling_plan(
        sample_plan.pass_index, sample_plan.count, stream_id=sample_plan.stream_id + 1
    )
    selection_plan = make_sampling_plan(
        sample_plan.pass_index, NEXT_CHAIN_SAMPLE_COUNT, stream_id=sample_plan.stream_id + 3
    )
    exact_count = 0
    next_choices = {}

    for sample_index in range(sample_plan.count):
        if AUTO_STOP_REQUESTED:
            raise StopIteration
        state, _, _ = _simulate_refill_chain_once(board_after_current_merge, sample_index, sample_plan)
        state_key = _board_key(state)
        # 同一批前瞻样本若落到相同局面，下一步选择相同；打分样本仍分别处理。
        if state_key not in next_choices:
            next_choices[state_key] = _select_next_swap(state, selection_plan)
        sample_best = next_choices[state_key]
        if sample_best is None:
            next_sample_swaps.append(None)
            next_gain_samples.append(0.0)
            next_chain_samples.append(0.0)
            future_gain_samples.append(0.0)
            continue

        estimate = sample_best["chain_estimate"]
        if estimate["exact"]:
            chain_gain = estimate["gain"]
            exact_count += 1
        else:
            # 选点用的备用样本不能再次用作最终打分，避免挑中幸运样本。
            chain_gain = _sample_chain_remainder(
                estimate["expansion"], sample_index, next_sample_plan
            )
        next_sample_swaps.append((sample_best["p1"], sample_best["p2"]))
        next_gain_samples.append(sample_best["filter_gain"])
        next_chain_samples.append(chain_gain)
        future_gain_samples.append(sample_best["filter_gain"] + chain_gain)

    result.update(
        next_gain=statistics.fmean(next_gain_samples),
        next_chain_gain=statistics.fmean(next_chain_samples),
        future_gain=statistics.fmean(future_gain_samples),
        next_sample_swaps=next_sample_swaps,
        next_gain_samples=next_gain_samples,
        next_chain_samples=next_chain_samples,
        next_legal_count=sum(swap is not None for swap in next_sample_swaps),
        next_chain_exact_count=exact_count,
    )
    return result


def _complete_candidate(candidate, evaluation, sample_plan):
    """交换收益包含首次补球；最终评分按实际纳入评估的步数取平均。"""
    result = {**candidate, **evaluation}
    result["sample_pass"] = sample_plan.pass_index
    result["current_swap_gain"] = (
        result["immediate_gain"] + result["current_first_refill_gain"]
    )
    result["current_gain"] = result["current_swap_gain"] + result["current_chain_gain"]
    if result["lookahead_enabled"]:
        result["real_score"] = result["current_gain"] + result["future_gain"]
        result["final_score"] = result["real_score"] / 2.0
    else:
        result["real_score"] = result["current_gain"]
        result["final_score"] = result["current_gain"]
    return result


def _evaluate_candidates(candidates, sample_plan, current_step):
    """同一轮中相同棋盘只提交一次评估任务。"""
    evaluated = {}
    pending = {}

    for candidate in candidates:
        board_key = _board_key(candidate["board"])
        if board_key not in pending:
            pending[board_key] = candidate["board"]

    worker_count = min(EVALUATION_WORKERS, len(pending))
    if pending:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                key: executor.submit(
                    _evaluate_candidate_samples, board, sample_plan, current_step
                )
                for key, board in pending.items()
            }
            for key, future in futures.items():
                evaluation = future.result()
                evaluated[key] = evaluation

    results = []
    for index, candidate in enumerate(candidates, 1):
        if AUTO_STOP_REQUESTED:
            raise StopIteration
        board_key = _board_key(candidate["board"])
        results.append(
            _complete_candidate(candidate, evaluated[board_key], sample_plan)
        )
        print(f"\r{index}/{len(candidates)}", end="", flush=True)
    print()
    results.sort(key=lambda result: result["final_score"], reverse=True)
    return results, worker_count


def swap_opportunity_score(board):
    """按一次换球能形成的三连机会排序；仅作快筛同收益时的辅助指标。"""
    opportunities = {(target, ball) for target, ball, _ in _swap_setups(board)}

    # 重叠窗口或多个供球位置，不重复奖励同一次换入需求。
    # 用单组三连的净收益及补球价值加权；机会总分不代表可同时获得的能量。
    return sum(
        BALL_ENERGY[color][level + 1]
        - 3 * BALL_ENERGY[color][level]
        + EXPECTED_REFILL_GAIN_PER_MERGE
        for _, (color, level) in opportunities
    )


def _shortlist_candidates(candidates, limit, *, lookahead_enabled=True, tie_breaker=None):
    """主收益优先，只对跨越入选边界的同收益候选计算三连机会分。"""
    if limit <= 0:
        return []
    candidates.sort(key=lambda result: result["filter_gain"], reverse=True)
    # 最后一步不再用未来交换机会决定入选。
    if len(candidates) <= limit or not lookahead_enabled:
        return candidates[:limit]

    cutoff = candidates[limit - 1]["filter_gain"]
    if candidates[limit]["filter_gain"] != cutoff:
        return candidates[:limit]

    start, end = limit - 1, limit + 1
    while start > 0 and candidates[start - 1]["filter_gain"] == cutoff:
        start -= 1
    while end < len(candidates) and candidates[end]["filter_gain"] == cutoff:
        end += 1
    tied = candidates[start:end]
    if tie_breaker is None:
        tie_breaker = swap_opportunity_score
    # 机会分也相同时，稳定排序保留原有坐标顺序。
    tied.sort(key=lambda result: tie_breaker(result["board"]), reverse=True)
    return candidates[:start] + tied[: limit - start]


def _refine_shortlist(candidates, current_step):
    """保留头部候选，用连锁期望重评边界内外的少量交换。"""
    limit = FAST_FILTER_LIMIT
    initial = _shortlist_candidates(
        candidates, limit, lookahead_enabled=current_step < TOTAL_GAME_MOVES
    )
    if len(candidates) <= limit or not initial:
        return initial
    chosen = {(item["p1"], item["p2"]) for item in initial}
    extras = [item for item in candidates if (item["p1"], item["p2"]) not in chosen]
    protected = min(FILTER_PROTECTED_COUNT, max(0, limit - 1))
    minimum_protected = min(FILTER_MIN_PROTECTED_COUNT, protected)
    cutoff = initial[protected]["filter_gain"]
    while (protected > minimum_protected
           and initial[protected - 1]["filter_gain"] - cutoff <= CANDIDATE_NEAR_GAIN):
        protected -= 1
    extras = _bounded_near_candidates(
        extras, FILTER_EXTRA_CANDIDATES, FILTER_MAX_EXTRA_CANDIDATES,
        cutoff_gain=initial[-1]["filter_gain"],
    )
    boundary = initial[protected:] + extras
    plan = make_sampling_plan(current_step, FILTER_CHAIN_SAMPLE_COUNT, stream_id=4)
    ranked = []
    for candidate in boundary:
        estimate = _estimate_chain_gain(candidate["board"], plan, FILTER_CHAIN_NODE_LIMIT)
        ranked.append((candidate, candidate["filter_gain"] + estimate["gain"]))
    # 相同估计收益时保留原入选者；精筛样本与最终备用样本使用不同序列。
    ranked.sort(key=lambda item: item[1], reverse=True)
    return initial[:protected] + [item for item, _ in ranked[:limit - protected]]


def analyze_all_swaps(board, *, current_step=1):
    """快速筛选固定数量候选，当前连锁优先精算，前瞻使用共同补球样本。

    有前瞻时最终评分为两步完整收益的平均；第 20 步及以后直接取当前完整收益。
    快速筛选同收益时参考三连机会分，再用连锁期望重评边界候选。
    """
    current_step = int(current_step)
    start = time.perf_counter()
    quick = []
    for p1, p2 in ALL_SWAPS:
        if AUTO_STOP_REQUESTED:
            raise StopIteration
        candidate = evaluate_swap(board, p1, p2)
        if candidate is not None:
            quick.append(candidate)

    if not quick:
        return []

    quick = _refine_shortlist(quick, current_step)
    quick_elapsed = time.perf_counter() - start

    final_start = time.perf_counter()
    sample_plan = make_evaluation_plan(current_step)
    final, worker_count = _evaluate_candidates(
        quick, sample_plan, current_step
    )
    final_elapsed = time.perf_counter() - final_start

    if PRINT_SEARCH_BENCHMARK:
        elapsed = time.perf_counter() - start
        print(
            f"[搜索] {len(ALL_SWAPS)} -> {len(quick)} | "
            f"当前连锁精算={sum(r['current_chain_exact'] for r in final)}/{len(final)} | "
            f"前瞻样本={sample_plan.count if current_step < TOTAL_GAME_MOVES else 0} | 线程={worker_count} | "
            f"快速筛选={quick_elapsed:.3f}s | 最终评估={final_elapsed:.3f}s | "
            f"总耗时={elapsed:.3f}s"
        )
    return final


# 输出与调试信息


def cell_to_text(cell):
    return "?" if cell is None else f"{cell[1]}{cell[0]}"


def position_name(pos):
    return f"r{pos[0] + 1}c{pos[1] + 1}"


def print_board_state(board):
    print()
    print("=" * 90)
    print("识别结果")
    print("=" * 90)
    for r in range(ROWS):
        print(
            f"第{r + 1}行： "
            + "   ".join(cell_to_text(board[r][c]) for c in range(COLS))
        )


# DEBUG 图片与筛选结果文件
def _debug_writer_loop():
    while True:
        task = DEBUG_WRITE_QUEUE.get()
        try:
            if task is DEBUG_WRITER_STOP:
                return

            save_board_debug_images(*task)
        except Exception as exc:
            print(f"DEBUG 图片保存失败：{exc!r}")
        finally:
            DEBUG_WRITE_QUEUE.task_done()


def start_debug_writer():
    global DEBUG_WRITER_THREAD

    if not DEBUG_SAVE_IMAGES:
        return

    if DEBUG_WRITER_THREAD is None or not DEBUG_WRITER_THREAD.is_alive():
        DEBUG_WRITER_THREAD = threading.Thread(
            target=_debug_writer_loop, name="debug-writer", daemon=True
        )
        DEBUG_WRITER_THREAD.start()


def enqueue_debug_images(screenshot, x1, y1, dx, dy, debug_dir, step):
    if not DEBUG_SAVE_IMAGES:
        return

    start_debug_writer()
    DEBUG_WRITE_QUEUE.put((screenshot.copy(), x1, y1, dx, dy, debug_dir, step))


def finish_debug_writer():
    global DEBUG_WRITER_THREAD

    if DEBUG_WRITER_THREAD is None:
        return

    # 退出前等待所有图片写完，确保 DEBUG 文件不会丢失。
    DEBUG_WRITE_QUEUE.join()
    DEBUG_WRITE_QUEUE.put(DEBUG_WRITER_STOP)
    DEBUG_WRITE_QUEUE.join()
    DEBUG_WRITER_THREAD.join()
    DEBUG_WRITER_THREAD = None


def get_debug_step_dir(step):
    """返回单次识别对应的 DEBUG 目录，仅在启用 DEBUG 时创建。"""

    step_dir = os.path.join(DEBUG_DIR, f"step_{step:02d}")

    if DEBUG_SAVE_IMAGES:
        os.makedirs(step_dir, exist_ok=True)

    return step_dir


def save_board_debug_images(screenshot, x1, y1, dx, dy, debug_dir, step):
    """保存一次已完成交换对应的棋盘和逐格 DEBUG 图片。"""

    os.makedirs(BOARD_DEBUG_DIR, exist_ok=True)

    cv2.imwrite(os.path.join(debug_dir, "board.png"), screenshot)

    # 独立目录按成功交换步数保存；step 目录保存对应的历史记录。
    cv2.imwrite(os.path.join(BOARD_DEBUG_DIR, f"board_{step:02d}.png"), screenshot)

    cell_w = abs(dx) * 0.90
    cell_h = abs(dy) * 0.90

    for row in range(ROWS):
        for col in range(COLS):
            cx = x1 + col * dx
            cy = y1 + row * dy

            cell = crop_cell(screenshot, cx, cy, cell_w, cell_h)

            if cell is not None:
                recognize_cell(cell, row, col, debug_dir)


def save_swap_results(results, step):
    """将筛选结果直接写入文件，不修改其他线程共用的标准输出。"""
    if not DEBUG_SAVE_IMAGES:
        return
    os.makedirs(SEARCH_DEBUG_DIR, exist_ok=True)
    output_path = os.path.join(SEARCH_DEBUG_DIR, f"search_step_{step:02d}.txt")
    with open(output_path, "w", encoding="utf-8") as file:
        print_swap_results(results, file=file)


def print_swap_results(results, *, file=None):
    emit = partial(print, file=file)
    emit()
    emit("=" * 120)
    emit("筛选结果")
    emit("=" * 120)

    if not results:
        emit("当前没有找到有效交换。")
        return

    # 输出全部进入最终评估的候选。
    for i, result in enumerate(results, 1):
        emit(f"{i}. {position_name(result['p1'])} <-> {position_name(result['p2'])}")

        emit(f"   当前交换收益 = {result.get('current_swap_gain', 0.0):.2f}")

        current_samples = result.get("current_chain_samples", [])
        current_sample_count = result.get("current_sample_count", len(current_samples))

        if result.get("current_chain_exact", False):
            emit(f"   当前连锁收益期望 = {result['current_chain_gain']:.2f}（概率分支精确计算）")
        elif current_samples:
            emit(
                "   当前连锁混合估计：平均="
                + f"{statistics.fmean(current_samples):.2f} | "
                + ", ".join(f"{x:.2f}" for x in current_samples)
                + f" ({current_sample_count}次)"
            )
            emit(
                f"   已计入精确收益 = {result['current_chain_exact_part']:.2f} | "
                f"剩余分支概率 = {result['current_chain_remaining_probability']:.2%}"
            )

        emit(f"   当前完整收益 = {result.get('current_gain', 0.0):.2f}")

        if result.get("lookahead_enabled", True):
            emit("   下一步期望最优交换 = 根据实际补球结果重新选择")

            emit(f"   下一步交换收益期望 = {result.get('next_gain', 0.0):.2f}")

            next_chain_samples = result.get("next_chain_samples", [])
            if next_chain_samples:
                emit(
                    "   下一步连锁样本（"
                    f"{len(next_chain_samples)}个）："
                    + ", ".join(f"{x:.2f}" for x in next_chain_samples)
                    + " | 平均="
                    + f"{statistics.fmean(next_chain_samples):.2f}"
                )

            emit(f"   下一步完整收益期望 = {result.get('future_gain', 0.0):.2f}")

        else:
            emit("   已到最后一步，最终评分不计入下一步交换收益期望")

        emit(f"   最终评分 = {result.get('final_score', 0.0):.2f}")


def print_best_swap(results):
    print()
    print("=" * 90)
    print("推荐交换")
    print("=" * 90)

    if not results:
        print()
        print("没有找到可以立即合成的交换。")
        return

    best = results[0]

    print()
    print(f"第1步：点击 {position_name(best['p1'])}")

    print(f"第2步：点击 {position_name(best['p2'])}")

    print()
    print(f"当前交换收益：{best.get('current_swap_gain', 0.0):.2f}")
    current_samples = best.get("current_chain_samples", [])
    if best.get("current_chain_exact", False):
        print(f"当前连锁收益期望：{best['current_chain_gain']:.2f}（概率分支精确计算）")
    elif current_samples:
        print(
            f"当前连锁混合估计（{len(current_samples)}次）："
            + ", ".join(f"{x:.2f}" for x in current_samples)
            + " | 平均="
            + f"{statistics.fmean(current_samples):.2f}"
        )
        print(
            f"已计入精确收益：{best['current_chain_exact_part']:.2f} | "
            f"剩余分支概率：{best['current_chain_remaining_probability']:.2%}"
        )

    print(f"当前完整收益：{best.get('current_gain', 0.0):.2f}")

    if best.get("lookahead_enabled", True):
        print(f"下一步交换收益期望：{best.get('next_gain', 0.0):.2f}")

        next_chain_samples = best.get("next_chain_samples", [])
        if next_chain_samples:
            print(
                "下一步连锁样本（"
                f"{len(next_chain_samples)}个）："
                + ", ".join(f"{x:.2f}" for x in next_chain_samples)
                + " | 平均="
                + f"{statistics.fmean(next_chain_samples):.2f}"
            )

        print(f"下一步完整收益期望：{best.get('future_gain', 0.0):.2f}")
    else:
        print("已到最后一步，最终评分不计入下一步交换收益期望")
    print(f"最终评分：{best.get('final_score', 0.0):.2f}")


# 自动运行与程序入口
def run_auto_loop(x1, y1, dx, dy):
    """
    自动运行主循环：

        截图
        ↓
        识别
        ↓
        计算当前最优交换
        ↓
        点击
        ↓
        检测棋盘稳定
        ↓
        下一轮

    每步重新截图，搜索结果只用于本次交换。
    """

    loop_start = time.monotonic()

    move_count = 0
    last_unknown_count = None
    waiting_for_board = False

    setup_hotkeys()

    with mss.MSS() as sct:
        full_monitor = sct.monitors[0]
        monitor = get_board_capture_monitor(x1, y1, dx, dy, full_monitor)
        capture_x1, capture_y1 = get_capture_board_origin(x1, y1, monitor)

        while True:
            if AUTO_STOP_REQUESTED:
                break

            if RESELECT_REQUESTED.is_set():
                RESELECT_REQUESTED.clear()
                calibration = calibrate(wait_for_enter=False)
                calibrated = wait_for_calibrated_board(
                    sct, full_monitor, calibration
                )
                if calibrated is None:
                    break
                x1, y1, dx, dy, monitor = calibrated
                capture_x1, capture_y1 = get_capture_board_origin(x1, y1, monitor)
                last_unknown_count = None
                waiting_for_board = False
                print("已更新棋盘框选区域，继续自动运行。")
                continue

            wait_while_paused()

            if AUTO_STOP_REQUESTED:
                break
            if RESELECT_REQUESTED.is_set():
                continue

            # 防止异常情况下无限运行。

            elapsed = time.monotonic() - loop_start

            if elapsed >= GAME_MAX_SECONDS:
                print()
                print("=" * 90)

                print(f"自动运行时间达到 {GAME_MAX_SECONDS:.1f}s，停止。")

                break

            current_step = move_count + 1
            if not waiting_for_board:
                print()
                print("=" * 90)

                print(f"自动第 {current_step} 步")

            # 截取当前棋盘。

            raw = np.array(sct.grab(monitor))

            screenshot = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            # 与初始等待共用同一条识别路径。
            _, unknown_count, board_state = recognize_board_snapshot(
                screenshot, capture_x1, capture_y1, dx, dy
            )

            if unknown_count > 0:
                # 20 步完成后只要仍有未识别格，通常表示游戏已结束。
                # 使用与 F9 相同的停止流程，不再继续等待或误执行下一步。
                if move_count >= TOTAL_GAME_MOVES:
                    request_auto_stop("已完成 20 步，自动退出")
                    break

                if unknown_count != last_unknown_count:
                    print(
                        f"\r本轮未识别到{unknown_count}个能量球，等待中...          ",
                        end="",
                        flush=True,
                    )

                last_unknown_count = unknown_count
                waiting_for_board = True

                time.sleep(BOARD_CHECK_INTERVAL)

                continue

            if waiting_for_board:
                print("\r" + " " * 60 + "\r", end="", flush=True)

            last_unknown_count = None
            waiting_for_board = False

            # 计算当前最优交换。

            print_board_state(board_state)

            try:
                results = analyze_all_swaps(board_state, current_step=current_step)
            except StopIteration:
                # StopIteration 表示热键触发的正常停止，不视为程序错误。
                print()
                print("=" * 90)
                print("收到停止指令，停止当前搜索。")
                print("=" * 90)
                break

            # 搜索完成后再次检查停止状态，避免误执行点击。
            if AUTO_STOP_REQUESTED:
                print()
                print("收到停止指令，不执行本轮交换。")
                break

            # 暂停恢复后重新截图和搜索；F7 必须先更新区域，不能点击旧坐标。
            if RESELECT_REQUESTED.is_set() or AUTO_PAUSED:
                continue

            if not results:
                # 没有立即可合成交换时暂停，等待用户处理。
                pause_for_no_move()

                if AUTO_STOP_REQUESTED:
                    break

                wait_while_paused()

                continue

            print_best_swap(results)

            # 执行选中的交换。

            swap_completed = click_best_swap(
                results, x1, y1, dx, dy, sct, monitor, step=current_step
            )

            if swap_completed is RESELECT_RESULT or swap_completed is RETRY_RESULT:
                continue

            if swap_completed:
                # 仅以完成两次点击计步；静态棋盘测试也正常计数。
                move_count += 1
                save_swap_results(results, move_count)

                if DEBUG_SAVE_IMAGES:
                    debug_step_dir = get_debug_step_dir(move_count)

                    enqueue_debug_images(
                        screenshot,
                        capture_x1,
                        capture_y1,
                        dx,
                        dy,
                        debug_step_dir,
                        move_count,
                    )

    print()
    print("=" * 90)

    print("自动运行结束")

    print(f"完成交换：{move_count}")


def main():
    """完成热键注册、棋盘校准、初始识别和自动交换。"""

    global AUTO_PAUSED
    global AUTO_STOP_REQUESTED

    AUTO_PAUSED = False
    AUTO_STOP_REQUESTED = False
    RESELECT_REQUESTED.clear()
    setup_hotkeys()

    print()
    print("=" * 90)
    print("三打白骨精 项目地址:https://github.com/Fylxom/ZmxyUnpack/releases/tag/1")
    print("=" * 90)

    print()
    print("F7：重新框选")
    print("F8：暂停 / 继续")
    print("F9：退出")

    # 只有启用 DEBUG 时才创建根目录。
    if DEBUG_SAVE_IMAGES:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    calibration = calibrate()

    if calibration is None or AUTO_STOP_REQUESTED:
        print("收到停止指令，程序结束。")
        return

    # 框选完成后直接等待完整棋盘稳定出现。
    with mss.MSS() as sct:
        full_monitor = sct.monitors[0]
        calibrated = wait_for_calibrated_board(sct, full_monitor, calibration)
        if calibrated is None:
            print("收到停止指令，程序结束。")
            return
        x1, y1, dx, dy, _ = calibrated
        print("初始棋盘已稳定，开始截图...")

    run_auto_loop(x1, y1, dx, dy)

    print()
    print("=" * 90)
    print("识别完成")
    print("=" * 90)

    print()
    print(f"调试目录：{os.path.abspath(DEBUG_DIR)}")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("用户取消。")

    except Exception as e:
        print()
        print("=" * 80)
        print("程序发生错误")
        print("=" * 80)

        print(repr(e))

        import traceback

        traceback.print_exc()

        print()
        input("按 Enter 退出...")

    finally:
        close_selection_overlay()
        finish_debug_writer()
