# 是否保存 DEBUG 调试图片；正常运行建议关闭，以免大量写盘拖慢点击。
DEBUG_SAVE_IMAGES = False

# 是否输出搜索各阶段耗时。
PRINT_SEARCH_BENCHMARK = True

# 是否只截取棋盘区域。点击坐标仍使用屏幕绝对坐标。
CAPTURE_BOARD_ONLY = True

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import keyboard
import cv2
import numpy as np
import pyautogui
import mss
import os
import math
import time
import random
import threading
import statistics
import queue
import contextlib
import io
import ctypes

# 自动运行状态
AUTO_PAUSED = False
AUTO_STOP_REQUESTED = False
HOTKEYS_REGISTERED = False
RESELECT_REQUESTED = threading.Event()
RESELECT_RESULT = object()

# 当前校准方框。程序运行期间保留显示，退出时统一关闭。
SELECTION_OVERLAY = None

# 交换评估缓存
SWAP_EVAL_CACHE = {}


# 颜色编码：火为 f、雷为 r、风为 w。
# 识别主要依据颜色面积；连通区域和中心位置用于筛选候选区域，
# 不使用模板匹配或 OCR。


ROWS = 3
COLS = 9

# 按当前棋盘截图 862×288 作为正常尺寸，防止误框选过小区域。
REFERENCE_BOARD_WIDTH = 862
REFERENCE_BOARD_HEIGHT = 288
MIN_BOARD_WIDTH = REFERENCE_BOARD_WIDTH / 2
MIN_BOARD_HEIGHT = REFERENCE_BOARD_HEIGHT / 2

DEBUG_DIR = "debug"
BOARD_DEBUG_DIR = os.path.join(DEBUG_DIR, "board")
SEARCH_DEBUG_DIR = os.path.join(DEBUG_DIR, "search")


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
    return (
        x1 - monitor["left"],
        y1 - monitor["top"],
    )


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
    b, g, r = cv2.split(screenshot)
    frame_mask = (
        (r >= 150)
        & (g >= 90)
        & (b >= 60)
        & (r >= g + 25)
        & (g >= b + 15)
    )

    x_projection = frame_mask.sum(axis=0)
    y_projection = frame_mask.sum(axis=1)

    x_runs = _projection_runs(
        x_projection,
        max(8, int(frame_mask.shape[0] * 0.08)),
    )
    y_runs = _projection_runs(
        y_projection,
        max(8, int(frame_mask.shape[1] * 0.08)),
    )

    def find_centers(runs, expected_start, expected_step, count):
        pitch = abs(expected_step)
        if pitch <= 1:
            return None

        candidates = [
            run
            for run in runs
            if pitch * 0.45 <= run[1] - run[0] + 1 <= pitch * 1.20
        ]

        centers = []
        used = set()

        for index in range(count):
            expected = expected_start + index * expected_step
            available = [
                run
                for run_index, run in enumerate(candidates)
                if run_index not in used
                and abs((run[0] + run[1]) / 2 - expected)
                <= pitch * 0.55
            ]

            if not available:
                return None

            run = min(
                available,
                key=lambda item: abs(
                    (item[0] + item[1]) / 2 - expected
                ),
            )
            used.add(candidates.index(run))
            centers.append((run[0] + run[1]) / 2)

        return centers

    x_centers = find_centers(
        x_runs,
        x1,
        dx,
        COLS,
    )
    y_centers = find_centers(
        y_runs,
        y1,
        dy,
        ROWS,
    )

    if x_centers is None or y_centers is None:
        return None

    return (
        x_centers[0],
        y_centers[0],
        (x_centers[-1] - x_centers[0]) / (COLS - 1),
        (y_centers[-1] - y_centers[0]) / (ROWS - 1),
    )

# DEBUG 图片由独立线程顺序写入，避免阻塞主循环的识别、搜索和点击。
DEBUG_WRITE_QUEUE = queue.Queue()
DEBUG_WRITER_THREAD = None
DEBUG_WRITER_STOP = object()


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

    if (
        DEBUG_WRITER_THREAD is None
        or not DEBUG_WRITER_THREAD.is_alive()
    ):
        DEBUG_WRITER_THREAD = threading.Thread(
            target=_debug_writer_loop,
            name="debug-writer",
            daemon=True,
        )
        DEBUG_WRITER_THREAD.start()


def enqueue_debug_images(
    screenshot,
    x1,
    y1,
    dx,
    dy,
    debug_dir,
    step,
):
    if not DEBUG_SAVE_IMAGES:
        return

    start_debug_writer()
    DEBUG_WRITE_QUEUE.put(
        (
            screenshot.copy(),
            x1,
            y1,
            dx,
            dy,
            debug_dir,
            step,
        )
    )


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

    step_dir = os.path.join(
        DEBUG_DIR,
        f"step_{step:02d}"
    )

    if DEBUG_SAVE_IMAGES:
        os.makedirs(
            step_dir,
            exist_ok=True
        )

    return step_dir


# 每个方格只取中央区域，尽量排除方框、边框和外部背景。
CENTER_RATIO = 0.82


# 背景颜色与候选颜色的最小距离。
BACKGROUND_DISTANCE = 35


# 小于该比例的颜色区域视为噪声。
MIN_COLOR_RATIO = 0.005


# 阈值是“颜色像素面积 / 中央识别区域面积”，不是整个球的真实面积。
# 三种颜色分别设置 1/2 级和 2/3 级分界。

LEVEL_THRESHOLDS = {

    "f": {

        # 1 / 2 分界
        "12": 0.165,

        # 2 / 3 分界
        "23": 0.260
    },


    "r": {

        # 1 / 2 分界
        "12": 0.285,

        # 2 / 3 分界
        "23": 0.410
    },


    "w": {

        # 1 / 2 分界
        "12": 0.315,

        # 2 / 3 分界
        "23": 0.555
    }
}


# DEBUG 图片中的 BGR 标注颜色。
DRAW_COLOR = {

    "f": (0, 0, 255),

    "r": (255, 0, 255),

    "w": (255, 255, 0),

    "?": (0, 255, 255)
}


# 每个补球位置的 f/r/w 严格各占 1/3。
# 所有候选统一使用 9 个样本评估，避免中间阶段因样本过少误淘汰。
FINAL_SAMPLE_COUNT = 9

# 搜索运行统计，只用于观察计算量，不参与评分。
SEARCH_STATS = {
    "cache_hits": 0,
}


def is_left_mouse_button_down():
    """读取 Windows 当前左键状态。"""
    return bool(
        ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000
    )


def set_overlay_click_through(overlay):
    """让已完成的校准方框不拦截游戏鼠标点击。"""
    hwnd = overlay.winfo_id()
    user32 = ctypes.windll.user32
    get_window_long = getattr(
        user32,
        "GetWindowLongPtrW",
        user32.GetWindowLongW,
    )
    set_window_long = getattr(
        user32,
        "SetWindowLongPtrW",
        user32.SetWindowLongW,
    )

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
        overlay["borders"],
        geometries,
    ):
        border.geometry(
            f"{border_width}x{border_height}"
            f"{geometry_offset(x)}{geometry_offset(y)}"
        )

    overlay["root"].update_idletasks()
    overlay["root"].update()


def select_board_region():
    """等待用户拖动鼠标框选棋盘，并返回四个边界坐标。"""
    global SELECTION_OVERLAY

    import tkinter as tk

    close_selection_overlay()

    print()
    print(
        "按住鼠标左键，拖动鼠标, "
        "确保所有格子都包括在内后，松开鼠标完成框选。"
    )

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

        overlay = {
            "root": root,
            "borders": borders,
        }
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
            wait_while_paused()
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
        print(
            "确保程序切到前台，且所有格子完整显示后，"
            "按 Enter 开始框选。"
        )
        input()

    wait_while_paused()

    if AUTO_STOP_REQUESTED:
        return None

    while True:
        selected_region = select_board_region()

        if selected_region is None:
            return None

        grid_left, grid_top, grid_right, grid_bottom = selected_region
        grid_width = grid_right - grid_left
        grid_height = grid_bottom - grid_top

        if (
            grid_width >= MIN_BOARD_WIDTH
            and grid_height >= MIN_BOARD_HEIGHT
        ):
            break

        print(
            "框选区域过小，请重新框选。(按F9可退出程序)"
        )

    dx = grid_width / COLS
    dy = grid_height / ROWS

    # x1、y1 表示第 1 行第 1 列球的中心，点击坐标仍然准确。
    x1 = grid_left + dx / 2
    y1 = grid_top + dy / 2

    return x1, y1, dx, dy


def crop_cell(
    screenshot,
    cx,
    cy,
    cell_w,
    cell_h
):
    """按中心点和尺寸截取格子，越界区域会自动裁剪。"""

    h, w = screenshot.shape[:2]


    x1 = int(
        round(
            cx -
            cell_w / 2
        )
    )


    y1 = int(
        round(
            cy -
            cell_h / 2
        )
    )


    x2 = int(
        round(
            cx +
            cell_w / 2
        )
    )


    y2 = int(
        round(
            cy +
            cell_h / 2
        )
    )


    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        w,
        x2
    )

    y2 = min(
        h,
        y2
    )


    if x2 <= x1 or y2 <= y1:

        return None


    return screenshot[
        y1:y2,
        x1:x2
    ]


def get_center_region(cell):
    """截取格子中央区域，减少边框和背景对识别的干扰。"""

    h, w = cell.shape[:2]


    cw = int(
        w *
        CENTER_RATIO
    )


    ch = int(
        h *
        CENTER_RATIO
    )


    x1 = (
        w -
        cw
    ) // 2


    y1 = (
        h -
        ch
    ) // 2


    return cell[
        y1:y1 + ch,
        x1:x1 + cw
    ]


def estimate_background(img):
    """用格子四角的像素中位数估计当前格子的背景颜色。"""

    h, w = img.shape[:2]


    patch = max(
        3,
        int(
            min(h, w) *
            0.12
        )
    )


    patches = []


    patches.append(
        img[
            0:patch,
            0:patch
        ]
    )


    patches.append(
        img[
            0:patch,
            w-patch:w
        ]
    )


    patches.append(
        img[
            h-patch:h,
            0:patch
        ]
    )


    patches.append(
        img[
            h-patch:h,
            w-patch:w
        ]
    )


    pixels = []


    for p in patches:

        pixels.append(
            p.reshape(
                -1,
                3
            )
        )


    pixels = np.concatenate(
        pixels,
        axis=0
    )


    background = np.median(
        pixels,
        axis=0
    )


    return background.astype(
        np.float32
    )


def get_background_mask(
    img,
    background
):
    """返回与估计背景颜色差异足够大的像素掩码。"""

    diff = (
        img.astype(
            np.float32
        )
        -
        background.reshape(
            1,
            1,
            3
        )
    )


    distance = np.sqrt(
        np.sum(
            diff * diff,
            axis=2
        )
    )


    return (
        distance >
        BACKGROUND_DISTANCE
    )


def make_color_masks(img):
    """按颜色特征生成火、雷、风三种候选区域掩码。"""

    b = img[
        :,
        :,
        0
    ].astype(
        np.int16
    )


    g = img[
        :,
        :,
        1
    ].astype(
        np.int16
    )


    r = img[
        :,
        :,
        2
    ].astype(
        np.int16
    )


    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )


    h = hsv[
        :,
        :,
        0
    ].astype(
        np.int16
    )


    s = hsv[
        :,
        :,
        1
    ].astype(
        np.int16
    )


    v = hsv[
        :,
        :,
        2
    ].astype(
        np.int16
    )


    background = \
        estimate_background(img)


    not_background = \
        get_background_mask(
            img,
            background
        )


    # 火是红橙色，不能只用 R > G，因为棋盘背景也偏棕橙色。

    fire_hue = (
        (h <= 15)
        |
        (h >= 170)
    )


    fire_mask = (

        fire_hue

        &

        (s >= 135)

        &

        (v >= 100)

        &

        (r >= g + 35)

        &

        (r >= b + 60)

        &

        not_background
    )


    # 雷是紫色。

    thunder_hue = (

        (h >= 125)

        &

        (h <= 165)
    )


    thunder_mask = (

        thunder_hue

        &

        (s >= 80)

        &

        (v >= 80)

        &

        (r >= g + 20)

        &

        (b >= g + 20)

        &

        not_background
    )


    # 风是灰蓝色。

    wind_mask = (

        (s <= 105)

        &

        (v >= 75)

        &

        (b >= r + 3)

        &

        (b >= g - 8)

        &

        not_background
    )


    # 开运算去除小噪声。

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    fire_mask = cv2.morphologyEx(
        fire_mask.astype(
            np.uint8
        ),
        cv2.MORPH_OPEN,
        kernel
    )


    thunder_mask = cv2.morphologyEx(
        thunder_mask.astype(
            np.uint8
        ),
        cv2.MORPH_OPEN,
        kernel
    )


    wind_mask = cv2.morphologyEx(
        wind_mask.astype(
            np.uint8
        ),
        cv2.MORPH_OPEN,
        kernel
    )


    # 闭运算连接同一能量球中被高光分开的区域。

    kernel2 = np.ones(
        (5, 5),
        np.uint8
    )


    fire_mask = cv2.morphologyEx(
        fire_mask,
        cv2.MORPH_CLOSE,
        kernel2
    )


    thunder_mask = cv2.morphologyEx(
        thunder_mask,
        cv2.MORPH_CLOSE,
        kernel2
    )


    wind_mask = cv2.morphologyEx(
        wind_mask,
        cv2.MORPH_CLOSE,
        kernel2
    )


    return (
        fire_mask * 255,
        thunder_mask * 255,
        wind_mask * 255,
        background
    )


def get_best_component(mask):
    """从掩码中选择最可能代表能量球的连通区域。"""

    h, w = mask.shape


    center_x = w / 2
    center_y = h / 2


    count, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )


    best_label = -1
    best_score = -1


    for i in range(
        1,
        count
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]


        if area < 5:

            continue


        cx = centroids[i][0]
        cy = centroids[i][1]


        distance = math.sqrt(

            (cx - center_x) ** 2

            +

            (cy - center_y) ** 2
        )


        center_factor = 1.0 / (
            1.0 +
            distance
        )


        score = area * (
            0.5 +
            center_factor * 2.0
        )


        if score > best_score:

            best_score = score

            best_label = i


    if best_label < 0:

        return (
            np.zeros_like(mask),
            0
        )


    component = np.zeros_like(
        mask
    )


    component[
        labels == best_label
    ] = 255


    area = int(
        np.sum(
            component > 0
        )
    )


    return (
        component,
        area
    )


def estimate_level(
    color_type,
    ratio
):
    """按颜色对应的面积阈值将能量球判为 1、2 或 3 级。"""

    if ratio <= 0:

        return 0


    thresholds = \
        LEVEL_THRESHOLDS[
            color_type
        ]


    if ratio < thresholds["12"]:

        return 1


    if ratio < thresholds["23"]:

        return 2


    return 3


def recognize_cell(
    cell,
    row,
    col,
    debug_dir
):
    """识别单个格子的颜色、等级和置信度。"""

    unknown_result = {
        "type": "?",
        "level": 0,
        "area_ratio": 0,
        "confidence": 0,
    }

    if cell is None or cell.size == 0:
        return unknown_result, {}

    center = \
        get_center_region(
            cell
        )

    if center.size == 0:
        return unknown_result, {}


    h, w = center.shape[:2]

    total_pixels = h * w


    (
        fire_mask,
        thunder_mask,
        wind_mask,
        background
    ) = make_color_masks(
        center
    )


    masks = {

        "f": fire_mask,

        "r": thunder_mask,

        "w": wind_mask
    }


    data = {}


    for color_type, mask in \
            masks.items():

        component, pixel_area = \
            get_best_component(
                mask
            )


        ratio = (
            pixel_area /
            total_pixels
        )


        data[
            color_type
        ] = {

            "mask": mask,

            "component":
                component,

            "pixel_area":
                pixel_area,

            "ratio":
                ratio
        }


    valid = {}


    for color_type in (
        "f",
        "r",
        "w"
    ):

        ratio = data[
            color_type
        ]["ratio"]


        if ratio >= \
                MIN_COLOR_RATIO:

            valid[
                color_type
            ] = ratio


    if not valid:

        result = {

            "type": "?",

            "level": 0,

            "area_ratio": 0,

            "confidence": 0
        }


        return (
            result,
            data
        )


    best_type = max(
        valid,
        key=valid.get
    )


    best_ratio = valid[
        best_type
    ]


    sorted_values = sorted(
        valid.values(),
        reverse=True
    )


    if len(
        sorted_values
    ) >= 2:

        second_ratio = \
            sorted_values[1]

    else:

        second_ratio = 0


    if best_ratio > 0:

        color_gap = (

            best_ratio -
            second_ratio
        ) / best_ratio

    else:

        color_gap = 0


    level = estimate_level(
        best_type,
        best_ratio
    )


    # 面积特别小时判定为1级
    if best_ratio < 0.08:

        level = 1


    thresholds = \
        LEVEL_THRESHOLDS[
            best_type
        ]


    if level == 1:

        target = \
            thresholds["12"] * 0.65


    elif level == 2:

        target = (

            thresholds["12"] +
            thresholds["23"]
        ) / 2


    else:

        target = (

            thresholds["23"] +
            thresholds["23"] * 0.45
        )


    distance = abs(
        best_ratio -
        target
    )


    level_confidence = max(
        0,
        1.0 -
        distance / 0.20
    )


    confidence = (

        color_gap * 0.50

        +

        level_confidence * 0.50
    )


    result = {

        "type": best_type,

        "level": level,

        "area_ratio": best_ratio,

        "confidence":
            confidence
    }


    # 保存原始掩码和候选区域，便于定位识别问题。
    if DEBUG_SAVE_IMAGES and debug_dir is not None:
        base = os.path.join(
            debug_dir,
            f"r{row + 1}c{col + 1}"
        )

        cv2.imwrite(
            base + "_center.png",
            center
        )

        cv2.imwrite(
            base + "_fire.png",
            fire_mask
        )

        cv2.imwrite(
            base + "_thunder.png",
            thunder_mask
        )

        cv2.imwrite(
            base + "_wind.png",
            wind_mask
        )

        cv2.imwrite(
            base + "_component.png",
            data[best_type]["component"]
        )


    # 生成带轮廓、颜色、等级和面积比例的识别图。
    debug_img = center.copy()


    contours, _ = cv2.findContours(
        data[
            best_type
        ]["component"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        debug_img,
        contours,
        -1,
        DRAW_COLOR[
            best_type
        ],
        2
    )


    label = f"{level}{best_type} {best_ratio:.3f}"


    cv2.putText(
        debug_img,
        label,
        (3, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


    if DEBUG_SAVE_IMAGES and debug_dir is not None:
        cv2.imwrite(
            base + "_result.png",
            debug_img
        )


    return (
        result,
        data
    )


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
    while AUTO_PAUSED and not AUTO_STOP_REQUESTED:
        time.sleep(0.10)


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


# 点击后根据棋盘画面变化自动等待动画结束，再进入下一轮识别。
AUTO_CLICK_ENABLED = True

# 两次点击之间的间隔（秒）。
AUTO_CLICK_DELAY = 0.12

# 鼠标点击后移到棋盘外，避免指针遮挡小球影响识别
MOUSE_MOVE_OUT_X = 10
MOUSE_MOVE_OUT_Y = 10

# 点击完成后，至少等待这么久再开始判断棋盘。
MIN_WAIT_AFTER_SWAP = 0.20

# 第一步通常会启动更长的首次合成动画，单独留出启动时间。
FIRST_STEP_MIN_WAIT_AFTER_SWAP = 0.80

# 初始随机出球后可能立即触发三连，单独留出合成动画的观察时间。
# 只在启动时使用一次，不影响后续交换后的快速稳定判断。
INITIAL_BOARD_STABLE_TIME = 0.80

# 检查棋盘的间隔（秒）。
BOARD_CHECK_INTERVAL = 0.08

# 棋盘识别状态连续稳定这么久后，认为动画结束。
BOARD_STABLE_TIME = 0.25

# 最多等待时间，避免游戏动画异常时无限等待。
MAX_WAIT_AFTER_SWAP = 3.00

# 自动运行最长时间（秒）
GAME_MAX_SECONDS = 1200.0


def get_cell_center_from_board(r, c, x1, y1, dx, dy):
    return (
        int(round(x1 + c * dx)),
        int(round(y1 + r * dy)),
    )


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
            cell = crop_cell(
                screenshot,
                cx,
                cy,
                cell_w,
                cell_h,
            )

            if cell is None:
                result = {
                    "type": "?",
                    "level": 0,
                    "area_ratio": 0,
                    "confidence": 0,
                }
            else:
                result, _ = recognize_cell(
                    cell,
                    row,
                    col,
                    None,
                )

            if result["type"] == "?":
                unknown_count += 1

            board_row.append(result)

        board.append(board_row)

    if unknown_count > 0:
        return board, unknown_count, None

    board_state = make_board_state(board)
    return board, 0, board_state


def save_board_debug_images(
    screenshot,
    x1,
    y1,
    dx,
    dy,
    debug_dir,
    step,
):
    """保存一次已完成交换对应的棋盘和逐格 DEBUG 图片。"""

    os.makedirs(
        BOARD_DEBUG_DIR,
        exist_ok=True,
    )

    cv2.imwrite(
        os.path.join(
            debug_dir,
            "board.png"
        ),
        screenshot
    )

    # 独立目录按成功交换步数保存；step 目录保存对应的历史记录。
    cv2.imwrite(
        os.path.join(
            BOARD_DEBUG_DIR,
            f"board_{step:02d}.png",
        ),
        screenshot,
    )

    cell_w = abs(dx) * 0.90
    cell_h = abs(dy) * 0.90

    for row in range(ROWS):
        for col in range(COLS):
            cx = x1 + col * dx
            cy = y1 + row * dy

            cell = crop_cell(
                screenshot,
                cx,
                cy,
                cell_w,
                cell_h
            )

            if cell is not None:
                recognize_cell(
                    cell,
                    row,
                    col,
                    debug_dir
                )


def wait_for_initial_board_ready(sct, monitor, x1, y1, dx, dy):
    """等待 27 个球全部识别成功且棋盘状态稳定后返回截图。"""

    print("等待能量球出现...", end="", flush=True)

    capture_x1, capture_y1 = get_capture_board_origin(
        x1,
        y1,
        monitor,
    )

    previous_state = None
    stable_start = None
    refined_geometry = None

    while True:
        if AUTO_STOP_REQUESTED:
            return None

        if RESELECT_REQUESTED.is_set():
            return RESELECT_RESULT

        wait_while_paused()

        if AUTO_STOP_REQUESTED:
            return None

        raw = np.array(
            sct.grab(
                monitor
            )
        )

        screenshot = cv2.cvtColor(
            raw,
            cv2.COLOR_BGRA2BGR
        )

        _, unknown_count, current_state = (
            recognize_board_snapshot(
                screenshot,
                capture_x1,
                capture_y1,
                dx,
                dy,
            )
        )

        if unknown_count == 0 and refined_geometry is None:
            refined_geometry = refine_grid_geometry(
                screenshot,
                capture_x1,
                capture_y1,
                dx,
                dy,
            )

            if refined_geometry is not None:
                capture_x1, capture_y1, dx, dy = refined_geometry
                _, unknown_count, current_state = (
                    recognize_board_snapshot(
                        screenshot,
                        capture_x1,
                        capture_y1,
                        dx,
                        dy,
                    )
                )

        if unknown_count > 0:
            stable_start = None
            previous_state = None
            print(
                f"\r等待能量球出现... 已识别 "
                f"{ROWS * COLS - unknown_count}/{ROWS * COLS} 个",
                end="",
                flush=True
            )
        elif current_state != previous_state:
            previous_state = current_state
            stable_start = time.monotonic()
        elif (
            time.monotonic() - stable_start
            >= INITIAL_BOARD_STABLE_TIME
        ):
            print(
                "\r已识别到完整棋盘（27个能量球）。        "
            )
            return (
                raw,
                capture_x1,
                capture_y1,
                dx,
                dy,
            )

        time.sleep(BOARD_CHECK_INTERVAL)


def wait_for_board_stable(
    sct,
    monitor,
    x1,
    y1,
    dx,
    dy,
    minimum_wait=None,
):
    """
    交换完成后等待棋盘稳定。

    只有在完整识别到 27 格且棋盘状态连续不变时返回。
    """

    if minimum_wait is None:
        minimum_wait = MIN_WAIT_AFTER_SWAP

    time.sleep(minimum_wait)

    capture_x1, capture_y1 = get_capture_board_origin(
        x1,
        y1,
        monitor,
    )

    start_time = time.monotonic()

    previous_state = None
    stable_start = None

    while True:

        if AUTO_STOP_REQUESTED:
            print("\r收到停止指令，停止等待棋盘稳定。")
            return time.monotonic() - start_time

        if RESELECT_REQUESTED.is_set():
            return RESELECT_RESULT

        elapsed = (
            time.monotonic()
            -
            start_time
        )

        if elapsed >= MAX_WAIT_AFTER_SWAP:

            print(
                f"  棋盘等待达到上限 "
                f"{MAX_WAIT_AFTER_SWAP:.2f}s，"
                f"强制进入下一轮。"
            )

            return elapsed


        time.sleep(BOARD_CHECK_INTERVAL)

        raw = np.array(
            sct.grab(
                monitor
            )
        )
        screenshot = cv2.cvtColor(
            raw,
            cv2.COLOR_BGRA2BGR
        )

        _, unknown_count, current_state = (
            recognize_board_snapshot(
                screenshot,
                capture_x1,
                capture_y1,
                dx,
                dy,
            )
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
            print(
                "\r棋盘状态发生变化，重新等待稳定...     ",
                end="",
                flush=True,
            )
            continue

        stable_elapsed = time.monotonic() - stable_start
        print(
            f"\r检测棋盘状态稳定：{stable_elapsed:.2f}s",
            end="",
            flush=True,
        )

        if stable_elapsed < BOARD_STABLE_TIME:
            continue

        print(
            f"\r棋盘已稳定，等待 {elapsed:.2f}s。              "
        )
        return elapsed


def click_best_swap(
    results,
    x1,
    y1,
    dx,
    dy,
    sct,
    monitor,
    step=1,
):
    """
    执行当前一步最优交换，然后等待棋盘稳定。

    返回：
        True  = 成功执行
        False = 没有可执行交换
    """

    if AUTO_STOP_REQUESTED:
        return False

    if not results:

        print()
        print("=" * 90)
        print("自动操作")
        print("=" * 90)
        print("没有可以立即合成的交换，本轮停止。")

        return False


    if not AUTO_CLICK_ENABLED:

        print(
            "自动点击已关闭。"
        )

        return False


    if pyautogui is None:

        print(
            "错误：未安装 pyautogui。"
        )

        print(
            "python -m pip install pyautogui"
        )

        return False


    best = results[0]

    p1 = best["p1"]
    p2 = best["p2"]


    x1_click, y1_click = \
        get_cell_center_from_board(
            *p1,
            x1,
            y1,
            dx,
            dy
        )


    x2_click, y2_click = \
        get_cell_center_from_board(
            *p2,
            x1,
            y1,
            dx,
            dy
        )


    # 执行交换。鼠标移到屏幕角落时保留 PyAutoGUI 的紧急停止保护。
    try:
        pyautogui.click(
            x1_click,
            y1_click
        )

        time.sleep(
            AUTO_CLICK_DELAY
        )

        if AUTO_STOP_REQUESTED:
            return False

        pyautogui.click(
            x2_click,
            y2_click
        )

        # 点击完成后立即移出棋盘，避免鼠标指针遮挡小球
        pyautogui.moveTo(
            MOUSE_MOVE_OUT_X,
            MOUSE_MOVE_OUT_Y,
            duration=0.05
        )
    except pyautogui.FailSafeException:
        request_auto_stop(
            "检测到鼠标位于屏幕角落，已安全停止自动点击"
        )
        return False


    print()
    print(
        "交换已执行，等待棋盘稳定..."
    )


    stable_result = wait_for_board_stable(
        sct,
        monitor,
        x1,
        y1,
        dx,
        dy,
        minimum_wait=(
            FIRST_STEP_MIN_WAIT_AFTER_SWAP
            if step == 1
            else MIN_WAIT_AFTER_SWAP
        ),
    )

    if stable_result is RESELECT_RESULT:
        return RESELECT_RESULT

    return True


def run_auto_loop(
    x1,
    y1,
    dx,
    dy
):
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

    不预测未来连锁。
    """

    loop_start = time.monotonic()

    move_count = 0
    debug_step_count = 0
    last_unknown_count = None
    waiting_for_board = False

    setup_hotkeys()

    with mss.MSS() as sct:

        full_monitor = sct.monitors[0]
        monitor = get_board_capture_monitor(
            x1,
            y1,
            dx,
            dy,
            full_monitor,
        )
        capture_x1, capture_y1 = get_capture_board_origin(
            x1,
            y1,
            monitor,
        )


        while True:

            if AUTO_STOP_REQUESTED:
                break

            if RESELECT_REQUESTED.is_set():
                RESELECT_REQUESTED.clear()
                calibration = calibrate(
                    wait_for_enter=False
                )

                if calibration is None:
                    break

                x1, y1, dx, dy = calibration
                monitor = get_board_capture_monitor(
                    x1,
                    y1,
                    dx,
                    dy,
                    full_monitor,
                )
                capture_x1, capture_y1 = get_capture_board_origin(
                    x1,
                    y1,
                    monitor,
                )
                last_unknown_count = None
                waiting_for_board = False
                print("已更新棋盘框选区域，继续自动运行。")
                continue

            wait_while_paused()

            if AUTO_STOP_REQUESTED:
                break

            # 防止异常情况下无限运行。

            elapsed = (
                time.monotonic()
                -
                loop_start
            )


            if elapsed >= \
                    GAME_MAX_SECONDS:

                print()
                print(
                    "=" * 90
                )

                print(
                    "自动运行时间达到 "
                    f"{GAME_MAX_SECONDS:.1f}s，停止。"
                )

                break


            move_count += 1
            if not waiting_for_board:
                print()
                print(
                    "=" * 90
                )

                print(
                    f"自动第 "
                    f"{move_count} 步"
                )


            # 截取当前棋盘。


            raw = np.array(
                sct.grab(
                    monitor
                )
            )


            screenshot = \
                cv2.cvtColor(
                    raw,
                    cv2.COLOR_BGRA2BGR
                )

            # 识别 3×9 个格子。

            cell_w = abs(dx) * 0.90
            cell_h = abs(dy) * 0.90


            board = []


            for row in range(ROWS):

                board_row = []


                for col in range(COLS):

                    cx = (
                        capture_x1 +
                        col * dx
                    )

                    cy = (
                        capture_y1 +
                        row * dy
                    )


                    cell = crop_cell(
                        screenshot,
                        cx,
                        cy,
                        cell_w,
                        cell_h
                    )


                    if cell is None:

                        result = {
                            "type": "?",
                            "level": 0,
                            "area_ratio": 0,
                            "confidence": 0,
                        }

                    else:

                        result, _ = \
                            recognize_cell(
                                cell,
                                row,
                                col,
                                None
                            )


                    board_row.append(
                        result
                    )


                board.append(
                    board_row
                )


            # 有未知格子时只等待，不执行点击。

            unknown_count = sum(
                1
                for row in board
                for result in row
                if result["type"] == "?"
            )


            if unknown_count > 0:

                # 20 步完成后只要仍有未识别格，通常表示游戏已结束。
                # 使用与 F9 相同的停止流程，不再继续等待或误执行下一步。
                if debug_step_count >= TOTAL_GAME_MOVES:
                    move_count -= 1
                    request_auto_stop(
                        "已完成 20 步，棋盘仍有未识别能量球，自动执行 F9"
                    )
                    break

                if unknown_count != last_unknown_count:
                    print(
                        f"\r本轮未识别到{unknown_count}个能量球，等待中..."
                        "          ",
                        end="",
                        flush=True
                    )

                last_unknown_count = unknown_count
                waiting_for_board = True

                move_count -= 1

                time.sleep(
                    BOARD_CHECK_INTERVAL
                )

                continue

            if waiting_for_board:
                print(
                    "\r" + " " * 60 + "\r",
                    end="",
                    flush=True
                )

            last_unknown_count = None
            waiting_for_board = False


            # 计算当前最优交换。

            board_state = \
                make_board_state(
                    board
                )


            print_board_state(
                board_state
            )


            try:
                results = \
                    analyze_all_swaps(
                        board_state,
                        current_step=move_count
                    )
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

            save_swap_results(
                results,
                move_count,
            )


            if not results:

                # 没有立即可合成交换时暂停，等待用户处理。
                pause_for_no_move()

                if AUTO_STOP_REQUESTED:
                    break

                wait_while_paused()

                # 恢复后重新识别；本轮不计入实际交换步数。
                move_count -= 1

                continue


            print_best_swap(
                results
            )


            # 执行选中的交换。

            swap_completed = click_best_swap(
                results,
                x1,
                y1,
                dx,
                dy,
                sct,
                monitor,
                step=move_count,
            )

            if swap_completed is RESELECT_RESULT:
                continue

            if swap_completed:
                debug_step_count += 1

                if DEBUG_SAVE_IMAGES:
                    debug_step_dir = get_debug_step_dir(
                        debug_step_count
                    )

                    enqueue_debug_images(
                        screenshot,
                        capture_x1,
                        capture_y1,
                        dx,
                        dy,
                        debug_step_dir,
                        debug_step_count,
                    )


    print()
    print(
        "=" * 90
    )

    print(
        "自动运行结束"
    )

    print(
        f"完成交换：{move_count}"
    )
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
    return (
        a is not None and
        b is not None and
        a[0] == b[0] and
        a[1] == b[1]
    )


def mergeable(ball):
    # 三级球：可以交换，但不能进入任何合成候选。
    return ball is not None and ball[1] < 3


def empty_cells(board):
    return [(r, c) for r in range(ROWS) for c in range(COLS)
            if board[r][c] is None]

def copy_board(board):
    return [row.copy() for row in board]


def swap_cells(board, p1, p2):
    new_board = copy_board(board)
    r1, c1 = p1
    r2, c2 = p2
    new_board[r1][c1], new_board[r2][c2] = (
        new_board[r2][c2],
        new_board[r1][c1],
    )
    return new_board


def find_horizontal_matches(board):
    matches = []

    for r in range(ROWS):
        c = 0
        while c < COLS:
            ball = board[r][c]

            if not mergeable(ball):
                c += 1
                continue

            start = c
            c += 1

            while c < COLS and same_ball(ball, board[r][c]):
                c += 1

            if c - start >= 3:
                matches.append([(r, x) for x in range(start, c)])

    return matches


def find_vertical_matches(board):
    matches = []

    for c in range(COLS):
        r = 0
        while r < ROWS:
            ball = board[r][c]

            if not mergeable(ball):
                r += 1
                continue

            start = r
            r += 1

            while r < ROWS and same_ball(ball, board[r][c]):
                r += 1

            if r - start >= 3:
                matches.append([(x, c) for x in range(start, r)])

    return matches


def find_matches(board):
    return (
        find_horizontal_matches(board)
        + find_vertical_matches(board)
    )


# 搜索、评分与合成模拟
# 重要：
#   下面的规则以已确认的真实游戏行为为准。
#   1. 3×9 棋盘，无下落。
#   2. 交换可以是任意两格：C(27,2)=351。
#   3. 候选组合按“最下方 -> 最左侧”比较整个组合。
#   3.1 三级球可以参与任意交换，但不能参与合成。
#   4. 同一轮可以同时处理互不冲突的组合。
#   5. 长横连按游戏规则拆分：
#        6 个一级同球 -> 2级、空、空、2级、空、空
#        7 个一级同球 -> 2级、空、空、2级、空、空、1级
#      即每 3 个一组，剩余球保留。
#   6. 3级不能继续合成。
#   7. 合成产生的空位原地补一级球：
#        火 / 雷 / 风，各 1/3。
#   8. 补球后重新扫描；如果出现多个组合，继续使用同一
#      全局优先级。
# 搜索流程：
#   枚举 351 种交换 -> 按即时收益筛选 -> 采样评估后续收益。
# 合成模拟决定真实收益；评分只用于在候选交换中排序。

BALL_ENERGY = {
    "f": {1: 5.0, 2: 20.0, 3: 80.0},
    "r": {1: 4.0, 2: 16.0, 3: 64.0},
    "w": {1: 3.0, 2: 12.0, 3: 48.0},
}

TYPE_ORDER = ("f", "r", "w")

# 评分参数集中在此，便于统一调整。
SCORE_WEIGHTS = {
    # 高等级球保留价值
    "inventory": 0.08,

    # 当前棋盘存在两个同球相邻时的潜力
    "pair": 1.5,

    # 空位很少时略微提高“可继续操作”的价值
    "mobility": 0.25,
}

# 让随机前瞻使用同一批种子比较候选，降低候选间的随机采样误差。
LOOKAHEAD_SEED_BASE = 0x5A17C9

# 基础棋盘操作

def match_anchor(group):
    """游戏全局排序：最下方优先，其次最左侧。"""
    return min(
        group,
        key=lambda p: (-p[0], p[1])
    )


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
        ordered[i:i + 3]
        for i in range(0, len(ordered), 3)
        if len(ordered[i:i + 3]) >= 3
    ]


def expand_matches(matches):
    expanded = []

    for group in matches:
        if len(group) <= 3:
            expanded.append(group)
        else:
            expanded.extend(split_long_group(group))

    return expanded


def apply_merge_groups(board, groups):
    """
    同一轮执行所有互不冲突的3球组合。
    每组的生成位置为该组的优先位置。
    """
    state = copy_board(board)
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

        for r, c in group:
            state[r][c] = None

        state[anchor[0]][anchor[1]] = (ball_type, new_level)

        gain += BALL_ENERGY[ball_type].get(new_level, 0.0)
        merge_count += 1

    return state, merge_count, gain


def resolve_merges(board, max_rounds=20):
    """
    严格模拟游戏的连续合成：
        扫描 -> 按全局优先级选本轮不冲突组合
        -> 同轮执行 -> 补一级球 -> 再扫描

    没有任何“下落”逻辑。
    """
    state = copy_board(board)
    total_gain = 0.0
    total_merges = 0
    rounds = 0

    for _ in range(max_rounds):
        matches = find_matches(state)

        if not matches:
            break

        expanded = expand_matches(matches)
        selected = choose_merge_groups(expanded)

        if not selected:
            break

        state, merge_count, gain = apply_merge_groups(
            state,
            selected
        )

        if merge_count <= 0:
            break

        total_merges += merge_count
        total_gain += gain
        rounds += 1

        # 空位暂时保留，由 resolve_with_refill 统一补球。

        if rounds >= max_rounds:
            break

    return state, total_merges, total_gain


def refill_random(board, rng):
    state = copy_board(board)

    # 分层 RNG 需要在每轮补球前重置位置编号；普通 RNG 不需要。
    begin_refill = getattr(rng, "begin_refill", None)
    if begin_refill is not None:
        begin_refill()

    for r, c in empty_cells(state):
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
    """
    state = copy_board(board)
    total_gain = 0.0
    total_merges = 0
    rounds_used = 0

    while rounds_used < max_rounds:
        # 先把当前状态中所有无需补球即可继续发生的合成全部做完。
        stable_state, merge_count, gain = resolve_merges(
            state,
            max_rounds=max_rounds - rounds_used,
        )

        state = stable_state
        total_gain += float(gain)
        total_merges += int(merge_count)

        if merge_count > 0:
            rounds_used += 1

        # 没有空位，且当前已经稳定 -> 连锁结束。
        if not empty_cells(state):
            if not find_matches(state):
                break
            continue

        # 只有确定性合成阶段结束后才进行随机补球。
        state = refill_random(state, rng)

        # 补球后下一轮继续先处理全部确定性合成。
        rounds_used += 1

    return state, total_merges, total_gain


def level3_position_quality(board):
    """
    三级球可以交换，所以不做“永久阻塞”惩罚。
    这里只计算一个很小的局面项：
      - 周围同属性一级/二级球越多，说明三级球当前占据的位置越可能
        需要被挪动；
      - 边缘位置天然比中间位置更容易处理。
     该项只作为次级排序的小修正，不改变真实能量价值。
    """
    score = 0.0

    for r in range(ROWS):
        for c in range(COLS):
            ball = board[r][c]
            if ball is None or ball[1] != 3:
                continue

            nearby = 0
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ROWS and 0 <= cc < COLS:
                    other = board[rr][cc]
                    if other is not None and other[0] == ball[0] and other[1] < 3:
                        nearby += 1

            # 仅为很小的位置修正；三级球仍保留完整真实能量。
            score -= nearby * 0.5

    return score


def board_potential(board, current_step=None):
    """
    只用于评分，不改变真实合成规则。
    数值尽量保持在真实能量同量纲附近。
    """
    value = 0.0

    # 保留高级球的价值。
    for row in board:
        for ball in row:
            if ball is None:
                continue

            value += BALL_ENERGY[ball[0]].get(ball[1], 0.0) * SCORE_WEIGHTS["inventory"]

    # 两球潜力。
    pair_count = 0

    for r in range(ROWS):
        for c in range(COLS):
            ball = board[r][c]
            if ball is None:
                continue

            if c + 1 < COLS and same_ball(ball, board[r][c + 1]):
                pair_count += 1

            if r + 1 < ROWS and same_ball(ball, board[r + 1][c]):
                pair_count += 1

    value += pair_count * SCORE_WEIGHTS["pair"]


    # 可交换性：空位越多，后续随机状态通常越容易出现有效交换。
    empty = len(empty_cells(board))
    value += empty * SCORE_WEIGHTS["mobility"]

    # 三级球是“可交换但不可合成”的受限球，不做永久占位惩罚。
    # 三级球只属于 board_potential 的辅助惩罚。
    # 从第1步开始线性衰减，第15步归零，第15步以后保持0。
    level3_penalty = level3_position_quality(board)
    if current_step is None:
        level3_weight = 1.0
    else:
        level3_weight = max(
            0.0,
            min(1.0, (15.0 - float(current_step)) / 14.0)
        )
    value += level3_penalty * level3_weight

    return value


def evaluate_swap(board, p1, p2):
    """
    交换评估：

    immediate_gain = 交换后、第一次随机补球之前，所有连续确定性合成
                     的累计真实收益。

    current_chain_gain = 第一次随机补球之后才产生的真实连锁收益期望。

    关键边界：
        交换
        -> 确定性合成第1轮
        -> 若升级球立即形成新三连，继续确定性合成
        -> 直到稳定/出现空位
        -> 此时才开始随机补球
    """
    a = board[p1[0]][p1[1]]
    b = board[p2[0]][p2[1]]

    if a is None or b is None:
        return None

    if same_ball(a, b):
        return None

    simulated = swap_cells(board, p1, p2)
    matches = find_matches(simulated)

    if not matches:
        return make_candidate(
            p1=p1,
            p2=p2,
            board=simulated,
            merge_gain=0.0,
            merge_count=0,
        )

    # 交换后先把第一次补球之前的确定性连续合成全部处理完。
    resolved, merge_count, merge_gain = resolve_merges(
        simulated,
        max_rounds=20,
    )

    return make_candidate(
        p1=p1,
        p2=p2,
        board=resolved,
        merge_gain=merge_gain,
        merge_count=merge_count,
    )


def generate_all_swaps():
    swaps = []

    for r1 in range(ROWS):
        for c1 in range(COLS):
            p1 = (r1, c1)

            for r2 in range(r1, ROWS):
                start_c = c1 + 1 if r2 == r1 else 0

                for c2 in range(start_c, COLS):
                    swaps.append((p1, (r2, c2)))

    return swaps



CANDIDATE_FIELDS = (
    "p1", "p2", "board",
    "immediate_gain",
    "future_gain",
    "chain_gain",
    "potential",
    "potential_bonus",
    "real_score",
    "final_score",
    "quick_score",
)




def _board_cache_key(board):
    """
    将棋盘规范化成不可变 tuple，避免直接使用 list 作为 dict key。
    仅用于缓存，不参与评分。
    """
    rows = []
    for row in board:
        rows.append(
            tuple(
                tuple(cell) if isinstance(cell, (list, tuple))
                else cell
                for cell in row
            )
        )
    return tuple(rows)


def _unique_swap_candidates(candidates):
    """
    保持候选第一次出现的顺序，以 (p1,p2) 为唯一键。
    不改变候选评分，也不改变交换合法性。
    """
    result = []
    seen = set()

    for candidate in candidates:
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")

        if p1 is None or p2 is None:
            result.append(candidate)
            continue

        key = (tuple(p1), tuple(p2))
        reverse_key = (tuple(p2), tuple(p1))

        if key in seen or reverse_key in seen:
            continue

        seen.add(key)
        result.append(candidate)

    return result


def make_candidate(
    *,
    p1, p2, board,
    merge_gain=0.0, merge_count=0,
    quick_score=None,
    immediate_gain=None, future_gain=0.0, chain_gain=0.0,
    potential=0.0, potential_bonus=0.0,
    real_score=None, final_score=None,
):
    """构造统一的交换候选结果。"""
    immediate = float(
        merge_gain if immediate_gain is None else immediate_gain
    )
    quick = (
        immediate
        if quick_score is None
        else float(quick_score)
    )
    potential = float(potential or 0.0)
    potential_bonus = float(potential_bonus or 0.0)
    future = float(future_gain)
    chain = float(chain_gain)

    if real_score is None:
        real_score = immediate + future

    if final_score is None:
        final_score = real_score

    return {
        "p1": p1,
        "p2": p2,
        "board": board,
        "immediate_gain": immediate,
        "future_gain": future,
        "chain_gain": chain,
        "potential": potential,
        "potential_bonus": potential_bonus,
        "real_score": float(real_score),
        "final_score": float(final_score),
        "quick_score": quick,

        # 与核心字段同步的派生字段
        "merge_gain": immediate,
        "merge_count": int(merge_count or 0),
        "score": float(final_score),
        "first_score": immediate,
        "second_expected": future,
        "chain_value": chain,
        "future_potential": potential,
        "future_score": float(final_score),
    }


def candidate_from_result(result):
    return make_candidate(
        p1=result.get("p1"),
        p2=result.get("p2"),
        board=result.get("board"),
        merge_gain=result.get(
            "merge_gain",
            result.get("immediate_gain", 0.0),
        ),
        merge_count=result.get("merge_count", 0),
        immediate_gain=result.get("immediate_gain"),
        future_gain=result.get(
            "future_gain",
            result.get("second_expected", 0.0),
        ),
        chain_gain=result.get(
            "chain_gain",
            result.get("chain_value", 0.0),
        ),
        potential=result.get(
            "potential",
            result.get("future_potential", 0.0),
        ),
        potential_bonus=result.get(
            "potential_bonus", 0.0
        ),
        real_score=result.get("real_score"),
        final_score=result.get(
            "final_score",
            result.get("score"),
        ),
        quick_score=result.get(
            "quick_score",
            result.get(
                "_quick_score",
                result.get("score", 0.0),
            ),
        ),
    )


def assert_candidate(candidate):
    missing = [
        field
        for field in CANDIDATE_FIELDS
        if field not in candidate
    ]
    if missing:
        raise KeyError(
            f"候选结果缺少统一字段: {missing}"
        )
    return candidate


# 搜索配置。
# 减少后续随机模拟数量，在速度和搜索范围之间取平衡。
FAST_FILTER_LIMIT = 12
# 只并行随机评估；截图、识别和点击始终由主线程执行。
EVALUATION_WORKERS = 8
EVALUATION_CACHE_LIMIT = 30000

# 游戏通常进行 20 步。这里只控制评分是否继续看下一步，
# 不是运行步数上限，因此偶发执行到第 21 步时仍然可以继续运行。
TOTAL_GAME_MOVES = 20

# 下一步交换候选

def _future_legal_swaps(board):
    """
    枚举所有合法下一步交换。
    三级球可以交换；不要求相邻；相同球交换无意义。
    """
    for p1, p2 in generate_all_swaps():
        a = board[p1[0]][p1[1]]
        b = board[p2[0]][p2[1]]

        if a is None or b is None:
            continue

        if same_ball(a, b):
            continue

        yield p1, p2





@dataclass(frozen=True)
class SamplingPlan:
    """
    一次评估使用的随机种子集合。

    所有采样与评估函数都接收该对象，避免混用种子列表和样本数量。
    """
    pass_index: int
    seeds: tuple

    @property
    def count(self):
        return len(self.seeds)


def make_sampling_plan(pass_index, count):
    """
    创建固定样本数的随机种子计划。
    """
    pass_index = int(pass_index)
    count = int(count)

    if count <= 0 or count % 3 != 0:
        raise ValueError(
            "样本数量必须是大于 0 的 3 的倍数"
        )

    seeds = tuple(
        (
            LOOKAHEAD_SEED_BASE
            + pass_index * 1000003
            + i * 9176
        ) & ((1 << 63) - 1)
        for i in range(count)
    )

    return SamplingPlan(
        pass_index=pass_index,
        seeds=seeds,
    )



def make_evaluation_plan():
    """创建当前搜索使用的统一采样计划。"""
    return make_sampling_plan(2, FINAL_SAMPLE_COUNT)




def _balanced_type_for_sample(
    sample_index,
    empty_index,
    sample_plan,
):
    """
    对每个补球位置进行独立且均衡的分层采样。

    关键修正：
        不再使用：
            (sample_index + empty_index) % 3

        因为这种写法会把不同空位“绑死”成相关变量。
        例如 9 个样本、两个空位时，只会得到：
            (f,r), (r,w), (w,f), ...
        而不会得到 3×3 的全部组合。

    现在把每一个补球位置视为一个独立随机变量。

    对 9 个样本：
        第 0 个空位：每种颜色 3 次
        第 1 个空位：每种颜色 3 次
        并且两个变量的 3×3 组合各出现一次。

    对超过 2 个变量的情况，使用有限样本下的循环平衡设计。
    """
    if not isinstance(sample_plan, SamplingPlan):
        raise TypeError("sample_plan 必须是 SamplingPlan")

    count = sample_plan.count
    if count <= 0 or count % 3 != 0:
        raise ValueError(
            f"样本数量必须是大于 0 的 3 的倍数：{count}"
        )

    sample_index = int(sample_index)
    empty_index = int(empty_index)

    if not 0 <= sample_index < count:
        raise IndexError("样本索引超出采样计划范围")

    if empty_index < 0:
        raise IndexError("空位索引必须大于等于 0")

    colors = ("f", "r", "w")

    # 9 个样本时，前两个补球位置完整覆盖 3×3 种颜色组合。
    if count == 9:
        a = sample_index // 3
        b = sample_index % 3

        if empty_index == 0:
            digit = a
        elif empty_index == 1:
            digit = b
        else:
            # 后续位置使用拉丁方设计，保持单个位置的颜色分布均衡。
            digit = (a + b) % 3
            if empty_index >= 3:
                digit = (a + (empty_index - 1) * b) % 3

        return colors[digit]

    if count == 6:
        # 6 个样本无法覆盖全部 9 种组合，但每个位置仍各出现 2 次。
        block = sample_index // 3
        pos = sample_index % 3
        digit = (pos + block * empty_index) % 3
        return colors[digit]

    # 其他样本数使用独立的均衡循环。
    return colors[
        (sample_index + empty_index * (sample_index // 3 + 1)) % 3
    ]




def _validate_sampling_plan_balance(
    board,
    sample_plan,
):
    """
    验证每个空位的边缘分布严格为 1/3。
    """
    if not isinstance(sample_plan, SamplingPlan):
        raise TypeError("sample_plan 必须是 SamplingPlan")

    expected = sample_plan.count // 3
    report = []

    for empty_index, (r, c) in enumerate(empty_cells(board)):
        counts = {"f": 0, "r": 0, "w": 0}

        for sample_index in range(sample_plan.count):
            ball_type = _balanced_type_for_sample(
                sample_index,
                empty_index,
                sample_plan,
            )
            if ball_type not in counts:
                raise AssertionError(
                    f"无效的能量球类型：{ball_type!r}"
                )
            counts[ball_type] += 1

        if counts != {
            "f": expected,
            "r": expected,
            "w": expected,
        }:
            raise AssertionError(
                f"位置 {(r, c)} 的采样分布不均衡："
                f"{counts}；样本数量={sample_plan.count}"
            )

        report.append((r, c, counts))

    return report



class _BalancedRefillRNG:
    """
    用于分层随机补球的 RNG。

    resolve_with_refill() 本身不修改；它仍然调用：
        refill_random(state, rng)
    而 refill_random 仍然调用：
        rng.choice(TYPE_ORDER)

    这里仅替换 choice() 的取样方式：
    对每一个独立的补球位置，跨全部样本
    严格 f/r/w 各占 1/3。

    sample_index 决定同一 draw 在哪个颜色层；
    draw_index 在同一个 sample 内递增。
    seed 仅作为非补球随机操作的后备 RNG。
    """
    __slots__ = (
        "sample_index",
        "draw_index",
        "fallback",
        "sample_count",
    )

    def __init__(self, seed, sample_index, sample_count=9):
        self.sample_index = int(sample_index)
        self.draw_index = 0
        self.fallback = random.Random(seed)
        self.sample_count = int(sample_count)

    def begin_refill(self):
        """
        每一次“补球事件”都从新的独立变量序号 0 开始。

        这非常重要：
        一次补球有两个空位时，它们应该是两个独立随机变量；
        下一次因为连锁又产生空位时，则重新建立新的补球变量组。
        """
        self.draw_index = 0

    def choice(self, seq):
        """为补球位置提供均衡颜色，其他序列交给后备 RNG。"""
        try:
            values = tuple(seq)
        except TypeError:
            values = ()

        if set(values) == {"f", "r", "w"} and len(values) == 3:
            colors = ("f", "r", "w")

            count = getattr(self, "sample_count", 9)

            if count == 9:
                a = self.sample_index // 3
                b = self.sample_index % 3

                if self.draw_index == 0:
                    digit = a
                elif self.draw_index == 1:
                    digit = b
                else:
                    digit = (
                        a + self.draw_index * b
                    ) % 3
            elif count == 6:
                block = self.sample_index // 3
                pos = self.sample_index % 3
                digit = (
                    pos + block * self.draw_index
                ) % 3
            else:
                digit = (
                    self.sample_index
                    + self.draw_index
                ) % 3

            self.draw_index += 1
            return colors[digit]

        return self.fallback.choice(seq)


def _simulate_refill_chain_once(
    board_after_current_merge,
    sample_index,
    sample_plan,
):
    """
    对当前交换后的补球连锁执行一次随机样本模拟。

    传入的棋盘已经完成第一次补球前的确定性合成；这里只统计第一次
    随机补球之后产生的收益。
    """
    if not isinstance(sample_plan, SamplingPlan):
        raise TypeError("sample_plan 必须是 SamplingPlan")

    rng = _BalancedRefillRNG(
        sample_plan.seeds[sample_index],
        sample_index,
    )
    rng.sample_count = sample_plan.count

    state = copy_board(board_after_current_merge)

    if empty_cells(state):
        state = refill_random(state, rng)

    state, _, chain_gain = resolve_with_refill(
        state,
        rng,
        max_rounds=20,
    )

    return state, float(chain_gain)




def _simulate_next_swap_on_sample(
    board,
    p1,
    p2,
    sample_index,
    sample_plan,
):
    """
    对一个具体的下一步交换执行一次随机补球模拟。

    ``next_gain`` 是交换后、第一次补球前的确定性合成收益；
    ``next_chain_gain`` 是第一次补球及其后续连锁产生的收益。
    评估只使用固定数量的样本，不展开指数级概率树。
    """
    if not isinstance(sample_plan, SamplingPlan):
        raise TypeError("sample_plan 必须是 SamplingPlan")

    a = board[p1[0]][p1[1]]
    b = board[p2[0]][p2[1]]

    if a is None or b is None or same_ball(a, b):
        return None

    swapped = swap_cells(board, p1, p2)

    if not find_matches(swapped):
        return None

    # 第一次补球前的确定性连锁计入 next_gain。
    next_state, merge_count, next_gain = resolve_merges(
        swapped,
        max_rounds=20,
    )

    if merge_count <= 0:
        return None

    # 第一次补球是收益边界，之后的连锁计入 next_chain_gain。
    if empty_cells(next_state):
        rng = _BalancedRefillRNG(
            sample_plan.seeds[sample_index],
            sample_index,
            sample_plan.count,
        )
        next_state = refill_random(
            next_state,
            rng,
        )
        final_board, _, next_chain_gain = resolve_with_refill(
            next_state,
            rng,
            max_rounds=20,
        )
    else:
        final_board = next_state
        next_chain_gain = 0.0

    return {
        "next_gain": float(next_gain),
        "next_chain_gain": float(next_chain_gain),
        "future_gain": float(
            next_gain + next_chain_gain
        ),
        "final_board": final_board,
        "potential": float(
            board_potential(final_board, 3)
        ),
    }



def _evaluate_candidate_samples(
    board_after_current_merge,
    sample_plan,
    current_step,
):
    """
    评估当前连锁和下一步交换的随机期望。

    先生成每个样本的当前连锁结果，再比较每个具体下一步交换
    在全部样本上的平均收益。
    """
    if not isinstance(sample_plan, SamplingPlan):
            raise TypeError("sample_plan 必须是 SamplingPlan")

    # 验证每个补球位置的颜色分布保持均衡。
    balance_report = _validate_sampling_plan_balance(
        board_after_current_merge,
        sample_plan,
    )

    samples = []
    current_chain_values = []

    for sample_index in range(sample_plan.count):
        state, chain_gain = _simulate_refill_chain_once(
            board_after_current_merge,
            sample_index,
            sample_plan,
        )

        samples.append({
            "sample_index": sample_index,
            "seed": sample_plan.seeds[sample_index],
            "state": state,
            "current_chain_gain": float(chain_gain),
        })

        current_chain_values.append(
            float(chain_gain)
        )

    current_chain_expected = statistics.fmean(
        current_chain_values
    )

    # 第 20 步已经是最后一步：保留当前交换及其连锁收益，
    # 不再模拟下一次补球和下一步交换，避免把不存在的收益计入评分。
    if current_step >= TOTAL_GAME_MOVES:
        return {
            "current_chain_gain": float(current_chain_expected),
            "current_chain_samples": current_chain_values,
            "current_sample_count": sample_plan.count,
            "sample_plan": sample_plan,
            "balance_report": balance_report,
            "next_gain": 0.0,
            "next_chain_gain": 0.0,
            "future_gain": 0.0,
            "potential": 0.0,
            "next_p1": None,
            "next_p2": None,
            "next_gain_samples": [],
            "next_chain_samples": [],
            "next_legal_count": 0,
            "lookahead_enabled": False,
        }

    # 对同一个具体交换，在全部样本上计算“交换后、首次补球前”的
    # 确定性真实收益；无法形成合成的样本记为 0，分母始终固定。
    per_swap = {}
    all_swap_keys = set()

    for sample in samples:
        state = sample["state"]
        for p1, p2 in _future_legal_swaps(state):
            all_swap_keys.add((p1, p2))

    if not all_swap_keys:
        return {
            "current_chain_gain": current_chain_expected,
            "current_chain_samples": current_chain_values,
            "current_sample_count": sample_plan.count,
            "sample_plan": sample_plan,
            "balance_report": balance_report,
            "next_gain": 0.0,
            "next_chain_gain": 0.0,
            "future_gain": 0.0,
            "potential": 0.0,
            "next_p1": None,
            "next_p2": None,
            "next_gain_samples": [],
            "next_chain_samples": [],
            "next_legal_count": 0,
            "lookahead_enabled": True,
        }

    for swap_key in all_swap_keys:
        p1, p2 = swap_key
        values = []

        for sample in samples:
            outcome = _simulate_next_swap_on_sample(
                sample["state"],
                p1,
                p2,
                sample["sample_index"],
                sample_plan,
            )

            if outcome is None:
                values.append({
                    "next_gain": 0.0,
                    "next_chain_gain": 0.0,
                    "future_gain": 0.0,
                    "potential": 0.0,
                    "sample_index": sample["sample_index"],
                    "legal": False,
                })
            else:
                values.append({
                    **outcome,
                    "sample_index": sample["sample_index"],
                    "legal": True,
                })

        per_swap[swap_key] = values

    best = None

    for swap_key, values in per_swap.items():
        # 固定全部样本求平均，得到下一步交换收益期望。
        next_gain_samples = [
            float(v["next_gain"]) for v in values
        ]
        next_gain_expected = statistics.fmean(
            next_gain_samples
        )

        legal_count = sum(
            1 for v in values if v.get("legal", False)
        )

        if best is None or next_gain_expected > best["next_gain"]:
            best = {
                "swap_key": swap_key,
                "next_gain": float(next_gain_expected),
                "values": values,
                "next_gain_samples": next_gain_samples,
                "legal_count": legal_count,
            }

    potential_values = [
        float(v.get("potential", 0.0))
        for v in best["values"]
        if v.get("legal", False)
    ]
    potential = (
        statistics.fmean(potential_values)
        if potential_values else 0.0
    )

    return {
        "current_chain_gain": float(current_chain_expected),
        "current_chain_samples": current_chain_values,
        "current_sample_count": sample_plan.count,
        "sample_plan": sample_plan,
        "balance_report": balance_report,

        # 最终评分只使用“下一步直接交换收益期望”。
        "next_gain": float(best["next_gain"]),

        # 该字段保留在统一结果结构中，但不参与评分。
        "next_chain_gain": 0.0,
        "future_gain": float(best["next_gain"]),

        "potential": float(potential),
        "next_p1": best["swap_key"][0],
        "next_p2": best["swap_key"][1],
        "next_gain_samples": list(best["next_gain_samples"]),
        "next_chain_samples": [],
        "next_legal_count": int(best["legal_count"]),
        "lookahead_enabled": True,
    }



def analyze_all_swaps(board, *args, **kwargs):
    """
    在全部 351 种交换中选择当前最优方案。

    流程：
      351 种交换 -> 快速筛选前 12 名
      -> 9 个均衡随机样本统一评估 -> 最终排序

    当前连锁：
      每个随机样本的真实连锁收益平均值

    下一步：
      对每个具体交换跨全部随机样本求期望，
      然后选择期望最高的交换。

    最终评分：
      当前收益 + 下一步收益期望

    局面潜力只显示，不进入评分。
    """
    t0 = time.perf_counter()
    current_step = int(kwargs.get("current_step", 1))

    def evaluate_candidates(candidates, sample_plan):
        if not candidates:
            return []

        if not isinstance(
            sample_plan,
            SamplingPlan,
        ):
            raise TypeError("候选评估需要 SamplingPlan")

        results = []
        pending = []
        evaluated = {}

        # 每个候选只读自己的棋盘副本，先并行完成随机评估；
        # 缓存和统计仍由主线程处理，避免并发写入全局状态。
        executor = None
        try:
            for index, result in enumerate(candidates, 1):
                if AUTO_STOP_REQUESTED:
                    break

                b2 = result.get("board")
                if b2 is None:
                    evaluated[index] = (result, None)
                    continue

                cache_key = (
                    "balanced_sampling",
                    _board_cache_key(b2),
                    sample_plan.pass_index,
                    sample_plan.seeds,
                    current_step < TOTAL_GAME_MOVES,
                )

                if cache_key in SWAP_EVAL_CACHE:
                    SEARCH_STATS["cache_hits"] += 1
                    evaluated[index] = (
                        result,
                        SWAP_EVAL_CACHE[cache_key],
                    )
                else:
                    pending.append((index, result, cache_key))

            if pending:
                worker_count = min(
                    EVALUATION_WORKERS,
                    len(pending),
                )
                executor = ThreadPoolExecutor(
                    max_workers=worker_count
                )

                for index, result, cache_key in pending:
                    evaluated[index] = (
                        result,
                        executor.submit(
                            _evaluate_candidate_samples,
                            result["board"],
                            sample_plan,
                            current_step,
                        ),
                        cache_key,
                    )

            for index, result in enumerate(candidates, 1):
                if index not in evaluated:
                    break

                item = evaluated[index]
                if item[1] is None:
                    results.append(result)
                    continue

                future = item[1]
                if hasattr(future, "result"):
                    future = future.result()
                    SWAP_EVAL_CACHE[item[2]] = future

                r = dict(result)

                r["current_chain_gain"] = float(
                    future["current_chain_gain"]
                )

                # 保留逐样本结果，便于检查平均收益的来源。
                r["current_chain_samples"] = list(
                    future.get("current_chain_samples", [])
                )
                r["current_sample_count"] = int(
                    future.get(
                        "current_sample_count",
                        sample_plan.count,
                    )
                )
                r["sample_pass"] = int(
                    sample_plan.pass_index
                )
                r["next_gain_samples"] = list(
                    future.get("next_gain_samples", [])
                )
                r["next_chain_samples"] = list(
                    future.get("next_chain_samples", [])
                )

                r["current_gain"] = (
                    r["immediate_gain"]
                    + r["current_chain_gain"]
                )

                r["next_gain"] = float(
                    future["next_gain"]
                )

                r["next_chain_gain"] = float(
                    future["next_chain_gain"]
                )

                r["future_gain"] = float(
                    future["future_gain"]
                )

                r["potential"] = float(
                    future["potential"]
                )

                r["next_p1"] = future["next_p1"]
                r["next_p2"] = future["next_p2"]
                r["lookahead_enabled"] = bool(
                    future.get("lookahead_enabled", True)
                )

                # 统一结果结构中的补充字段。
                r["chain_gain"] = r["current_chain_gain"]
                r["second_expected"] = r["next_gain"]
                r["chain_value"] = r["current_chain_gain"]
                r["future_potential"] = r["potential"]

                # 最终评分 = 当前交换收益 + 当前交换后连锁期望
                #           + 下一步交换收益期望。
                # 下一步交换后的连锁收益不参与评分。
                lookahead_gain = (
                    r["next_gain"]
                    if r.get("lookahead_enabled", True)
                    else 0.0
                )
                r["real_score"] = (
                    r["current_gain"]
                    + lookahead_gain
                )

                # 局面潜力不换算成能量。
                r["potential_bonus"] = 0.0
                r["final_score"] = r["real_score"]
                r["future_score"] = r["final_score"]
                r["score"] = r["final_score"]

                assert_candidate(r)
                results.append(r)

                print(
                    f"\r{index}/{len(candidates)}",
                    end="",
                    flush=True
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        print()

        # 主排序只看真实累计收益。
        results.sort(
            key=lambda r: r["final_score"],
            reverse=True
        )

        return results

    # 快速筛选全部 351 种交换，保留前 12 名。
    quick_start = time.perf_counter()
    quick = []

    for p1, p2 in generate_all_swaps():
        try:
            result = evaluate_swap(board, p1, p2)
        except Exception as exc:
            if PRINT_SEARCH_BENCHMARK:
                print(
                    f"交换评估出错 "
                    f"{p1}<->{p2}: {exc!r}"
                )
            continue

        if result is None:
            continue

        result = candidate_from_result(result)
        result["quick_score"] = float(
            result["immediate_gain"]
        )
        result["final_score"] = float(
            result["immediate_gain"]
        )
        result["score"] = result["final_score"]
        assert_candidate(result)
        quick.append(result)

    if not quick:
        if PRINT_SEARCH_BENCHMARK:
            print(
                "调试：快速筛选结果为空；"
                "请检查 evaluate_swap / make_candidate。"
            )
        return []

    quick.sort(
        key=lambda r: (
            r["quick_score"],
            board_potential(
                r["board"],
                current_step
            )
        ),
        reverse=True
    )

    quick = _unique_swap_candidates(quick)
    quick = quick[:FAST_FILTER_LIMIT]
    quick_elapsed = time.perf_counter() - quick_start

    # 对全部快速筛选候选统一使用 9 个均衡样本评估。
    final_start = time.perf_counter()
    final = evaluate_candidates(
        quick,
        make_evaluation_plan(),
    )
    final_elapsed = time.perf_counter() - final_start

    if len(SWAP_EVAL_CACHE) > EVALUATION_CACHE_LIMIT:
        while len(SWAP_EVAL_CACHE) > EVALUATION_CACHE_LIMIT:
            SWAP_EVAL_CACHE.pop(
                next(iter(SWAP_EVAL_CACHE))
            )

    if PRINT_SEARCH_BENCHMARK:
        elapsed = time.perf_counter() - t0
        print(
            f"[搜索] 351 -> {len(quick)} | "
            f"样本={FINAL_SAMPLE_COUNT} | "
            f"线程={EVALUATION_WORKERS} | "
            f"快速筛选={quick_elapsed:.3f}s | "
            f"最终评估={final_elapsed:.3f}s | "
            f"缓存命中={SEARCH_STATS['cache_hits']} | "
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


def save_swap_results(results, step):
    """将完整候选筛选结果保存到 DEBUG 文本文件，不刷屏终端。"""
    if not DEBUG_SAVE_IMAGES:
        return

    os.makedirs(SEARCH_DEBUG_DIR, exist_ok=True)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_swap_results(results)

    output_path = os.path.join(
        SEARCH_DEBUG_DIR,
        f"search_step_{step:02d}.txt",
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(output.getvalue())


def print_swap_results(results):
    print()
    print("=" * 120)
    print("筛选结果")
    print("=" * 120)

    if not results:
        print("当前没有找到有效交换。")
        return

    # 输出全部快速筛选后进入最终评估的候选。
    for i, result in enumerate(
        results[:FAST_FILTER_LIMIT],
        1,
    ):
        print(
            f"{i}. {position_name(result['p1'])} <-> "
            f"{position_name(result['p2'])}"
        )

        print(
            f"   当前交换收益 = "
            f"{result.get('immediate_gain', 0.0):.2f}"
        )

        current_samples = result.get("current_chain_samples", [])
        current_sample_count = result.get(
            "current_sample_count",
            len(current_samples),
        )

        if current_samples:
            print(
                "   当前连锁样本：平均="
                + f"{statistics.fmean(current_samples):.2f} | "
                + ", ".join(f"{x:.2f}" for x in current_samples)
                + f" ({current_sample_count}个)"
            )

        print(
            f"   当前完整收益 = "
            f"{result.get('current_gain', 0.0):.2f}"
        )

        if result.get("lookahead_enabled", True):
            next_p1 = result.get("next_p1")
            next_p2 = result.get("next_p2")

            if next_p1 is not None and next_p2 is not None:
                next_swap_text = (
                    f"{position_name(next_p1)} <-> "
                    f"{position_name(next_p2)}"
                )
            else:
                next_swap_text = "无"

            print(
                f"   下一步期望最优交换 = {next_swap_text}"
            )

            print(
                f"   下一步交换收益期望 = "
                f"{result.get('next_gain', 0.0):.2f}"
            )

            legal_counts = result.get("next_legal_counts", {})
            if legal_counts:
                print(
                    "   下一步交换分层可用样本 = "
                    + ", ".join(
                        f"{t}:{legal_counts.get(t, 0)}"
                        for t in TYPE_ORDER
                    )
                )
        else:
            print("   已到最后一步，最终评分不计入下一步交换收益期望")

        print(
            f"   局面潜力 = "
            f"{result.get('potential', 0.0):.2f} | "
            f"仅作辅助，不计入最终能量评分"
        )

        print(
            f"   最终评分 = "
            f"{result.get('final_score', result.get('score', 0.0)):.2f}"
        )


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
    print(
        f"第1步：点击 {position_name(best['p1'])}"
    )

    print(
        f"第2步：点击 {position_name(best['p2'])}"
    )

    print()
    print(
        f"当前交换收益："
        f"{best.get('immediate_gain', 0.0):.2f}"
    )
    current_samples = best.get("current_chain_samples", [])
    if current_samples:
        print(
            f"当前连锁样本（{len(current_samples)}个）："
            + ", ".join(f"{x:.2f}" for x in current_samples)
            + " | 平均="
            + f"{statistics.fmean(current_samples):.2f}"
        )

    print(
        f"当前完整收益："
        f"{best.get('current_gain', 0.0):.2f}"
    )

    if best.get("lookahead_enabled", True):
        next_p1 = best.get("next_p1")
        next_p2 = best.get("next_p2")

        if next_p1 is not None and next_p2 is not None:
            print(
                "下一步期望最优交换："
                f"{position_name(next_p1)} <-> {position_name(next_p2)}"
            )

        print(
            f"下一步交换收益期望："
            f"{best.get('next_gain', 0.0):.2f}"
        )
    else:
        print("已到最后一步，最终评分不计入下一步交换收益期望")
    print(
        f"局面潜力（仅辅助）："
        f"{best.get('potential', 0.0):.2f}"
    )
    print(
        f"最终评分："
        f"{best.get('final_score', best.get('score', 0.0)):.2f}"
    )


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
        os.makedirs(
            DEBUG_DIR,
            exist_ok=True
        )

    calibration = calibrate()

    if calibration is None or AUTO_STOP_REQUESTED:
        print("收到停止指令，程序结束。")
        return

    x1, y1, dx, dy = calibration


    # 框选完成后直接等待完整棋盘稳定出现。
    with mss.MSS() as sct:

        full_monitor = sct.monitors[0]
        monitor = get_board_capture_monitor(
            x1,
            y1,
            dx,
            dy,
            full_monitor,
        )

        while True:
            print()
            initial_result = wait_for_initial_board_ready(
                sct,
                monitor,
                x1,
                y1,
                dx,
                dy
            )

            if initial_result is RESELECT_RESULT:
                RESELECT_REQUESTED.clear()
                calibration = calibrate(
                    wait_for_enter=False
                )

                if calibration is None:
                    print("收到停止指令，程序结束。")
                    return

                x1, y1, dx, dy = calibration
                monitor = get_board_capture_monitor(
                    x1,
                    y1,
                    dx,
                    dy,
                    full_monitor,
                )
                continue

            if initial_result is None:
                print("收到停止指令，程序结束。")
                return

            _, capture_x1, capture_y1, dx, dy = initial_result
            x1 = capture_x1 + monitor["left"]
            y1 = capture_y1 + monitor["top"]

            print("初始棋盘已稳定，开始截图...")
            break


    run_auto_loop(
        x1,
        y1,
        dx,
        dy
    )

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
