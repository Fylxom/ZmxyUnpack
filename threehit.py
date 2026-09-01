# 是否保存 DEBUG 调试图片；False 时不产生 debug 图片
DEBUG_SAVE_IMAGES = False

from dataclasses import dataclass
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
from concurrent.futures import ThreadPoolExecutor

# 自动运行/搜索停止标志
STOP_EVENT = threading.Event()
AUTO_STOP_REQUESTED = False
# 交换评估缓存
SWAP_EVAL_CACHE = {}


# ============================================================
# 造梦西游5 棋盘识别 V5.12-fixed2-fixed
#
# 颜色面积识别
#
#   火 = f
#   雷 = r
#   风 = w
#
# 不使用：
#
#   模板匹配
#   多尺度模板
#   OCR
#
# 等级判断：
#
#   每一种颜色分别建立自己的面积阈值
#
# ============================================================


ROWS = 3
COLS = 9

DEBUG_DIR = "debug"


# ============================================================
# 中央识别区域
#
# 每个方格只取中央区域。
#
# 这样可以尽量排除：
#
#   方框
#   边框
#   外部背景
# ============================================================

CENTER_RATIO = 0.82


# ============================================================
# 背景排除
# ============================================================

BACKGROUND_DISTANCE = 35


# ============================================================
# 最小颜色面积
# ============================================================

MIN_COLOR_RATIO = 0.005


# ============================================================
# 等级阈值
#
# 注意：
#
# 这是“颜色像素面积 / 中央识别区域面积”
#
# 不是整个球的真实面积。
#
# ------------------------------------------------------------
#
# 火：
#   1 / 2 = 0.170
#   2 / 3 = 0.280
#
# 雷：
#   1 / 2 = 0.315
#   2 / 3 = 0.395
#
# 风：
#   1 / 2 = 0.375
#   2 / 3 = 0.600
#
# ============================================================

LEVEL_THRESHOLDS = {

    "f": {

        # 1 / 2 分界
        "12": 0.170,

        # 2 / 3 分界
        "23": 0.280
    },


    "r": {

        # 1 / 2 分界
        "12": 0.315,

        # 2 / 3 分界
        "23": 0.395
    },


    "w": {

        # 1 / 2 分界
        "12": 0.375,

        # 2 / 3 分界
        "23": 0.600
    }
}


# ============================================================
# 类型名称
# ============================================================

TYPE_NAME = {

    "f": "火",

    "r": "雷",

    "w": "风",

    "?": "?"
}


# ============================================================
# 绘图颜色
# ============================================================

DRAW_COLOR = {

    "f": (0, 0, 255),

    "r": (255, 0, 255),

    "w": (255, 255, 0),

    "?": (0, 255, 255)
}


# ============================================================
# 鼠标校准
# ============================================================


# V5.3/V5.12 统计计数器（仅用于运行统计，不参与评分）
V53_STATS = {
    "quick_candidates": 0,
    "future_simulations": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

# ============================================================
# V5.12 统一采样配置
# ============================================================
# 每个补球位置的 f/r/w 严格各占 1/3。
# Stage 2 使用 6 个样本；Stage 3 使用 9 个样本。
V512_STAGE2_SAMPLE_COUNT = 6
V512_STAGE3_SAMPLE_COUNT = 9

# 兼容旧 V5.3 运行统计；只统计计算量，不参与评分。
V53_STATS = {
    "quick_candidates": 0,
    "future_simulations": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

def wait_mouse(message):

    print()
    print(message)

    print(
        "请把鼠标移动到目标位置，"
        "然后按 Enter。"
    )

    input()

    pos = pyautogui.position()

    x = int(pos.x)
    y = int(pos.y)

    print(
        f"记录坐标：({x}, {y})"
    )

    return x, y


# ============================================================
# 棋盘校准
# ============================================================

def calibrate():

    print()
    print("=" * 80)
    print("框选区域")
    print("=" * 80)

    x1, y1 = wait_mouse(
        "① 第1行第1列方框中心"
    )

    x2, y2 = wait_mouse(
        "② 第3行第9列方框中心"
    )


    dx = (
        x2 - x1
    ) / (
        COLS - 1
    )


    dy = (
        y2 - y1
    ) / (
        ROWS - 1
    )


    print()
    print("=" * 80)
    print("校准结果")
    print("=" * 80)

    print(
        f"左上：({x1}, {y1})"
    )

    print(
        f"右下：({x2}, {y2})"
    )

    print(
        f"横向格距：{dx:.3f}"
    )

    print(
        f"纵向格距：{dy:.3f}"
    )


    return (
        x1,
        y1,
        dx,
        dy
    )


# ============================================================
# 截取格子
# ============================================================

def crop_cell(
    screenshot,
    cx,
    cy,
    cell_w,
    cell_h
):

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


# ============================================================
# 取得中央区域
# ============================================================

def get_center_region(cell):

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


# ============================================================
# 估计棋盘背景
# ============================================================

def estimate_background(img):

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


# ============================================================
# 背景距离
# ============================================================

def get_background_mask(
    img,
    background
):

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


# ============================================================
# 创建三种颜色 Mask
# ============================================================

def make_color_masks(img):

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


    # ========================================================
    # 火
    #
    # 红橙色
    #
    # 不能仅仅使用 R > G。
    #
    # 因为棋盘背景也是棕橙色。
    # ========================================================

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


    # ========================================================
    # 雷
    #
    # 紫色
    # ========================================================

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


    # ========================================================
    # 风
    #
    # 灰蓝色
    # ========================================================

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


    # ========================================================
    # 去噪
    # ========================================================

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


    # ========================================================
    # 闭运算
    # ========================================================

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


# ============================================================
# 找最佳颜色区域
# ============================================================

def get_best_component(mask):

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


# ============================================================
# 测量区域
# ============================================================

def measure_component(
    component
):

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:

        return {

            "area": 0,

            "width": 0,

            "height": 0,

            "diameter": 0,

            "circularity": 0
        }


    contour = max(
        contours,
        key=cv2.contourArea
    )


    area = cv2.contourArea(
        contour
    )


    x, y, w, h = \
        cv2.boundingRect(
            contour
        )


    perimeter = cv2.arcLength(
        contour,
        True
    )


    if perimeter > 0:

        circularity = (

            4.0 *
            math.pi *
            area
            /
            (
                perimeter *
                perimeter
            )
        )

    else:

        circularity = 0


    diameter = (

        2.0 *
        math.sqrt(
            max(area, 0)
            /
            math.pi
        )
    )


    return {

        "area": area,

        "width": w,

        "height": h,

        "diameter": diameter,

        "circularity":
            circularity
    }


# ============================================================
# 等级判断
# ============================================================

def estimate_level(
    color_type,
    ratio
):

    if ratio <= 0:

        return 0


    thresholds = \
        LEVEL_THRESHOLDS[
            color_type
        ]


    # --------------------------------------------------------
    # 1级
    # --------------------------------------------------------

    if ratio < thresholds["12"]:

        return 1


    # --------------------------------------------------------
    # 2级
    # --------------------------------------------------------

    if ratio < thresholds["23"]:

        return 2


    # --------------------------------------------------------
    # 3级
    # --------------------------------------------------------

    return 3


# ============================================================
# 识别单个格子
# ============================================================

def recognize_cell(
    cell,
    row,
    col,
    debug_dir
):

    center = \
        get_center_region(
            cell
        )


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


    # ========================================================
    # 分析三种颜色
    # ========================================================

    for color_type, mask in \
            masks.items():

        component, pixel_area = \
            get_best_component(
                mask
            )


        measurement = \
            measure_component(
                component
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
                ratio,

            "measurement":
                measurement
        }


    # ========================================================
    # 找颜色
    # ========================================================

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


    # ========================================================
    # 没有颜色
    # ========================================================

    if not valid:

        result = {

            "type": "?",

            "level": 0,

            "area_ratio": 0,

            "width_ratio": 0,

            "height_ratio": 0,

            "diameter": 0,

            "confidence": 0
        }


        return (
            result,
            data
        )


    # ========================================================
    # 最大颜色
    # ========================================================

    best_type = max(
        valid,
        key=valid.get
    )


    best_ratio = valid[
        best_type
    ]


    # ========================================================
    # 第二大颜色
    # ========================================================

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


    # ========================================================
    # 颜色差距
    # ========================================================

    if best_ratio > 0:

        color_gap = (

            best_ratio -
            second_ratio
        ) / best_ratio

    else:

        color_gap = 0


    # ========================================================
    # 几何
    # ========================================================

    measurement = data[
        best_type
    ]["measurement"]


    width_ratio = (

        measurement["width"]
        /
        w
    )


    height_ratio = (

        measurement["height"]
        /
        h
    )


    diameter = \
        measurement["diameter"]


    # ========================================================
    # 等级
    # ========================================================

    level = estimate_level(
        best_type,
        best_ratio
    )


    # ========================================================
    # 特殊保护
    #
    # 面积特别小：
    #
    # 不允许直接成为 3级。
    # ========================================================

    if best_ratio < 0.08:

        level = 1


    # ========================================================
    # 置信度
    # ========================================================

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


    # ========================================================
    # 最终
    # ========================================================

    result = {

        "type": best_type,

        "level": level,

        "area_ratio": best_ratio,

        "width_ratio":
            width_ratio,

        "height_ratio":
            height_ratio,

        "diameter":
            diameter,

        "confidence":
            confidence
    }


    # ========================================================
    # DEBUG
    # ========================================================

    base = os.path.join(
        debug_dir,
        f"r{row + 1}c{col + 1}"
    )


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_center.png",
        center
    )


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_fire.png",
        fire_mask
    )


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_thunder.png",
        thunder_mask
    )


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_wind.png",
        wind_mask
    )


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_component.png",
        data[
            best_type
        ]["component"]
    )


    # ========================================================
    # Debug 图
    # ========================================================

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


    label = (

        f"{level}{best_type} "

        f"A={best_ratio:.3f}"
    )


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


    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        base +
        "_result.png",
        debug_img
    )


    return (
        result,
        data
    )


# ============================================================
# 主程序
# ============================================================


# ============================================================
# 暂停/继续控制
# ============================================================

# F8：暂停 / 继续
# F9：退出
AUTO_PAUSED = False
AUTO_STOP_REQUESTED = False


def setup_hotkeys_v43():
    global AUTO_PAUSED
    global AUTO_STOP_REQUESTED

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
        global AUTO_STOP_REQUESTED
        AUTO_STOP_REQUESTED = True
        print()
        print("=" * 90)
        print("收到停止指令")
        print("=" * 90)

    keyboard.add_hotkey("f8", toggle_pause)
    keyboard.add_hotkey("f9", stop_program)


def wait_while_paused_v43():
    """
    暂停状态下保持程序运行。
    F8 恢复，F9 退出。
    """
    while AUTO_PAUSED and not AUTO_STOP_REQUESTED:
        time.sleep(0.10)


def pause_for_no_move_v43():
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
    print("F8 = 继续")
    print("F9 = 退出")
    print("=" * 90)


# ============================================================
# 自适应等待 + 自动连续交换
# ============================================================
#
#   交换后检测棋盘图像是否稳定。
#   棋盘稳定后立即进入下一轮。
#
#   普通三连 -> 等待时间短
#   连锁合成 -> 检测到棋盘持续变化时继续等待


AUTO_CLICK_ENABLED = True

# 两次点击之间的间隔
AUTO_CLICK_DELAY = 0.12

# 鼠标点击后移到棋盘外，避免指针遮挡小球影响识别
MOUSE_MOVE_OUT_X = 10
MOUSE_MOVE_OUT_Y = 10

# 点击完成后，至少等待这么久再开始判断棋盘
MIN_WAIT_AFTER_SWAP = 0.35

# 检查棋盘的间隔
BOARD_CHECK_INTERVAL = 0.08

# 棋盘连续稳定这么久，认为动画结束
BOARD_STABLE_TIME = 0.45

# 最多等待这么久。
# 如果游戏动画异常，也不会无限卡住。
MAX_WAIT_AFTER_SWAP = 3.00

# 检测图像变化的阈值。
# 使用灰度绝对差的平均值。
BOARD_CHANGE_THRESHOLD = 2.5

# 自动运行最长时间（秒）
GAME_MAX_SECONDS = 3600.0


def get_cell_center_from_board(r, c, x1, y1, dx, dy):
    return (
        int(round(x1 + c * dx)),
        int(round(y1 + r * dy)),
    )


def get_board_detection_region(screenshot, x1, y1, dx, dy):
    """
    只截取棋盘区域，不检测整个屏幕。

    边界覆盖 3×9 棋盘，并留少量余量。
    """
    h, w = screenshot.shape[:2]

    left = int(round(x1 - abs(dx) * 0.55))
    top = int(round(y1 - abs(dy) * 0.55))

    right = int(round(
        x1 + 8 * dx + abs(dx) * 0.55
    ))

    bottom = int(round(
        y1 + 2 * dy + abs(dy) * 0.55
    ))

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))

    return screenshot[top:bottom, left:right]


def capture_board_region(sct, monitor, x1, y1, dx, dy):
    """
    使用 MSS 直接截取棋盘区域。
    """

    screen_w = monitor["width"]
    screen_h = monitor["height"]

    left = int(round(x1 - abs(dx) * 0.55))
    top = int(round(y1 - abs(dy) * 0.55))

    width = int(round(abs(dx) * 8 + abs(dx) * 1.10))
    height = int(round(abs(dy) * 2 + abs(dy) * 1.10))

    left = max(0, min(screen_w - 1, left))
    top = max(0, min(screen_h - 1, top))

    width = max(1, min(screen_w - left, width))
    height = max(1, min(screen_h - top, height))

    raw = np.array(
        sct.grab({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })
    )

    return cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2GRAY
    )


def board_image_difference(a, b):
    """
    返回两个棋盘截图的平均像素变化。

    为了降低鼠标指针/轻微渲染变化影响，
    先做轻微模糊，再计算平均绝对差。
    """

    if a is None or b is None:
        return float("inf")

    if a.shape != b.shape:
        return float("inf")

    a_blur = cv2.GaussianBlur(
        a,
        (5, 5),
        0
    )

    b_blur = cv2.GaussianBlur(
        b,
        (5, 5),
        0
    )

    diff = cv2.absdiff(
        a_blur,
        b_blur
    )

    return float(
        np.mean(diff)
    )


def wait_for_board_stable(
    sct,
    monitor,
    x1,
    y1,
    dx,
    dy
):
    """
    交换完成后等待棋盘稳定。

    状态：

        点击
         ↓
        最短等待
         ↓
        截图比较
         ↓
        发生变化 -> stable_time 清零
         ↓
        没变化 -> 累计稳定时间
         ↓
        稳定达到 BOARD_STABLE_TIME
         ↓
        返回
    """

    time.sleep(
        MIN_WAIT_AFTER_SWAP
    )

    start_time = time.monotonic()

    previous = capture_board_region(
        sct,
        monitor,
        x1,
        y1,
        dx,
        dy
    )

    stable_start = None

    while True:

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


        time.sleep(
            BOARD_CHECK_INTERVAL
        )


        current = capture_board_region(
            sct,
            monitor,
            x1,
            y1,
            dx,
            dy
        )


        difference = \
            board_image_difference(
                previous,
                current
            )


        print(
            f"\r  检测棋盘稳定："
            f"{elapsed:.2f}s "
            f"变化={difference:.2f}",
            end="",
            flush=True
        )


        if difference <= \
                BOARD_CHANGE_THRESHOLD:

            if stable_start is None:

                stable_start = \
                    time.monotonic()

            stable_elapsed = (
                time.monotonic()
                -
                stable_start
            )

            if stable_elapsed >= \
                    BOARD_STABLE_TIME:

                print(
                    f"\r  棋盘已稳定，"
                    f"等待 {elapsed:.2f}s。"
                    f"              "
                )

                return elapsed

        else:

            stable_start = None


        previous = current


def click_best_swap_v43(
    results,
    x1,
    y1,
    dx,
    dy,
    sct,
    monitor
):
    """
    执行当前一步最优交换，然后等待棋盘稳定。

    返回：
        True  = 成功执行
        False = 没有可执行交换
    """

    if not results:

        print()
        print("=" * 90)
        print("V4.3 自动操作")
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


    # --------------------------------------------------------
    # 执行交换
    # --------------------------------------------------------

    pyautogui.click(
        x1_click,
        y1_click
    )

    time.sleep(
        AUTO_CLICK_DELAY
    )

    pyautogui.click(
        x2_click,
        y2_click
    )

    # 点击完成后立即移出棋盘，避免鼠标指针遮挡小球
    pyautogui.moveTo(MOUSE_MOVE_OUT_X, MOUSE_MOVE_OUT_Y, duration=0.05)


    print()
    print(
        "交换已执行，等待棋盘稳定..."
    )


    wait_for_board_stable(
        sct,
        monitor,
        x1,
        y1,
        dx,
        dy
    )


    return True


def run_auto_loop_v43(
    x1,
    y1,
    dx,
    dy
):
    global AUTO_STOP_REQUESTED
    STOP_EVENT.clear()
    AUTO_STOP_REQUESTED = False
    AUTO_STOP_REQUESTED = False
    """
    V4.3 主循环：

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

    setup_hotkeys_v43()

    with mss.MSS() as sct:

        monitor = sct.monitors[0]


        while True:

            if AUTO_STOP_REQUESTED:
                break

            wait_while_paused_v43()

            if AUTO_STOP_REQUESTED:
                break

            # ------------------------------------------------
            # 游戏时间保护
            # ------------------------------------------------

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


            print()
            print(
                "=" * 90
            )

            print(
                f"自动第 "
                f"{move_count} 步"
            )


            # ------------------------------------------------
            # 截图
            # ------------------------------------------------


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


            # ------------------------------------------------
            # 识别
            # ------------------------------------------------

            cell_w = abs(dx) * 0.90
            cell_h = abs(dy) * 0.90


            board = []


            for row in range(ROWS):

                board_row = []


                for col in range(COLS):

                    cx = (
                        x1 +
                        col * dx
                    )

                    cy = (
                        y1 +
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
                            "width_ratio": 0,
                            "height_ratio": 0,
                            "diameter": 0,
                            "confidence": 0,
                        }

                    else:

                        result, _ = \
                            recognize_cell(
                                cell,
                                row,
                                col,
                                DEBUG_DIR
                            )


                    board_row.append(
                        result
                    )


                board.append(
                    board_row
                )


            # ------------------------------------------------
            # 检查是否有识别失败
            # ------------------------------------------------

            unknown_count = sum(
                1
                for row in board
                for result in row
                if result["type"] == "?"
            )


            if unknown_count > 0:

                print(
                    f"警告：本轮有 "
                    f"{unknown_count} 个格子无法识别。"
                )

                print(
                    "为安全起见，本轮不点击。"
                )

                move_count -= 1

                time.sleep(
                    BOARD_CHECK_INTERVAL
                )

                continue


            # ------------------------------------------------
            # 计算当前最优交换
            # ------------------------------------------------

            board_state = \
                make_board_state(
                    board
                )


            print_board_state(
                board_state
            )


            try:
                results = \
                    analyze_all_swaps_v50_1(
                        board_state,
                        current_step=move_count
                    )
            except StopIteration:
                # 用户按下停止热键时，搜索线程/阶段会通过
                # StopIteration 终止计算；这属于正常退出路径，
                # 不应显示为“程序发生错误”。
                print()
                print("=" * 90)
                print("收到停止指令，停止当前搜索。")
                print("=" * 90)
                break

            # 搜索过程中收到停止指令时，即使分析器已经返回，
            # 也不能继续点击下一步。
            if AUTO_STOP_REQUESTED or STOP_EVENT.is_set():
                print()
                print("收到停止指令，不执行本轮交换。")
                break

            print_swap_results_v50_1(
                results
            )


            if not results:

                # 没有立即可合成交换时暂停，
                # 不结束整个程序。
                pause_for_no_move_v43()

                if AUTO_STOP_REQUESTED:
                    break

                # 等待用户 F8 恢复。
                wait_while_paused_v43()

                # 用户恢复后，重新执行本轮识别。
                # 当前 move 不算一次真正执行的交换。
                move_count -= 1

                continue


            print_best_swap(
                results
            )


            # ------------------------------------------------
            # 执行一次
            # ------------------------------------------------

            click_best_swap_v43(
                results,
                x1,
                y1,
                dx,
                dy,
                sct,
                monitor
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


def _line_match_at(board, r, c, dr, dc):
    """只检查一个位置所在的连续同球区域。"""
    ball = board[r][c]
    if not mergeable(ball):
        return None

    positions = [(r, c)]

    rr, cc = r + dr, c + dc
    while 0 <= rr < ROWS and 0 <= cc < COLS and same_ball(ball, board[rr][cc]):
        positions.append((rr, cc))
        rr += dr
        cc += dc

    rr, cc = r - dr, c - dc
    while 0 <= rr < ROWS and 0 <= cc < COLS and same_ball(ball, board[rr][cc]):
        positions.append((rr, cc))
        rr -= dr
        cc -= dc

    return positions if len(positions) >= 3 else None


def find_matches_after_swap(board, p1, p2):
    """
    交换只改变 p1 / p2 两个格子。
    如果交换前棋盘已经稳定，那么新产生的三连一定经过
    p1 或 p2，因此不需要再扫描整个 27 格棋盘。
    """
    simulated = swap_cells(board, p1, p2)
    found = []
    seen = set()

    for r, c in (p1, p2):
        for dr, dc in ((0, 1), (1, 0)):
            group = _line_match_at(simulated, r, c, dr, dc)
            if group:
                key = frozenset(group)
                if key not in seen:
                    seen.add(key)
                    found.append(group)

    return simulated, found






def analyze_all_swaps(board):
    results = []

    for p1, p2 in generate_all_swaps():
        result = evaluate_swap(board, p1, p2)
        if result is not None:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_match_group(group, board):
    ball = board[group[0][0]][group[0][1]]
    positions = " ".join(position_name(p) for p in group)

    print(
        f"      {cell_to_text(ball)}: {positions}"
    )


def print_swap_results(results):
    print()
    print("=" * 90)
    print("可立即产生合成的交换")
    print("=" * 90)

    if not results:
        print()
        print("当前没有发现能够立即形成三连的交换。")
        return

    for i, result in enumerate(results, 1):
        print()
        print(
            f"{i}. "
            f"{position_name(result['p1'])} <-> "
            f"{position_name(result['p2'])} "
            f"评分={result['score']}"
        )

        print("   合成组合：")

        for group in result["matches"]:
            print_match_group(
                group,
                result["board"]
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
    print(f"当前交换收益：{best.get('immediate_gain', 0.0):.2f}")
    print(
        "当前交换后连锁："
        f"{best.get('current_chain_gain', best.get('chain_gain', 0.0)):.2f}"
    )

    next_p1 = best.get("next_p1")
    next_p2 = best.get("next_p2")

    if next_p1 is not None and next_p2 is not None:
        print(
            "下一步期望最优交换："
            f"{position_name(next_p1)} <-> {position_name(next_p2)}"
        )

    print(
        f"下一步交换收益期望：{best.get('next_gain', 0.0):.2f}"
    )
    print(
        f"下一步交换后连锁期望："
        f"{best.get('next_chain_gain', 0.0):.2f}"
    )
    print(
        f"FutureGain：{best.get('future_gain', 0.0):.2f}"
    )
    print(
        f"board_potential（仅辅助）："
        f"{best.get('potential', 0.0):.2f}"
    )

    print()
    print("当前交换后产生：")

    for group in best["matches"]:
        print_match_group(
            group,
            best["board"]
        )



def print_simulated_board(result):
    if result is None:
        return

    print()
    print("=" * 90)
    print("推荐交换后的棋盘")
    print("=" * 90)

    board = result["board"]

    for r in range(ROWS):
        print(
            f"第{r + 1}行： "
            + "   ".join(
                cell_to_text(board[r][c])
                for c in range(COLS)
            )
        )



# ============================================================
# 两层前瞻搜索
# ============================================================

LOOKAHEAD_SAMPLES = 18
LOOKAHEAD_WEIGHT = 0.65
CHAIN_WEIGHT = 1.15

# ============================================================
# V5.2 搜索 / 评分 / 合成模拟器
#
# 重要：
#   下面的规则以已确认的真实游戏行为为准。
#
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
#
# 搜索：
#   351 全交换 -> 快速真实模拟 -> Top30
#   -> 64 次随机期望 -> Top10
#   -> 128 次随机期望 -> Top5
#   -> 256 次高精度期望
#
# 注意：基础模拟器是 ground truth；评分只决定“选哪一步”。
# ============================================================

BALL_ENERGY = {
    "f": {1: 5.0, 2: 20.0, 3: 80.0},
    "r": {1: 4.0, 2: 16.0, 3: 64.0},
    "w": {1: 3.0, 2: 12.0, 3: 48.0},
}

TYPE_ORDER = ("f", "r", "w")
TYPE_NAMES = {"f": "火", "r": "雷", "w": "风"}

# ---------- 搜索参数 ----------
SEARCH_TOP_K = 30
LOOKAHEAD_TOP_K = 10
FINAL_TOP_K = 5

SAMPLES_TOP_K = 64
SAMPLES_MID_K = 128
SAMPLES_FINAL_K = 256

SEARCH_WORKERS = 16

# ---------- 评分参数 ----------
# 权重集中放在这里，后续做 benchmark 时只需要修改这里。
SCORE_WEIGHTS = {
    # 当前交换/当前完整连锁的真实能量
    "immediate": 1.00,

    # V5.7 保留该配置项供旧逻辑兼容；最终评分不再折扣 FutureGain。
    "future": 1.00,

    # 当前一步中额外发生的连锁收益
    "chain": 1.00,

    # 高等级球保留价值
    "inventory": 0.08,

    # 当前棋盘存在两个同球相邻时的潜力
    "pair": 1.5,

    # 空位很少时略微提高“可继续操作”的价值
    "mobility": 0.25,
}

# 让随机前瞻使用同一批种子比较候选，降低候选间的 Monte Carlo 方差。
LOOKAHEAD_SEED_BASE = 0x5A17C9

# ============================================================
# 基础棋盘操作
# ============================================================

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

        # 每个被合成的位置产生空位，原地补一级球。
        # 使用独立 RNG 时由调用者负责确定随机结果。
        # 这里先保留空位，由 resolve_merges_with_refill 负责补球。

        if rounds >= max_rounds:
            break

    return state, total_merges, total_gain


def resolve_one_round(board):
    """
    只执行一次“当前已有组合”的同轮合成，不补球。
    返回：
        state, merge_count, gain, empty_count
    """
    matches = find_matches(board)

    if not matches:
        return copy_board(board), 0, 0.0, 0

    expanded = expand_matches(matches)
    selected = choose_merge_groups(expanded)

    if not selected:
        return copy_board(board), 0, 0.0, 0

    state, count, gain = apply_merge_groups(
        board,
        selected
    )

    return state, count, gain, len(empty_cells(state))


def refill_random(board, rng):
    state = copy_board(board)

    # V5.12 分层 RNG：每一次独立补球事件重新编号空位。
    # 普通 random.Random 没有 begin_refill()，因此保持兼容。
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
    该项只作为 tie-break / 小修正，不改变真实能量价值。
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
            matches=[],
            merge_gain=0.0,
            merge_count=0,
            score=0.0,
        )

    # 关键修正：这里不能只 resolve_one_round。
    # 必须把“第一次补球之前”的确定性连续合成全部吃完。
    resolved, merge_count, merge_gain = resolve_merges(
        simulated,
        max_rounds=20,
    )

    return make_candidate(
        p1=p1,
        p2=p2,
        board=resolved,
        matches=matches,
        merge_gain=merge_gain,
        merge_count=merge_count,
        score=merge_gain,
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
    "p1", "p2", "board", "matches",
    "immediate_gain",
    "future_gain",
    "chain_gain",
    "potential",
    "potential_bonus",
    "real_score",
    "final_score",
    "quick_score",
)




def _v53_board_key(board):
    """
    V5.3/V5.12 兼容棋盘缓存键。

    将棋盘规范化成不可变 tuple，避免直接使用 list 作为 dict key。
    不参与评分，只用于缓存。
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


def _v53_unique_fast(candidates):
    """
    V5.3 兼容的快速去重。

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
    p1, p2, board, matches=None,
    merge_gain=0.0, merge_count=0,
    score=None, first_score=None,
    second_expected=0.0, chain_value=0.0,
    future_potential=0.0,
    quick_score=None, future_score=0.0,
    immediate_gain=None, future_gain=0.0, chain_gain=0.0,
    potential=0.0, potential_bonus=0.0,
    real_score=None, final_score=None,
):
    """统一候选结构；旧字段只作为兼容别名。"""
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
    future = float(
        future_gain
        if future_gain is not None
        else second_expected
    )
    chain = float(
        chain_gain
        if chain_gain is not None
        else chain_value
    )

    if real_score is None:
        real_score = immediate + future

    if final_score is None:
        final_score = real_score

    return {
        "p1": p1,
        "p2": p2,
        "board": board,
        "matches": list(matches or []),
        "immediate_gain": immediate,
        "future_gain": future,
        "chain_gain": chain,
        "potential": potential,
        "potential_bonus": potential_bonus,
        "real_score": float(real_score),
        "final_score": float(final_score),
        "quick_score": quick,

        # 兼容字段
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
        matches=result.get("matches"),
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


# V5.12 搜索配置。
V53_FAST_FILTER_TOP = 30
V53_DEEP_TOP = 6
V53_CACHE_MAX = 30000
V53_BENCHMARK = False

# ============================================================
# 第一阶段：351 全交换快速搜索
# ============================================================

def analyze_all_swaps_basic(board):
    results = []

    for p1, p2 in generate_all_swaps():
        result = evaluate_swap(board, p1, p2)

        # V5.2 第一阶段允许“立即没有三连”的交换进入候选，
        # 因为它们可能通过下一次随机状态获得更高期望。
        if result is not None:
            results.append(result)

    results.sort(
        key=lambda x: (
            x["score"],
            board_potential(x["board"])
        ),
        reverse=True
    )

    return results


# ============================================================
# 随机前瞻
# ============================================================

def make_seed_list(samples, stage):
    return [
        (
            LOOKAHEAD_SEED_BASE
            + stage * 1000003
            + i * 9176
        )
        & ((1 << 63) - 1)
        for i in range(samples)
    ]





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
class V512SamplePlan:
    """
    V5.12 唯一的 sample/seed 数据载体。

    内部所有 sampling/evaluation 函数只接收 SamplePlan，
    不再混用 seed 列表、sample_count 整数等参数。
    """
    stage: int
    seeds: tuple

    @property
    def count(self):
        return len(self.seeds)


def make_v512_sample_plan(stage, count):
    """
    唯一的 SamplePlan 创建入口。
    seeds 永远是 tuple[int, ...]；count 永远通过 len(seeds) 得到。
    """
    stage = int(stage)
    count = int(count)

    if count <= 0 or count % 3 != 0:
        raise ValueError(
            "V5.12 sample count must be a positive multiple of 3"
        )

    seeds = tuple(
        (
            LOOKAHEAD_SEED_BASE
            + stage * 1000003
            + i * 9176
        ) & ((1 << 63) - 1)
        for i in range(count)
    )

    return V512SamplePlan(
        stage=stage,
        seeds=seeds,
    )



def make_v512_stage_plan(stage):
    """根据 stage 创建唯一的 V5.12 SamplePlan。"""
    stage = int(stage)
    if stage == 1:
        count = 6
    elif stage == 2:
        count = 9
    else:
        raise ValueError(f"unsupported V5.12 stage: {stage}")

    return make_v512_sample_plan(stage, count)




V512_BALANCED_TYPES = ("f", "r", "w")


def _balanced_type_for_sample(
    sample_index,
    empty_index,
    sample_plan,
):
    """
    V5.12 独立变量分层采样。

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

    对 6 个样本：
        每个空位仍严格 f/r/w 各 2 次；
        由于 6 不是 3²，不能覆盖全部 9 个组合，但不再人为绑定
        两个变量。

    对超过 2 个变量的情况，使用有限样本下的循环平衡设计。
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")

    count = sample_plan.count
    if count <= 0 or count % 3 != 0:
        raise ValueError(
            f"V5.12 sample count must be a positive multiple of 3: {count}"
        )

    sample_index = int(sample_index)
    empty_index = int(empty_index)

    if not 0 <= sample_index < count:
        raise IndexError("sample_index outside SamplePlan")

    if empty_index < 0:
        raise IndexError("empty_index must be >= 0")

    colors = ("f", "r", "w")

    # --------------------------------------------------------
    # 9 samples：两个变量做完整 3×3 枚举。
    #
    # sample:
    #   0 1 2 3 4 5 6 7 8
    # var0:
    #   f f f r r r w w w
    # var1:
    #   f r w f r w f r w
    #
    # 因而：
    #   P(var0=f,var1=r) = 1/9
    #   P(var0=r) = 1/3
    #   P(var1=w) = 1/3
    # --------------------------------------------------------
    if count == 9:
        a = sample_index // 3
        b = sample_index % 3

        if empty_index == 0:
            digit = a
        elif empty_index == 1:
            digit = b
        else:
            # 后续变量使用独立的拉丁方设计，保证单变量仍为 1/3，
            # 并尽量避免与前两个变量形成固定相位关系。
            # 对 empty_index=2：f,r,w 在每个 row/column 中均衡。
            digit = (a + b) % 3
            if empty_index >= 3:
                digit = (a + (empty_index - 1) * b) % 3

        return colors[digit]

    # --------------------------------------------------------
    # 6 samples：无法完整覆盖两个三值变量的 9 种组合。
    # 使用“块 + 相位”设计，使每个具体空位仍然 f/r/w 各 2 次，
    # 同时不采用简单 empty_index 相位绑定。
    # --------------------------------------------------------
    if count == 6:
        # 前 3 个样本和后 3 个样本分别覆盖 f/r/w。
        # 不同空位使用不同的拉丁排列。
        block = sample_index // 3
        pos = sample_index % 3
        digit = (pos + block * empty_index) % 3
        return colors[digit]

    # --------------------------------------------------------
    # 通用情况：每个具体变量独立做均衡循环。
    # --------------------------------------------------------
    return colors[
        (sample_index + empty_index * (sample_index // 3 + 1)) % 3
    ]




def _make_balanced_refill_board(
    board,
    sample_index,
    sample_plan,
):
    """
    根据统一 SamplePlan 构造一个完整随机补球世界。
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError(
            "sample_plan must be V512SamplePlan"
        )

    if not (
        0 <= int(sample_index) < sample_plan.count
    ):
        raise IndexError(
            "sample_index outside SamplePlan"
        )

    state = copy_board(board)
    empties = empty_cells(state)

    for empty_index, (r, c) in enumerate(empties):
        ball_type = _balanced_type_for_sample(
            sample_index,
            empty_index,
            sample_plan,
        )
        state[r][c] = (
            ball_type,
            1,
        )

    return state



def _validate_v512_plan_balance(
    board,
    sample_plan,
):
    """
    验证每个空位的边缘分布严格为 1/3。
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")

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
                    f"invalid V5.12 ball type: {ball_type!r}"
                )
            counts[ball_type] += 1

        if counts != {
            "f": expected,
            "r": expected,
            "w": expected,
        }:
            raise AssertionError(
                f"V5.12 balance error at {(r, c)}: "
                f"{counts}; sample_count={sample_plan.count}"
            )

        report.append((r, c, counts))

    return report



def _v512_sample_matrix(sample_plan, empty_count):
    """
    调试用：展示一次补球事件中每个空位的样本颜色。
    用于确认“自由变量”没有被错误绑定。
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")

    return [
        [
            _balanced_type_for_sample(
                sample_index,
                empty_index,
                sample_plan,
            )
            for empty_index in range(empty_count)
        ]
        for sample_index in range(sample_plan.count)
    ]


def _refill_random_stratified(
    board,
    rng,
    sample_index=None,
    sample_plan=None,
    forced_first_type=None,
):
    """
    兼容旧调用，但 V5.12 正式路径必须使用 SamplePlan。

    不再接受 seed 列表或 sample_count。
    """
    if sample_plan is not None:
        return _make_balanced_refill_board(
            board,
            sample_index,
            sample_plan,
        )

    # 仅保留旧接口兼容路径；正式 V5.12 evaluator 不使用它。
    state = copy_board(board)
    empties = empty_cells(state)

    for empty_index, (r, c) in enumerate(empties):
        if (
            empty_index == 0
            and forced_first_type in TYPE_ORDER
        ):
            ball_type = forced_first_type
        else:
            ball_type = rng.choice(TYPE_ORDER)

        state[r][c] = (
            ball_type,
            1,
        )

    return state



class _V512BalancedRNG:
    """
    V5.12 的分层随机补球 RNG。

    resolve_with_refill() 本身不修改；它仍然调用：
        refill_random(state, rng)
    而 refill_random 仍然调用：
        rng.choice(TYPE_ORDER)

    这里仅替换 choice() 的取样方式：
    对每一个独立的补球 draw ordinal，跨 SamplePlan 的 samples
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
        # refill_random() 的正式调用是 choice(TYPE_ORDER)。
        # 不依赖 TYPE_ORDER 的顺序，显式返回游戏的 f/r/w。
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


def _v512_fill_initial_empties(
    board,
    sample_index,
    sample_plan,
    rng,
):
    """
    初次补球也走同一个 BalancedRNG，保证所有 refill draw 都处于
    同一个 sample 世界，而不是只有第一次补球 balanced。
    """
    state = copy_board(board)

    for r, c in empty_cells(state):
        state[r][c] = (
            rng.choice(("f", "r", "w")),
            1,
        )

    return state




def _v512_stage2_rng(sample_index, sample_plan):
    """
    Stage-2 sample-local RNG.

    The same SamplePlan seed is reused for every candidate next swap
    (common random numbers). Each draw uses the game's TYPE_ORDER through
    Python's uniform Random.choice, so f/r/w each have probability 1/3.
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")
    return _V512BalancedRNG(
        sample_plan.seeds[sample_index],
        sample_index,
        sample_plan.count,
    )



def _simulate_refill_chain_once(
    board_after_current_merge,
    sample_index,
    sample_plan,
):
    """
    One genuine random sample for current post-refill chain.

    board_after_current_merge has already completed all deterministic
    pre-refill merges. Only gains after the first random refill belong here.
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")

    rng = _V512BalancedRNG(
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
    One genuine random sample for a concrete next swap.

    next_gain:
        all deterministic merges after the swap and before the first refill.

    next_chain_gain:
        all gains beginning with the first random refill and continuing
        through subsequent real chain resolution.

    This is deliberately N-sample Monte Carlo, matching the original
    V5.12 requirement. It does NOT enumerate an exponential probability
    tree and therefore cannot hang on a normal board.
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError("sample_plan must be V512SamplePlan")

    a = board[p1[0]][p1[1]]
    b = board[p2[0]][p2[1]]

    if a is None or b is None or same_ball(a, b):
        return None

    swapped = swap_cells(board, p1, p2)

    if not find_matches(swapped):
        return None

    # All pre-refill deterministic cascade belongs to next_gain.
    next_state, merge_count, next_gain = resolve_merges(
        swapped,
        max_rounds=20,
    )

    if merge_count <= 0:
        return None

    # First refill is the boundary: only gains after it are next_chain_gain.
    if empty_cells(next_state):
        rng = _V512BalancedRNG(
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



def _v512_evaluate_candidate_samples(
    board_after_current_merge,
    sample_plan,
    current_step,
):
    """
    V5.12 唯一核心评估器。

    SamplePlan -> 每个 sample -> 当前连锁
    -> 每个具体下一步交换 -> 跨全部 samples 求期望
    -> 交换之间取最大。

    不存在 seed/list/count 混用。
    """
    if not isinstance(sample_plan, V512SamplePlan):
        raise TypeError(
            "sample_plan must be V512SamplePlan"
        )

    # 审计 balanced sampling。
    balance_report = _validate_v512_plan_balance(
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

    # 下一步交换收益期望：
    # 对同一个具体交换，在全部 sample 上计算“交换后、首次补球前”
    # 的确定性真实收益；无法形成合成的 sample 记 0。
    # 因此分母永远固定为 sample_plan.count。
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
        # 固定全部 N 个样本求平均，这就是下一步交换收益期望。
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

        # 最终评分只使用这个“下一步直接交换收益期望”。
        "next_gain": float(best["next_gain"]),

        # 兼容旧字段，但不再参与评分。
        "next_chain_gain": 0.0,
        "future_gain": float(best["next_gain"]),

        "potential": float(potential),
        "next_p1": best["swap_key"][0],
        "next_p2": best["swap_key"][1],
        "next_gain_samples": list(best["next_gain_samples"]),
        "next_chain_samples": [],
        "next_legal_count": int(best["legal_count"]),
    }



def evaluate_future_once(
    board,
    sample_index,
    sample_plan,
    current_step=1,
):
    """
    单 sample 兼容接口。
    正式路径仍由统一 SamplePlan 管理。
    """
    state, current_chain_gain = (
        _simulate_refill_chain_once(
            board,
            sample_index,
            sample_plan,
        )
    )

    outcomes = {}

    for p1, p2 in _future_legal_swaps(state):
        outcome = _simulate_next_swap_on_sample(
            state,
            p1,
            p2,
            sample_index,
            sample_plan,
        )

        if outcome is not None:
            outcomes[(p1, p2)] = outcome

    return {
        "current_chain_gain": float(
            current_chain_gain
        ),
        "outcomes": outcomes,
    }


def evaluate_future_expected(
    board,
    sample_plan,
    current_step=1,
):
    """统一 SamplePlan 的期望前瞻接口。"""
    return _v512_evaluate_candidate_samples(
        board,
        sample_plan,
        current_step,
    )



def analyze_all_swaps_v50_1(board, *args, **kwargs):
    """
    V5.12-fixed2-fixed：

    351 个当前交换
      -> Top 30
      -> 6 个 balanced samples
      -> Top 6
      -> 9 个 balanced samples 重新计算
      -> 最终排序

    当前连锁：
      mean(每个随机样本的真实连锁收益)

    下一步：
      对每个具体交换跨全部随机样本求期望，
      然后选择期望最高的交换。

    最终评分：
      CurrentGain + FutureGain

    board_potential 只显示，不进入评分。
    """
    t0 = time.perf_counter()
    current_step = int(kwargs.get("current_step", 1))

    def evaluate_stage(candidates, sample_plan):
        if not candidates:
            return []

        if not isinstance(
            sample_plan,
            V512SamplePlan,
        ):
            raise TypeError(
                "evaluate_stage requires V512SamplePlan"
            )

        results = []

        for index, result in enumerate(candidates, 1):
            if STOP_EVENT.is_set() or AUTO_STOP_REQUESTED:
                # 正常停止：返回已完成的部分结果。
                # 外层 run_auto_loop_v43 会再次检查停止状态，
                # 因而不会误执行交换。
                break

            b2 = result.get("board")
            if b2 is None:
                results.append(result)
                continue

            # 缓存的是“这个候选棋盘 + 这批 seeds”对应的完整统计，
            # 不再缓存一个被误解为期望的单一路径结果。
            cache_key = (
                "v512_unified_sampling",
                _v53_board_key(b2),
                sample_plan.stage,
                sample_plan.seeds,
            )

            if cache_key in SWAP_EVAL_CACHE:
                V53_STATS["cache_hits"] += 1
                future = SWAP_EVAL_CACHE[cache_key]
            else:
                V53_STATS["cache_misses"] += 1

                future = _v512_evaluate_candidate_samples(
                    b2,
                    sample_plan,
                    current_step,
                )

                SWAP_EVAL_CACHE[cache_key] = future
                V53_STATS["future_simulations"] += (
                    sample_plan.count
                )

            r = dict(result)

            r["current_chain_gain"] = float(
                future["current_chain_gain"]
            )

            # V5.12-fixed2-fixed 审计字段：这些是同一批 seeds 的逐样本真实结果，
            # current_chain_gain 本身则是它们的 mean。
            r["current_chain_samples"] = list(
                future.get("current_chain_samples", [])
            )
            r["current_sample_count"] = int(
                future.get(
                    "current_sample_count",
                    sample_plan.count,
                )
            )
            r["sample_plan_stage"] = int(
                sample_plan.stage
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

            # 兼容旧字段。
            r["chain_gain"] = r["current_chain_gain"]
            r["second_expected"] = r["next_gain"]
            r["chain_value"] = r["current_chain_gain"]
            r["future_potential"] = r["potential"]

            # 最终评分：
            # CurrentGain = 当前交换收益 + 当前交换后连锁期望
            # FinalScore  = CurrentGain + 下一步交换收益期望
            # 下一步交换后的连锁收益不参与评分。
            r["real_score"] = (
                r["current_gain"]
                + r["next_gain"]
            )

            # board_potential 不再换算成能量。
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

        print()

        # 主排序只看真实累计收益。
        results.sort(
            key=lambda r: r["final_score"],
            reverse=True
        )

        return results

    # --------------------------------------------------------
    # Stage 1：351 -> Top 30
    # --------------------------------------------------------
    quick = []

    for p1, p2 in generate_all_swaps():
        try:
            result = evaluate_swap(board, p1, p2)
        except Exception as exc:
            if V53_BENCHMARK:
                print(
                    f"V5.12 evaluate_swap error "
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

    V53_STATS["quick_candidates"] += len(quick)

    if not quick:
        if V53_BENCHMARK:
            print(
                "V5.12 调试：Stage 1 quick=0；"
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

    quick = _v53_unique_fast(quick)
    quick = quick[:V53_FAST_FILTER_TOP]

    # --------------------------------------------------------
    # Stage 2：6 个 balanced samples
    # --------------------------------------------------------
    stage2 = evaluate_stage(
        quick,
        make_v512_stage_plan(1),
    )

    stage2 = stage2[:V53_DEEP_TOP]

    # --------------------------------------------------------
    # Stage 3：9 个 balanced samples 重新计算
    # --------------------------------------------------------
    final = evaluate_stage(
        stage2,
        make_v512_stage_plan(2),
    )

    if len(SWAP_EVAL_CACHE) > V53_CACHE_MAX:
        while len(SWAP_EVAL_CACHE) > V53_CACHE_MAX:
            SWAP_EVAL_CACHE.pop(
                next(iter(SWAP_EVAL_CACHE))
            )

    if V53_BENCHMARK:
        elapsed = time.perf_counter() - t0
        print(
            f"[V5.12-fixed2-fixed benchmark] 351 -> {len(quick)} -> "
            f"{len(stage2)} | "
            f"samples={V512_STAGE2_SAMPLE_COUNT}+"
            f"{V512_STAGE3_SAMPLE_COUNT} "
            f"cache_hit={V53_STATS['cache_hits']} "
            f"time={elapsed:.3f}s"
        )

    return final



def explain_score_components(result):
    """
    V5.12-fixed2-fixed 离线检查辅助：
      CurrentGain = 当前交换收益 + 当前交换后连锁期望
      FinalScore  = CurrentGain + 下一步交换收益期望

    下一步交换后的连锁收益不参与最终评分。
    board_potential 不参与最终能量评分。
    """
    immediate = float(result.get("immediate_gain", 0.0))
    current_chain = float(result.get("current_chain_gain", 0.0))
    next_gain = float(result.get("next_gain", 0.0))

    current_gain = immediate + current_chain
    final_score = current_gain + next_gain

    return {
        "current_gain": current_gain,
        "next_gain": next_gain,
        "final_score": final_score,
    }



def benchmark_swap_count():
    """离线 sanity check：3×9 棋盘任意两格交换必须是 351。"""
    return len(generate_all_swaps())


def print_swap_results_v50_1(results):
    print()
    print("=" * 120)
    print("筛选结果")
    print("=" * 120)

    if not results:
        print("当前没有找到有效交换。")
        return

    for i, result in enumerate(results[:12], 1):
        print(
            f"{i}. {position_name(result['p1'])} <-> "
            f"{position_name(result['p2'])}"
        )

        print(
            f"   当前交换收益 = "
            f"{result.get('immediate_gain', 0.0):.2f}"
        )

        print(
            f"   当前交换后连锁期望 = "
            f"{result.get('current_chain_gain', 0.0):.2f}"
        )

        current_samples = result.get("current_chain_samples", [])
        current_sample_count = result.get(
            "current_sample_count",
            len(current_samples),
        )
        current_strata = result.get("current_chain_strata", {})
        if current_samples:
            print(
                f"   当前连锁样本（{current_sample_count}个） = "
                + ", ".join(f"{x:.2f}" for x in current_samples)
                + " | mean="
                + f"{statistics.fmean(current_samples):.2f}"
            )

        print(
            f"   CurrentGain = "
            f"{result.get('current_gain', 0.0):.2f}"
        )

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

        print(
            f"   board_potential = "
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
    print(
        f"当前交换后连锁期望："
        f"{best.get('current_chain_gain', 0.0):.2f}"
    )

    current_samples = best.get("current_chain_samples", [])
    if current_samples:
        print(
            f"当前连锁样本（{len(current_samples)}个）："
            + ", ".join(f"{x:.2f}" for x in current_samples)
            + " | mean="
            + f"{statistics.fmean(current_samples):.2f}"
        )

    print(
        f"当前完整收益 CurrentGain："
        f"{best.get('current_gain', 0.0):.2f}"
    )

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
    print(
        f"board_potential（仅辅助）："
        f"{best.get('potential', 0.0):.2f}"
    )
    print(
        f"最终评分："
        f"{best.get('final_score', best.get('score', 0.0)):.2f}"
    )

    print()
    print("当前交换后产生：")

    for group in best["matches"]:
        print_match_group(
            group,
            best["board"]
        )



def analyze_board_v40(board, x1, y1, dx, dy):
    board_state = make_board_state(board)

    print_board_state(board_state)

    results = analyze_all_swaps_v50_1(
        board_state,
        current_step=1
    )

    print_swap_results(
        results
    )

    print_best_swap(
        results
    )

    if results:
        print_simulated_board(
            results[0]
        )
    click_best_swap_v43(results, x1, y1, dx, dy)

    return results


def main():

    print()
    print("=" * 90)
    print("三打白骨精")
    print("=" * 90)

    print()
    print("F8：暂停 / 继续")
    print("F9：退出")

    # ========================================================
    # DEBUG
    # ========================================================

    if DEBUG_SAVE_IMAGES:
        os.makedirs(
            DEBUG_DIR,
            exist_ok=True
        )

    # ========================================================
    # 校准
    # ========================================================

    x1, y1, dx, dy = \
        calibrate()


    cell_w = abs(dx) * 0.90

    cell_h = abs(dy) * 0.90


    print()

    print(
        f"识别格子尺寸："
        f"{cell_w:.2f} × "
        f"{cell_h:.2f}"
    )


    # ========================================================
    # 准备截图
    # ========================================================

    print()
    print(
        "准备好后按 Enter 开始..."
    )

    input()


    # ========================================================
    # MSS
    # ========================================================

    with mss.MSS() as sct:

        monitor = sct.monitors[0]

        raw = np.array(
            sct.grab(
                monitor
            )
        )


    screenshot = cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2BGR
    )


    # ========================================================
    # 保存原始截图
    # ========================================================

    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(
        os.path.join(
            DEBUG_DIR,
            "screen.png"
        ),
        screenshot
    )


    board = []

    detailed = []


    # ========================================================
    # 逐格识别
    # ========================================================

    for row in range(ROWS):

        board_row = []

        detail_row = []


        for col in range(COLS):

            cx = (

                x1 +
                col * dx
            )


            cy = (

                y1 +
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

                    "width_ratio": 0,

                    "height_ratio": 0,

                    "diameter": 0,

                    "confidence": 0
                }


                data = {}


            else:

                result, data = \
                    recognize_cell(

                        cell,

                        row,

                        col,

                        DEBUG_DIR
                    )


            board_row.append(
                result
            )


            detail_row.append(
                (
                    result,
                    data
                )
            )


        board.append(
            board_row
        )


        detailed.append(
            detail_row
        )


# ========================================================
    # 绘制最终结果
    # ========================================================

    result_img = screenshot.copy()

    for row in range(ROWS):
        for col in range(COLS):
            result = board[row][col]

            cx = int(round(x1 + col * dx))
            cy = int(round(y1 + row * dy))

            t = result["type"]
            color = DRAW_COLOR[t]

            half_w = int(cell_w / 2)
            half_h = int(cell_h / 2)

            cv2.rectangle(
                result_img,
                (cx - half_w, cy - half_h),
                (cx + half_w, cy + half_h),
                color,
                2
            )

            cv2.circle(
                result_img,
                (cx, cy),
                3,
                (0, 255, 0),
                -1
            )

            text = (
                f"{result['level']}{t}"
                if t != "?"
                else "?"
            )

            cv2.putText(
                result_img,
                text,
                (cx - 15, cy - half_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # 自动运行
    # ========================================================

    run_auto_loop_v43(
        x1,
        y1,
        dx,
        dy
    )

    # ========================================================
    # 完成
    # ========================================================

    print()
    print("=" * 90)
    print("识别完成")
    print("=" * 90)

    print()
    print(f"调试目录：{os.path.abspath(DEBUG_DIR)}")


# ============================================================
# 程序入口
# ============================================================

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
