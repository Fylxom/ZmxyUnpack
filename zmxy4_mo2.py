#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mo2 资源提取器
================

根据 AssetLoader.as 中的 _hand_fz() / _parse_fz2() 重构。

用法：
    python extract_mo2.py cg1.swf

输出：
    cg1_extracted/
        manifest.txt
        0001_<name>.<ext>
        0002_<name>.<ext>
        ...

说明：
- mo2:
    3 bytes  "mo2"
    2 bytes  UInt16 chunk_count (big endian)
    chunk_count * 4 bytes Int32 chunk_length (big endian)
    后面依次存放 chunk
- 每个 chunk:
    raw DEFLATE 压缩
    解压后：
        UInt16 name_length
        name_length bytes: AS3 ByteArray.writeUTF() 格式
        UInt16 data_type
        剩余数据
- data_type == 1:
    原 AS3 调用 readObject()，这里先原样保存 AMF3 payload。
    本脚本不会伪造 AMF3 对象。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import zlib
from pathlib import Path


def u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("读取 UInt16 时超出文件范围")
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def i32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("读取 Int32 时超出文件范围")
    return struct.unpack_from(">i", data, offset)[0], offset + 4


def raw_inflate(data: bytes) -> bytes:
    """
    ActionScript ByteArray.inflate() 对应 raw DEFLATE。
    """
    try:
        return zlib.decompress(data, -zlib.MAX_WBITS)
    except zlib.error:
        # 有些资源可能实际是 zlib wrapper。
        try:
            return zlib.decompress(data)
        except zlib.error as e:
            raise ValueError(f"DEFLATE 解压失败: {e}") from e


def sanitize_name(name: str) -> str:
    """
    防止资源名称包含 Windows 非法字符或路径穿越。
    """
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    name = name.strip().strip(".")
    return name or "unnamed"


def detect_extension(data: bytes, data_type: int) -> str:
    """
    根据内容判断比较常见的资源类型。
    """
    if data.startswith((b"FWS", b"CWS", b"ZWS")):
        return ".swf"

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"

    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE":
        return ".wav"

    if data.startswith(b"OggS"):
        return ".ogg"

    if data.startswith(b"ID3"):
        return ".mp3"

    if data.startswith(b"PK\x03\x04"):
        return ".zip"

    if data.startswith(b"<?xml") or data.lstrip().startswith(b"<"):
        return ".xml"

    if data_type == 1:
        return ".amf3"

    return ".bin"


def swf_info(data: bytes) -> str:
    if len(data) < 3:
        return "不是完整 SWF"

    header = data[:3]

    if header == b"FWS":
        return "FWS / 未压缩 SWF"

    if header == b"CWS":
        return "CWS / ZLIB 压缩 SWF"

    if header == b"ZWS":
        return "ZWS / LZMA 压缩 SWF"

    return f"非标准 SWF 头: {header.hex(' ').upper()}"


def parse_mo2(data: bytes):
    if len(data) < 5:
        raise ValueError("文件太短")

    if data[:3] != b"mo2":
        raise ValueError(
            f"不是 mo2 文件，文件头为 {data[:3]!r}"
        )

    offset = 3

    count, offset = u16(data, offset)

    if count == 0:
        raise ValueError("mo2 chunk 数量为 0")

    lengths = []

    for index in range(count):
        length, offset = i32(data, offset)

        if length < 0:
            raise ValueError(
                f"chunk #{index + 1} 长度非法: {length}"
            )

        lengths.append(length)

    header_end = offset
    expected_size = header_end + sum(lengths)

    if expected_size != len(data):
        raise ValueError(
            "mo2 长度表与实际文件大小不一致\n"
            f"文件大小: {len(data)}\n"
            f"计算大小: {expected_size}\n"
            f"chunk 数: {count}\n"
            f"长度表结束位置: {header_end}"
        )

    chunks = []

    for index, length in enumerate(lengths, 1):
        compressed = data[offset:offset + length]
        offset += length

        try:
            decompressed = raw_inflate(compressed)
        except Exception as e:
            raise ValueError(
                f"chunk #{index} 解压失败，"
                f"压缩长度={length}: {e}"
            ) from e

        chunks.append({
            "index": index,
            "compressed_size": length,
            "data": decompressed,
        })

    return count, lengths, chunks


def parse_chunk(chunk: dict) -> dict:
    data = chunk["data"]

    if len(data) < 4:
        raise ValueError("解压后的 chunk 太短")

    offset = 0

    # AssetLoader.as:
    # var _loc5_:uint = _loc1_.readUnsignedShort();
    # _loc6_ = new ByteArray();
    # _loc1_.readBytes(_loc6_,0,_loc5_);
    name_length, offset = u16(data, offset)

    if name_length > len(data) - offset:
        raise ValueError(
            f"name_length={name_length} 超出 chunk 范围"
        )

    name_block = data[offset:offset + name_length]
    offset += name_length

    # ByteArray.writeUTF / readUTF：
    # 前两个字节是 UTF-8 字符串长度。
    if len(name_block) >= 2:
        utf_length = struct.unpack_from(">H", name_block, 0)[0]

        if 2 + utf_length <= len(name_block):
            name_bytes = name_block[2:2 + utf_length]
        else:
            # 容错：某些转换后的资源可能没有内部 UTF 长度。
            name_bytes = name_block
    else:
        name_bytes = b""

    name = name_bytes.decode("utf-8", errors="replace")

    if offset + 2 > len(data):
        raise ValueError("chunk 缺少 data_type")

    data_type, offset = u16(data, offset)

    payload = data[offset:]

    return {
        "index": chunk["index"],
        "name": name,
        "data_type": data_type,
        "payload": payload,
        "compressed_size": chunk["compressed_size"],
        "decompressed_size": len(data),
    }


def unique_path(directory: Path, filename: str) -> Path:
    path = directory / filename

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix

    n = 2

    while True:
        candidate = directory / f"{stem}_{n}{suffix}"

        if not candidate.exists():
            return candidate

        n += 1


def extract(input_file: Path, output_dir: Path) -> int:
    if not input_file.exists():
        print(f"错误：找不到文件：{input_file}")
        return 1

    data = input_file.read_bytes()

    print("=" * 70)
    print("MO2 Resource Extractor")
    print("=" * 70)
    print(f"输入文件: {input_file}")
    print(f"文件大小: {len(data):,} bytes")
    print(f"文件头  : {data[:3]!r}")

    if data[:3] != b"mo2":
        print("\n错误：这个文件不是 mo2 格式。")
        return 2

    try:
        count, lengths, chunks = parse_mo2(data)
    except Exception as e:
        print(f"\n解析 mo2 失败：{e}")
        return 3

    print(f"Chunk 数量: {count}")
    print(f"长度表前 10 项: {lengths[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    manifest.append(f"input={input_file}")
    manifest.append(f"file_size={len(data)}")
    manifest.append(f"format=mo2")
    manifest.append(f"chunk_count={count}")
    manifest.append("")

    success = 0
    failed = 0

    for chunk in chunks:
        index = chunk["index"]

        try:
            item = parse_chunk(chunk)

            name = sanitize_name(item["name"])
            payload = item["payload"]
            data_type = item["data_type"]

            extension = detect_extension(payload, data_type)

            # 名称为空时使用序号。
            if not name:
                name = f"chunk_{index:04d}"

            filename = (
                f"{index:04d}_{name}{extension}"
            )

            output_path = unique_path(
                output_dir,
                filename
            )

            output_path.write_bytes(payload)

            line = (
                f"{index:04d}\t"
                f"name={item['name']!r}\t"
                f"type={data_type}\t"
                f"compressed={item['compressed_size']}\t"
                f"decompressed={item['decompressed_size']}\t"
                f"payload={len(payload)}\t"
                f"format={swf_info(payload) if payload[:3] in (b'FWS', b'CWS', b'ZWS') else extension}\t"
                f"file={output_path.name}"
            )

            manifest.append(line)

            print(
                f"[OK] {index:03d}/{count} "
                f"{item['name']!r} "
                f"type={data_type} "
                f"payload={len(payload):,} "
                f"-> {output_path.name}"
            )

            success += 1

        except Exception as e:
            failed += 1

            print(
                f"[FAIL] {index:03d}/{count}: {e}"
            )

            manifest.append(
                f"{index:04d}\tERROR\t{e}"
            )

    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text(
        "\n".join(manifest),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("提取完成")
    print("=" * 70)
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_dir}")
    print(f"清单: {manifest_path}")

    # 自动检查有没有 SWF。
    swfs = list(output_dir.glob("*.swf"))

    if swfs:
        print("\n发现 SWF:")
        for swf in swfs:
            content = swf.read_bytes()[:3]
            print(
                f"  {swf.name} -> "
                f"{content!r} / {swf_info(content)}"
            )

    if failed:
        print(
            "\n注意：如果大量 chunk 解压成功但解析失败，"
            "很可能是原 AS3 中的 AMF3 结构需要进一步还原。"
        )

    return 0 if success else 4


def main() -> int:
    parser = argparse.ArgumentParser(
        description="提取 AssetLoader.as 使用的 mo2 资源"
    )

    parser.add_argument(
        "input",
        help="输入文件，例如 cg1.swf"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="输出目录，默认：输入文件名_extracted",
        default=None
    )

    args = parser.parse_args()

    input_file = Path(args.input)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_file.with_name(
            input_file.stem + "_extracted"
        )

    try:
        return extract(input_file, output_dir)

    except KeyboardInterrupt:
        print("\n用户取消")
        return 130

    except Exception as e:
        print(
            f"\n未处理异常: "
            f"{type(e).__name__}: {e}"
        )
        return 5


if __name__ == "__main__":
    sys.exit(main())
