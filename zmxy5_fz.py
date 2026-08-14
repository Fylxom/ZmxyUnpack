#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FZ extractor for the AssetLoader.as format.

This script follows:

    handFz()
      -> parseFz()
      -> readObject()                  # AMF3 root
      -> readObject()                  # each item, returned as ByteArray
      -> ByteArray.inflate()           # raw/zlib DEFLATE
      -> readObject()                  # item metadata
      -> data ByteArray
      -> uncompress("lzma") when ver == 2
         OR
      -> uncompress() when ver != 2

The provided cg1_5.swf starts with AMF3:
    0A 0B 01 ...
which is an AMF3 dynamic Object, not a SWF header.

Usage:
    python extract_fz.py cg1_5.swf

Optional:
    python extract_fz.py cg1_5.swf -o cg1_5_extracted
"""

from __future__ import annotations

import argparse
import json
import lzma
import struct
import sys
import zlib
from pathlib import Path


class AMF3Error(Exception):
    pass


class ByteReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise AMF3Error(
                f"读取 {n} bytes 超出范围: "
                f"pos={self.pos}, size={len(self.data)}"
            )
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.read(1)[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def u29(self) -> int:
        """
        AMF3 U29。
        前最多 3 个字节每个贡献 7 bit，
        第 4 个字节贡献 8 bit。
        """
        value = 0

        for _ in range(3):
            b = self.u8()
            if b & 0x80:
                value = (value << 7) | (b & 0x7F)
            else:
                return (value << 7) | b

        b = self.u8()
        return (value << 8) | b

    def string_raw(self, n: int) -> str:
        return self.read(n).decode("utf-8", errors="replace")


class AMF3Decoder:
    """
    实现 FZ 所需的 AMF3 子集。

    支持：
        Undefined
        Null
        False / True
        Integer
        Double
        String
        Date
        Array
        Object
        ByteArray

    对游戏资源来说最关键的是：
        Object + dynamic properties + ByteArray
    """

    def __init__(self):
        self.strings: list[str] = []
        self.objects: list[object] = []
        self.traits: list[dict] = []

    @staticmethod
    def u29_inline(raw: int) -> bool:
        return bool(raw & 1)

    def read_string(self, r: ByteReader) -> str:
        raw = r.u29()

        # low bit 0 = string reference
        if not (raw & 1):
            index = raw >> 1
            if index >= len(self.strings):
                raise AMF3Error(
                    f"字符串引用越界: {index}, "
                    f"已有 {len(self.strings)} 个字符串"
                )
            return self.strings[index]

        length = raw >> 1

        if length == 0:
            return ""

        value = r.string_raw(length)
        self.strings.append(value)
        return value

    def read_bytearray(self, r: ByteReader) -> bytes:
        raw = r.u29()

        if not (raw & 1):
            index = raw >> 1
            if index >= len(self.objects):
                raise AMF3Error(
                    f"ByteArray 对象引用越界: {index}"
                )
            obj = self.objects[index]

            if not isinstance(obj, (bytes, bytearray)):
                raise AMF3Error(
                    f"对象引用 {index} 不是 ByteArray"
                )

            return bytes(obj)

        length = raw >> 1
        data = r.read(length)

        self.objects.append(data)
        return data

    def read_object(self, r: ByteReader):
        marker = r.u8()

        # Undefined
        if marker == 0x00:
            return None

        # Null
        if marker == 0x01:
            return None

        # False
        if marker == 0x02:
            return False

        # True
        if marker == 0x03:
            return True

        # Integer
        if marker == 0x04:
            value = r.u29()

            # AMF3 signed 29-bit integer
            if value & 0x10000000:
                value -= 0x20000000

            return value

        # Double
        if marker == 0x05:
            return struct.unpack(">d", r.read(8))[0]

        # String
        if marker == 0x06:
            return self.read_string(r)

        # XMLDoc
        if marker == 0x07:
            return self.read_string(r)

        # Date
        if marker == 0x08:
            raw = r.u29()

            if not (raw & 1):
                index = raw >> 1
                return {
                    "__amf_type__": "date_ref",
                    "index": index,
                }

            milliseconds = struct.unpack(">d", r.read(8))[0]
            obj = {
                "__amf_type__": "date",
                "milliseconds": milliseconds,
            }
            self.objects.append(obj)
            return obj

        # Array
        if marker == 0x09:
            raw = r.u29()

            if not (raw & 1):
                index = raw >> 1
                return self.objects[index]

            dense_count = raw >> 1

            assoc = {}

            while True:
                key = self.read_string(r)
                if key == "":
                    break
                assoc[key] = self.read_object(r)

            dense = [
                self.read_object(r)
                for _ in range(dense_count)
            ]

            obj = {
                "__amf_type__": "array",
                "associative": assoc,
                "dense": dense,
            }

            self.objects.append(obj)
            return obj

        # Object
        if marker == 0x0A:
            raw = r.u29()

            # Object reference
            if not (raw & 1):
                index = raw >> 1

                if index >= len(self.objects):
                    raise AMF3Error(
                        f"对象引用越界: {index}"
                    )

                return self.objects[index]

            # Inline traits
            if raw & 0x02:
                dynamic = bool(raw & 0x08)
                sealed_count = raw >> 4

                class_name = self.read_string(r)

                sealed_names = [
                    self.read_string(r)
                    for _ in range(sealed_count)
                ]

                traits = {
                    "class_name": class_name,
                    "sealed_names": sealed_names,
                    "dynamic": dynamic,
                }

                self.traits.append(traits)
            else:
                trait_index = raw >> 2

                if trait_index >= len(self.traits):
                    raise AMF3Error(
                        f"Trait 引用越界: {trait_index}"
                    )

                traits = self.traits[trait_index]

            obj = {
                "__amf_type__": "object",
                "__class__": traits["class_name"],
            }

            # Important: append before reading properties,
            # because AMF3 objects can self-reference.
            self.objects.append(obj)

            for name in traits["sealed_names"]:
                obj[name] = self.read_object(r)

            if traits["dynamic"]:
                while True:
                    name = self.read_string(r)

                    if name == "":
                        break

                    obj[name] = self.read_object(r)

            return obj

        # XML
        if marker == 0x0B:
            return self.read_string(r)

        # ByteArray
        if marker == 0x0C:
            return self.read_bytearray(r)

        raise AMF3Error(
            f"未知 AMF3 marker: 0x{marker:02X} "
            f"at offset {r.pos - 1}"
        )


def raw_inflate(data: bytes) -> bytes:
    """
    AS3 ByteArray.inflate() 通常对应 raw DEFLATE。
    如果不是 raw，则尝试普通 zlib wrapper。
    """
    try:
        return zlib.decompress(data, -zlib.MAX_WBITS)
    except zlib.error:
        try:
            return zlib.decompress(data)
        except zlib.error as e:
            raise ValueError(f"DEFLATE 解压失败: {e}") from e


def uncompress_as3(data: bytes, mode: str) -> bytes:
    """
    对应 ByteArray.uncompress():

        ver == 2 -> uncompress("lzma")
        其他     -> uncompress()

    Python lzma 首先尝试 .lzma/ALONE 格式，
    再尝试 XZ。
    """
    if mode == "lzma":
        errors = []

        for fmt in (lzma.FORMAT_ALONE, lzma.FORMAT_XZ):
            try:
                return lzma.decompress(data, format=fmt)
            except lzma.LZMAError as e:
                errors.append(str(e))

        # 最后尝试 raw LZMA，常见参数组合。
        for lc, lp, pb in (
            (3, 0, 2),
            (3, 0, 0),
            (2, 0, 2),
        ):
            try:
                return lzma.decompress(
                    data,
                    format=lzma.FORMAT_RAW,
                    filters=[{
                        "id": lzma.FILTER_LZMA1,
                        "dict_size": 8 * 1024 * 1024,
                        "lc": lc,
                        "lp": lp,
                        "pb": pb,
                    }],
                )
            except lzma.LZMAError as e:
                errors.append(str(e))

        raise ValueError(
            "LZMA 解压失败；尝试的格式均不匹配: "
            + " | ".join(errors[:3])
        )

    # ByteArray.uncompress() 默认是 zlib。
    try:
        return zlib.decompress(data)
    except zlib.error:
        # 容错：部分资源可能使用 raw DEFLATE。
        return zlib.decompress(data, -zlib.MAX_WBITS)


def json_safe(value):
    """
    把 bytes 转换成 JSON 可保存的结构。
    """
    if isinstance(value, (bytes, bytearray)):
        return {
            "__amf_type__": "bytearray",
            "length": len(value),
            "hex_head": bytes(value[:64]).hex(" "),
        }

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


def clean_json_value(value):
    """
    将 AMF3 内部标记去掉，生成更接近普通 JSON 的结果。
    普通 Object -> dict
    AMF3 Array  -> dense 数组；有 associative 属性时保留为对象
    ByteArray  -> 保留为 {"__bytearray__": true, ...}
    """
    if isinstance(value, dict):
        amf_type = value.get("__amf_type__")

        if amf_type == "object":
            return {
                str(k): clean_json_value(v)
                for k, v in value.items()
                if not str(k).startswith("__")
            }

        if amf_type == "array":
            assoc = value.get("associative", {})
            dense = value.get("dense", [])

            if assoc:
                result = {
                    str(k): clean_json_value(v)
                    for k, v in assoc.items()
                }
                result["_dense"] = [
                    clean_json_value(v) for v in dense
                ]
                return result

            return [clean_json_value(v) for v in dense]

        if amf_type == "bytearray":
            return {
                "__bytearray__": True,
                "length": value.get("length", 0),
                "hex_head": value.get("hex_head", ""),
            }

        return {
            str(k): clean_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [clean_json_value(v) for v in value]

    return value


def safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")

    name = "".join(
        "_" if ord(ch) < 32 else ch
        for ch in name
    )

    name = name.strip().strip(".")

    return name or "unnamed"


def decode_amf(data: bytes):
    decoder = AMF3Decoder()
    reader = ByteReader(data)

    value = decoder.read_object(reader)

    return value, reader.pos


def find_bytearray(value):
    """
    从 metadata object 中递归寻找 data 字段。
    """
    if isinstance(value, dict):
        if "data" in value and isinstance(
            value["data"],
            (bytes, bytearray)
        ):
            return bytes(value["data"])

        for v in value.values():
            found = find_bytearray(v)
            if found is not None:
                return found

    if isinstance(value, list):
        for v in value:
            found = find_bytearray(v)
            if found is not None:
                return found

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract FZ/AMF3 resources from cg1_5.swf"
    )

    parser.add_argument(
        "input",
        help="输入文件，例如 cg1_5.swf"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出目录，默认：<输入名>_extracted"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"错误：找不到文件 {input_path}")
        return 1

    output_dir = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem + "_extracted"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    data = input_path.read_bytes()

    print("=" * 70)
    print("FZ / AMF3 Resource Extractor + Final AMF3→JSON")
    print("=" * 70)
    print(f"输入文件 : {input_path}")
    print(f"文件大小 : {len(data):,} bytes")
    print(
        "文件头   :",
        data[:32].hex(" ")
    )

    # ---------------------------------------------------------
    # 对应 parseFz():
    #
    # var obj = byte.readObject();
    # var fn = int(obj.fn);
    # ---------------------------------------------------------

    try:
        root, consumed = decode_amf(data)
    except Exception as e:
        print(f"\n根 AMF3 对象解析失败：{e}")
        return 2

    if not isinstance(root, dict):
        print(
            f"\n根对象不是 Object，而是 {type(root).__name__}"
        )
        return 3

    fn = root.get("fn")

    if not isinstance(fn, int):
        print("\n根对象没有可用的 fn 字段。")
        print("解析结果：")
        print(json.dumps(
            json_safe(root),
            ensure_ascii=False,
            indent=2
        )[:5000])
        return 4

    print(f"\nAMF3 根对象解析成功")
    print(f"fn = {fn}")
    print(f"根对象消耗字节 = {consumed}")

    # 保存根对象摘要
    (output_dir / "root.json").write_text(
        json.dumps(
            json_safe(root),
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # parseFz() 中随后继续对同一个 ByteArray：
    #   _loc3_ = _loc4_.readObject()
    reader = ByteReader(data)
    decoder = AMF3Decoder()

    # 重新读取根对象，使 decoder 的 reference table 与
    # 后续连续 readObject() 保持一致。
    root = decoder.read_object(reader)

    success = 0
    failed = 0
    manifest = []

    print("\n开始提取资源...")
    print("-" * 70)

    for index in range(1, fn + 1):
        try:
            item_container = decoder.read_object(reader)

            if not isinstance(
                item_container,
                (bytes, bytearray)
            ):
                raise ValueError(
                    f"第 {index} 项 readObject() "
                    f"不是 ByteArray，而是 "
                    f"{type(item_container).__name__}"
                )

            compressed_item = bytes(item_container)

            # AssetLoader.as:
            # ByteArray(_loc3_).inflate();
            item_data = raw_inflate(compressed_item)

            # _loc6_ = ByteArray(_loc3_).readObject();
            metadata, metadata_consumed = decode_amf(item_data)

            if not isinstance(metadata, dict):
                raise ValueError(
                    "解压后的对象不是 AMF3 Object"
                )

            prefix = str(
                metadata.get("prefixName", "")
            )

            version = metadata.get("ver")

            name = str(
                metadata.get("name", "")
            )

            payload = find_bytearray(metadata)

            if payload is None:
                raise ValueError(
                    "metadata 中没有找到 data ByteArray"
                )

            if version == 2:
                payload_out = uncompress_as3(
                    payload,
                    "lzma"
                )
                mode = "lzma"
            else:
                payload_out = uncompress_as3(
                    payload,
                    "zlib"
                )
                mode = "zlib"

            base_name = (
                name
                or prefix
                or f"item_{index:04d}"
            )

            base_name = safe_filename(base_name)

            # ---------------------------------------------------------
            # 最终 payload 可能仍然是 AMF3。
            #
            # 你上传的 Question.json 就属于这种情况：
            # 文件名虽然是 .json，但文件内容以 0x0A 开头，
            # 实际上是 AMF3 Object。
            #
            # 这里再做一次 AMF3 解码，并转换成真正的 JSON。
            # ---------------------------------------------------------
            final_value = None
            final_is_amf3 = False
            final_consumed = 0

            # 不再只检查第一个字节。
            # 对 .json 资源直接尝试完整 AMF3 解码：
            # 只有“成功解码 + 完整消费全部数据”才判定为 AMF3。
            #
            # 这样可以处理：
            #   0A ...            AMF3 Object
            #   09 ...            AMF3 Array
            #   06 ...            AMF3 String
            # 等情况。
            try:
                candidate, consumed = decode_amf(payload_out)

                if consumed == len(payload_out) and isinstance(
                    candidate, (dict, list)
                ):
                    final_value = candidate
                    final_consumed = consumed
                    final_is_amf3 = True
            except Exception:
                pass

            # 根据实际最终数据判断扩展名。
            if final_is_amf3:
                ext = ".json"
            elif payload_out.startswith(
                (b"FWS", b"CWS", b"ZWS")
            ):
                ext = ".swf"
            elif payload_out.startswith(
                b"\\x89PNG\\r\\n\\x1a\\n"
            ):
                ext = ".png"
            elif payload_out.startswith(b"<?xml") or \
                    payload_out.lstrip().startswith(b"<"):
                ext = ".xml"
            elif payload_out.startswith(b"{") or \
                    payload_out.startswith(b"["):
                ext = ".json"
            else:
                ext = ".bin"

            # 保留 metadata 中的原始 name 作为主要文件名。
            # 如果 name 已经包含扩展名，不重复添加。
            if "." in base_name.split("/")[-1]:
                # 对于原名是 .json 但内容是 AMF3 的情况，
                # 仍然使用 .json。
                filename = f"{index:04d}_{base_name}"
            else:
                filename = f"{index:04d}_{base_name}{ext}"

            # 如果最终识别为 AMF3，则保存真正可读的 JSON。
            if final_is_amf3:
                json_path = output_dir / filename

                json_path.write_text(
                    json.dumps(
                        clean_json_value(final_value),
                        ensure_ascii=False,
                        indent=2
                    ),
                    encoding="utf-8"
                )

                print(
                    f"[AMF3→JSON] "
                    f"{index:03d}/{fn} "
                    f"decoded={final_consumed:,}/{len(payload_out):,} "
                    f"-> {json_path.name}"
                )
            else:
                out_path = output_dir / filename
                out_path.write_bytes(payload_out)

            manifest.append({
                "index": index,
                "prefixName": prefix,
                "version": version,
                "name": name,
                "compressed_item_size": len(compressed_item),
                "inflated_item_size": len(item_data),
                "payload_size": len(payload),
                "output_size": len(payload_out),
                "final_format": "amf3->json" if final_is_amf3 else ext.lstrip("."),
                "compression": mode,
                "output": filename,
            })

            print(
                f"[OK] {index:03d}/{fn} "
                f"name={name!r} "
                f"ver={version} "
                f"payload={len(payload_out):,} "
                f"-> {filename}"
            )

            success += 1

        except Exception as e:
            print(
                f"[FAIL] {index:03d}/{fn}: {e}"
            )

            manifest.append({
                "index": index,
                "error": str(e),
            })

            failed += 1

    # ---------------------------------------------------------
    # 第二保险：处理已经存在于输出目录中的 AMF3 .json。
    #
    # 如果用户之前运行旧版脚本，目录里可能已经存在：
    #     Question.json
    # 但文件实际上还是：
    #     0A 0B 01 ...
    #
    # 这里自动扫描并转换这些旧文件，不需要重新提取整个 SWF。
    # 直接覆盖原 .json 文件，不生成 .amf3 备份。
    # ---------------------------------------------------------
    converted_existing = 0

    for existing in output_dir.glob("*.json"):
        if existing.name in {"manifest.json", "root.json"}:
            continue

        try:
            raw_existing = existing.read_bytes()

            if not raw_existing:
                continue

            value_existing, consumed_existing = decode_amf(
                raw_existing
            )

            if consumed_existing != len(raw_existing):
                continue

            if not isinstance(value_existing, (dict, list)):
                continue

            # 直接覆盖原 JSON，不生成 .amf3 备份。
            existing.write_text(
                json.dumps(
                    clean_json_value(value_existing),
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            converted_existing += 1
            print(
                f"[旧文件 AMF3→JSON] "
                f"{existing.name}"
            )

        except Exception:
            continue

    if converted_existing:
        print(
            f"\n额外转换旧 AMF3 JSON 文件: "
            f"{converted_existing}"
        )

    # 检查有没有未消费数据。
    remaining = reader.remaining()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "size": len(data),
                "fn": fn,
                "success": success,
                "failed": failed,
                "remaining_bytes": remaining,
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"fn       : {fn}")
    print(f"成功     : {success}")
    print(f"失败     : {failed}")
    print(f"剩余字节 : {remaining}")
    print(f"输出目录 : {output_dir}")

    if success == fn and remaining == 0:
        print("\n所有资源均成功解析。")
        return 0

    if success > 0:
        print(
            "\n已经成功提取部分资源。"
            "如果 LZMA 部分失败，请把具体 [FAIL] 输出发给我。"
        )
        return 0

    return 5


if __name__ == "__main__":
    sys.exit(main())
