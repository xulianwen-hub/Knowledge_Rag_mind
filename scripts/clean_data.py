"""
M0-5 数据清洗脚本。

功能：
1. 扫描 data/raw/ 下所有文件
2. 计算 SHA256 hash，识别重复
3. 生成 manifest.json（文件清单 + hash）
4. 按计划书要求：保留重复样本用于测 hash 去重
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knowresearch.config import settings


def compute_hash(filepath: Path) -> str:
    """计算文件 SHA256 hash。"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    raw_dir = Path(settings.data.raw_dir).resolve()
    processed_dir = Path(settings.data.processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 扫描所有文件
    all_files = sorted(
        [f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    )

    print(f"扫描到 {len(all_files)} 个文件")

    # 计算 hash
    file_info = []
    hash_map = {}

    for fp in all_files:
        fhash = compute_hash(fp)
        file_info.append({
            "filename": fp.name,
            "hash": fhash,
            "size_bytes": fp.stat().st_size,
            "suffix": fp.suffix.lower(),
        })
        if fhash in hash_map:
            hash_map[fhash].append(fp.name)
        else:
            hash_map[fhash] = [fp.name]

    # 统计
    duplicates = {h: names for h, names in hash_map.items() if len(names) > 1}
    print(f"唯一文件: {len(hash_map)}")
    print(f"重复组: {len(duplicates)}")
    for h, names in duplicates.items():
        print(f"  重复 (hash={h[:12]}...): {names}")

    # 复制到 processed/
    print(f"\n复制到 {processed_dir} ...")
    for fp in all_files:
        dest = processed_dir / fp.name
        if not dest.exists():
            shutil.copy2(fp, dest)

    # 写 manifest
    manifest_path = processed_dir / "manifest.json"
    manifest = {
        "total_files": len(all_files),
        "unique_hashes": len(hash_map),
        "duplicate_groups": len(duplicates),
        "duplicates": {h[:12]: names for h, names in duplicates.items()},
        "files": file_info,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest 已写入: {manifest_path}")
    print("清洗完成。")


if __name__ == "__main__":
    main()