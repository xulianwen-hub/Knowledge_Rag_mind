"""ObjectStoragePort 适配器：本地目录实现。

首版用本地文件系统，后续可换 MinIO / S3。
key 就是相对路径，put 时创建目录，get 时读取 bytes。
"""

from pathlib import Path

from knowresearch.core.ports import ObjectStoragePort


class LocalObjectStorage(ObjectStoragePort):
    """基于本地目录的对象存储。"""

    def __init__(self, base_dir: str):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """key → 绝对路径，防止路径穿越。"""
        resolved = (self._base / key).resolve()
        if not str(resolved).startswith(str(self._base.resolve())):
            raise ValueError(f"非法 key: {key}（路径穿越）")
        return resolved

    def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"对象不存在: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._resolve(prefix) if prefix else self._base
        if not base.exists():
            return []
        keys: list[str] = []
        base_resolved = self._base.resolve()
        if base.is_file():
            return [base.relative_to(base_resolved).as_posix()]
        for p in base.rglob("*"):
            if p.is_file():
                keys.append(p.relative_to(base_resolved).as_posix())
        return sorted(keys)