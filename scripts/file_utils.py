"""
File Utilities for Triton Operator Generation System

Provides common file operations for skill communication.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import shutil


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(filepath: Union[str, Path], data: Dict[str, Any], indent: int = 2):
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_json(filepath: Union[str, Path]) -> Optional[Dict]:
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(filepath: Union[str, Path], content: str):
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def read_text(filepath: Union[str, Path]) -> Optional[str]:
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def append_text(filepath: Union[str, Path], content: str):
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content)


def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_dir(src: Union[str, Path], dst: Union[str, Path]):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def delete_file(filepath: Union[str, Path]):
    filepath = Path(filepath)
    if filepath.exists():
        filepath.unlink()


def delete_dir(dirpath: Union[str, Path]):
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(dirpath)


def list_files(dirpath: Union[str, Path], pattern: str = "*") -> list:
    dirpath = Path(dirpath)
    if not dirpath.exists():
        return []
    return list(dirpath.glob(pattern))


def file_exists(filepath: Union[str, Path]) -> bool:
    return Path(filepath).exists()


def get_relative_path(filepath: Union[str, Path], base: Union[str, Path]) -> str:
    return str(Path(filepath).relative_to(Path(base)))


def resolve_path(filepath: Union[str, Path], base: Union[str, Path] = None) -> Path:
    p = Path(filepath)
    if p.is_absolute():
        return p
    if base:
        return (Path(base) / p).resolve()
    return p.resolve()
