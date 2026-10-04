"""标题编号模式识别。

支持中文学术论文常见的标题编号格式：
1. 阿拉伯数字多级：1, 1.1, 1.1.1, 2.3.1
2. 阿拉伯数字单级：1, 2, 3（带或不带点号）
3. 括号数字：(1), （1）, 1)
4. 罗马数字：I, II, III, IV / i, ii, iii
5. 字母：A, B, C / a, b, c
6. 中文数字：一、, 二、, 三、

为避免误判英文句子首字母，对罗马数字/字母格式加了约束：
- 行长度不超过 80 字符
- 编号后必须跟至少一个空格
- 单字符编号（如 I, A）后不能紧跟小写字母
"""

import re
from typing import Callable


def _is_roman(s: str) -> bool:
    """判断字符串是否是合法罗马数字。"""
    pattern = r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    return bool(re.match(pattern, s.upper())) and s.strip() != ""


# 模式定义：(正则, 层级提取函数, 优先级)
# 优先级数字越小越先匹配
_PATTERNS: list[tuple[re.Pattern, Callable, int]] = [
    # 1. 阿拉伯数字多级：1.1, 1.1.1, 2.3.1（最可靠，最高优先级）
    (
        re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)$"),
        lambda m: (len(m.group(1).split(".")), m.group(2).strip()),
        1,
    ),
    # 2. 阿拉伯数字带点单级：1. 2. 3.
    (
        re.compile(r"^\s*(\d+)\.\s+(.*)$"),
        lambda m: (1, m.group(2).strip()),
        2,
    ),
    # 2b. 阿拉伯数字不带点单级：1 引言  2 方法
    (
        re.compile(r"^\s*(\d+)\s+(.*)$"),
        lambda m: (1, m.group(2).strip()),
        2,
    ),
    # 3. 括号数字：(1) （1） 1)
    (
        re.compile(r"^\s*[(（](\d+)[)）]\s*(.*)$"),
        lambda m: (1, m.group(2).strip()),
        3,
    ),
    (
        re.compile(r"^\s*(\d+)\)\s+(.*)$"),
        lambda m: (1, m.group(2).strip()),
        3,
    ),
    # 4. 中文数字：一、 二、 三、
    (
        re.compile(r"^\s*([一二三四五六七八九十百千]+)、\s*(.*)$"),
        lambda m: (1, m.group(2).strip()),
        4,
    ),
]


def match_heading(line: str) -> tuple[int, str] | None:
    """尝试识别一行是否为标题。

    Args:
        line: 一行文本（不带换行符）

    Returns:
        (层级, 标题文本) 或 None（不是标题）
    """
    stripped = line.strip()
    if not stripped:
        return None

    # 先尝试高优先级模式（多级编号等，无额外约束）
    for pattern, extractor, priority in _PATTERNS:
        if priority <= 4:
            m = pattern.match(stripped)
            if m:
                level, title = extractor(m)
                if title:
                    return (level, title)

    # 罗马数字 / 字母：需要额外约束避免误判
    if len(stripped) > 80:
        return None

    # 罗马数字大写：I II III IV
    m = re.match(r"^(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\s+(.*)$", stripped)
    if m and _is_roman(m.group(1)) and len(m.group(1)) >= 1:
        level = 1
        title = m.group(2).strip()
        if title:
            return (level, title)

    # 罗马数字小写：i ii iii iv
    m = re.match(r"^(m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3}))\s+(.*)$", stripped)
    if m and _is_roman(m.group(1)):
        title = m.group(2).strip()
        if title:
            return (1, title)

    # 大写字母：A B C（单字符需避免误判，要求后面跟空格+内容）
    m = re.match(r"^([A-Z])\s+(.*)$", stripped)
    if m:
        letter = m.group(1)
        title = m.group(2).strip()
        if title and not title[0].islower():
            return (1, title)

    # 小写字母：a b c
    m = re.match(r"^([a-z])\s+(.*)$", stripped)
    if m:
        title = m.group(2).strip()
        if title:
            return (1, title)

    return None