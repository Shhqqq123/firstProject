from __future__ import annotations

import hashlib
import hmac
import os


def hash_password(password: str, iterations: int = 120_000) -> str:
    """把明文密码转换成可存储的哈希字符串。

    格式：
    pbkdf2_sha256$迭代次数$盐值hex$摘要hex

    为什么不用简单 md5/sha256？
    - PBKDF2 会做很多次迭代，显著增加暴力破解成本。
    - 每个密码都有随机 salt，避免彩虹表攻击。
    """
    # 每个密码单独生成随机盐，长度 16 字节够用。
    salt = os.urandom(16)
    # PBKDF2-HMAC-SHA256：常见且兼容性好。
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """校验用户输入密码是否与数据库中的哈希匹配。"""
    # 先按约定格式拆分。
    try:
        algo, iter_text, salt_hex, digest_hex = encoded.split("$", 3)
    except ValueError:
        return False

    # 只接受当前算法，避免未知格式被误判通过。
    if algo != "pbkdf2_sha256":
        return False

    try:
        iterations = int(iter_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False

    # 用同样参数重新计算摘要。
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    # 使用常量时间比较，降低时序攻击风险。
    return hmac.compare_digest(actual, expected)
