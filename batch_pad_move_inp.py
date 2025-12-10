import os
import re

BASE_INP  = r"D:\inp\Job-time2.inp"
OUT_DIR   = r"D:\inpout"
INST_A    = "dianban-1"
INST_B    = "dianban-1-lin-2-1"
COMBOS    = [(0.0, 0.0), (100.0, 0.0), (0.0, -100.0), (100.0, -100.0)]

# 匹配 *Instance 头
INST_HDR_RE = re.compile(r'^\s*\*Instance\s*,\s*name\s*=\s*([^,\s]+)\s*,\s*part\s*=\s*([^,\s]+)', re.IGNORECASE)
# 匹配数值 token（不改变其它字符与逗号、空格）
NUM_RE = re.compile(r'([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)')

def bump_kth_numbers(line: str, which_idx, dz: float) -> str:
    """
    仅对行内第 which_idx 个“数值 token”加 dz，保持原格式/小数位。which_idx 从 1 开始计数。
    """
    if dz == 0.0:
        return line
    tokens = list(NUM_RE.finditer(line))
    if not tokens or which_idx < 1 or which_idx > len(tokens):
        return line

    m = tokens[which_idx - 1]
    old_txt = m.group(1)
    try:
        old_val = float(old_txt)
    except ValueError:
        return line
    new_val = old_val + dz

    # 按原格式回写
    if ('e' in old_txt.lower()):
        new_txt = f"{new_val}"
    elif '.' in old_txt:
        # 保留原小数位数
        frac = old_txt.split('.', 1)[1]
        new_txt = f"{new_val:.{len(frac)}f}"
        # 若原本以 '.' 结尾（如 "124."），保持这个点
        if old_txt.endswith('.') and not new_txt.endswith('.'):
            if new_txt.endswith('.0'):
                new_txt = new_txt[:-2] + '.'
            else:
                new_txt = new_txt + '.'
    else:
        # 原来是整数
        if abs(new_val - round(new_val)) < 1e-12:
            new_txt = str(int(round(new_val)))
        else:
            new_txt = f"{new_val}"

    return line[:m.start()] + new_txt + line[m.end():]

def make_variant(lines_with_eol, dzA, dzB):
    out = []
    in_assembly = False
    current = None
    dz_cur = 0.0
    stage = 0   # 0=非坐标；1=第一行(只改第3个数)；2=第二行(改第3和第6个数)

    for raw in lines_with_eol:
        # 拆出内容与原行尾
        if raw.endswith('\r\n'):
            content, eol = raw[:-2], '\r\n'
        elif raw.endswith('\n'):
            content, eol = raw[:-1], '\n'
        else:
            content, eol = raw, ''

        low = content.strip().lower()

        # 进入/退出 Assembly
        if low.startswith('*assembly'):
            in_assembly = True
            out.append(content + eol)
            continue
        if low.startswith('*end assembly'):
            in_assembly = False
            current, dz_cur, stage = None, 0.0, 0
            out.append(content + eol)
            continue

        if in_assembly:
            m = INST_HDR_RE.match(content)
            if m:
                inst_name = m.group(1)
                current = inst_name
                if inst_name == INST_A and abs(dzA) > 0:
                    dz_cur = dzA; stage = 1
                elif inst_name == INST_B and abs(dzB) > 0:
                    dz_cur = dzB; stage = 1
                else:
                    dz_cur = 0.0; stage = 0
                out.append(content + eol)
                continue

            if current and dz_cur != 0.0 and stage in (1, 2):
                # 只改“数值 token 序号”的第3 / (第3和第6)
                if stage == 1:
                    new_content = bump_kth_numbers(content, 3, dz_cur)
                    out.append(new_content + eol)
                    stage = 2
                    continue
                else:
                    new_content = bump_kth_numbers(content, 3, dz_cur)
                    new_content = bump_kth_numbers(new_content, 6, dz_cur)
                    out.append(new_content + eol)
                    stage = 0
                    continue

            if low.startswith('*end instance'):
                current, dz_cur, stage = None, 0.0, 0
                out.append(content + eol)
                continue

        # 其他行：原样只写一次
        out.append(content + eol)

    return out

if __name__ == "__main__":
    if not os.path.isfile(BASE_INP):
        raise FileNotFoundError(f"找不到基准 INP：{BASE_INP}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 按原样读入（包含原始行尾）
    with open(BASE_INP, "r", encoding="utf-8", newline='') as f:
        src = f.readlines()

    base = os.path.splitext(os.path.basename(BASE_INP))[0]
    for dzA, dzB in COMBOS:
        out_lines = make_variant(src, dzA, dzB)
        out_path = os.path.join(OUT_DIR, f"{base}_dzA{dzA:+.0f}_dzB{dzB:+.0f}.inp")
        # 原样写回，不额外加工换行，避免“空行翻倍”
        with open(out_path, "w", encoding="utf-8", newline='') as fw:
            fw.writelines(out_lines)
        print(f"[OK] 写出: {out_path}  行数: {len(out_lines)}")
