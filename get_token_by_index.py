# ===== Constants you will modify before each run =====
TOKENS = "高盛( Gold man Sachs) 、 摩根 大 通( JP Morgan Chase) 和 摩根士丹 利( Morgan Stanley ) 已经 在 6月份 偿还"
INDEX = 19  # 0-based index
# =====================================================

def get_token(tokens: str, index: int) -> str:
    parts = tokens.split()
    if index < 0 or index >= len(parts):
        raise IndexError(f"Index {index} is out of range for {len(parts)} tokens.")
    return parts[index]

if __name__ == "__main__":
    try:
        result = get_token(TOKENS, INDEX)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
