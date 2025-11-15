# ===== Constants you will modify before each run =====
TOKENS = "apple banana cherry date"
TARGET_TOKEN = "cherry"
# =====================================================

def get_token_index(tokens: str, target: str) -> int:
    parts = tokens.split()
    try:
        return parts.index(target)
    except ValueError:
        raise ValueError(f"Token '{target}' not found in the input tokens.")

if __name__ == "__main__":
    try:
        index = get_token_index(TOKENS, TARGET_TOKEN)
        print(index)
    except Exception as e:
        print(f"Error: {e}")
