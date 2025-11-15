import requests
import sys

def get_babelnet_glosses(babelnet_id: str, api_key: str):
    """
    Query BabelNet for a synset's glosses given its BabelNet ID and API key.

    :param babelnet_id: e.g. 'bn:00031027n'
    :param api_key: your BabelNet API key as a string
    :return: list of gloss strings
    """
    url = "https://babelnet.io/v7/getSynset"
    params = {
        "id": babelnet_id,
        "key": api_key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    glosses = []
    # Depending on the API version, glosses can be in data['glosses']
    # Each gloss may be an object with fields like 'gloss' and 'language'
    if "glosses" in data:
        for g in data["glosses"]:
            gloss_text = g.get("gloss")
            if gloss_text:
                glosses.append(gloss_text)

    return glosses


if __name__ == "__main__":
    # Simple CLI usage:
    # python get_babelnet_gloss.py bn:00031027n YOUR_API_KEY
    if len(sys.argv) != 3:
        print("Usage: python get_babelnet_gloss.py <BABELNET_ID> <API_KEY>")
        sys.exit(1)

    babelnet_id = sys.argv[1]
    api_key = sys.argv[2]

    try:
        glosses = get_babelnet_glosses(babelnet_id, api_key)
        if not glosses:
            print(f"No glosses found for synset {babelnet_id}.")
        else:
            print(f"Glosses for {babelnet_id}:")
            for i, gloss in enumerate(glosses, start=1):
                print(f"{i}. {gloss}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Error: {e}")
