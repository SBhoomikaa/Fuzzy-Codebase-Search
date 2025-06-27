import yaml

def parse_abbreviations(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    final_abbr = {}
    degree_priority = {
        'recommended': 3,
        'context sensitive': 2,
        'not recommended': 1
    }

    for entry in data:
        word = entry.get('word', None)
        abbrs_list = entry.get('abbrs', [])

        # Make sure abbrs_list is a list
        if isinstance(abbrs_list, dict):
            abbrs_list = [abbrs_list]
        elif not isinstance(abbrs_list, list):
            continue

        for abbr_entry in abbrs_list:
            abbr = abbr_entry.get('abbr')
            degree = abbr_entry.get('degree', '').lower()

            if abbr is None or word is None:
                continue

            priority = degree_priority.get(degree, 0)

            # Keep only highest priority abbreviation mapping
            if abbr not in final_abbr or priority > final_abbr[abbr][1]:
                final_abbr[abbr] = (word, priority)

    # Return simple dict with abbreviation -> word
    return {abbr: word for abbr, (word, _) in final_abbr.items()}


def expand_abbreviation(token, abbr_dict):
    if token in abbr_dict:
        return [[abbr_dict[token], 'chosen']]
    else:
        return [[token, 'original']]


if __name__ == "__main__":
    print("Loaded abbreviations:", abbr_dict)
    token = input("Enter abbreviation (or 'exit' to quit): ").strip()
    expansions = expand_abbreviation(token, abbr_dict)
    print("Expansion(s):", expansions)
