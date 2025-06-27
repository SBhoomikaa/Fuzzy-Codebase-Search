from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import yaml
import os
import ast

app = Flask(__name__)
CORS(app)

def levenshtein(a, b):
    d = [[0 for _ in range(len(b)+1)] for _ in range(len(a)+1)]
    for i in range(0, len(a)+1):
        for j in range(0, len(b)+1):
            if i == 0 or j == 0:
                d[i][j] = max(i, j)
            else:    
                d[i][j] = 0
    cost = 1
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                cost = 0
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
            cost = 1
    return d[len(a)][len(b)]

def normalized_levenshtein(a, b):
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0  
    return 1 - (dist / max_len)

def parse_abbreviations(filename):
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        return {}

    final_abbr = {}
    degree_priority = {
        'recommended': 3,
        'context sensitive': 2,
        'not recommended': 1
    }

    for entry in data:
        word = entry.get('word', None)
        abbrs_list = entry.get('abbrs', [])

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

            if abbr not in final_abbr or priority > final_abbr[abbr][1]:
                final_abbr[abbr] = (word, priority)

    return {abbr: word for abbr, (word, _) in final_abbr.items()}

def expand_abbreviation(token, abbr_dict):
    if token in abbr_dict:
        return [abbr_dict[token], 'chosen']
    else:
        return [token, 'original']

def preprocess(a):
    tokens = re.split(r'[-_|\d+\s]', a)
    result = []
    for token in tokens:
        if not token:
            continue
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', token)
        for part in parts:
            part_lower = part.lower()
            if len(part_lower) > 4:  
                split_positions = []
                common_splits = ['get', 'set', 'make', 'show', 'high', 'low', 'min', 'max', 'pass', 'stop', 'start', 'end', 'first', 'last']
                
                for split_word in common_splits:
                    if part_lower.startswith(split_word) and len(part_lower) > len(split_word):
                        split_positions.append(len(split_word))
                    elif part_lower.endswith(split_word) and len(part_lower) > len(split_word):
                        split_positions.append(len(part_lower) - len(split_word))
                
                if split_positions:
                    split_pos = split_positions[0]  
                    result.extend([part_lower[:split_pos], part_lower[split_pos:]])
                else:
                    result.append(part_lower)
            else:
                result.append(part_lower)
    return result

def needleman_wunsch(conf_array, len_a, len_b, gap_penalty=-0.2):
    sim_matrix = []
    idx = 0
    for i in range(len_a):
        row = []
        for j in range(len_b):
            row.append(conf_array[idx])
            idx += 1
        sim_matrix.append(row)
    
    dp = [[0]*(len_b+1) for _ in range(len_a + 1)]

    for i in range(1, len_a + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, len_b + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            match = dp[i - 1][j - 1] + sim_matrix[i - 1][j - 1]
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
    
    return dp[len_a][len_b]

def confidence_score(a, b, abbr_dict):
    total_comparisons = len(a) * len(b)
    confidence_array = [0] * total_comparisons
    
    u = 0
    
    for token_a in a:
        x = expand_abbreviation(token_a, abbr_dict)
        for token_b in b:
            y = expand_abbreviation(token_b, abbr_dict)
            
            if x[1] == "original" and y[1] == "original":
                confidence_array[u] = normalized_levenshtein(token_a, token_b)
            elif x[1] == "original":
                confidence_array[u] = max(
                    normalized_levenshtein(token_a, token_b),
                    normalized_levenshtein(token_a, y[0])
                )
            elif y[1] == "original":
                confidence_array[u] = max(
                    normalized_levenshtein(token_a, token_b),
                    normalized_levenshtein(x[0], token_b)
                )
            else:
                confidence_array[u] = max(
                    normalized_levenshtein(token_a, token_b),
                    normalized_levenshtein(token_a, y[0]),
                    normalized_levenshtein(x[0], token_b),
                    normalized_levenshtein(x[0], y[0])
                )
            u += 1
    
    alignment_score = needleman_wunsch(confidence_array, len(a), len(b))
    
    gap_penalty = -0.2
    max_length = max(len(a), len(b))
    min_length = min(len(a), len(b))
    
    max_possible_score = min_length * 1.0
    min_possible_score = max_length * gap_penalty
    
    if max_possible_score == min_possible_score:
        normalized_score = 1.0
    else:
        normalized_score = (alignment_score - min_possible_score) / (max_possible_score - min_possible_score)
    
    return normalized_score

def extract_identifiers_from_codebase(codebase_path):
    identifiers = []
    # Use a set to track unique identifiers and avoid duplicates
    seen_identifiers = set()
    file_count = 0  
    
    for root, dirs, files in os.walk(codebase_path):
        for file in files:
            if file.endswith(".py"):
                if file_count >= 20000:
                    return identifiers
                try:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, codebase_path)
                    folder_name = os.path.dirname(relative_path) if os.path.dirname(relative_path) else "."
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        node = ast.parse(f.read())
                        for n in ast.walk(node):
                            # Only extract function and method definitions
                            if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
                                # Skip magic methods (dunder methods)
                                if n.name.startswith("__") and n.name.endswith("__"):
                                    continue
                                    
                                # Create unique key for deduplication
                                unique_key = (n.name, relative_path)
                                if unique_key not in seen_identifiers:
                                    seen_identifiers.add(unique_key)
                                    identifiers.append({
                                        'name': n.name,
                                        'file': file,
                                        'folder': folder_name,
                                        'full_path': relative_path
                                    })
                    file_count += 1
                except Exception as e:
                    print(f"Skipping {file}: {e}")          
    return identifiers

# Load abbreviations dictionary once at startup
abbr_dict = parse_abbreviations("yml.txt")

@app.route('/')
def index():
    with open('front.html', 'r',encoding='utf-8') as f:
        return f.read()

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        user_input = data.get('query', '')
        codebase_path = data.get('codebase_path', 'codebase')
        
        if not user_input:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not os.path.exists(codebase_path):
            return jsonify({'error': f'Codebase path "{codebase_path}" does not exist'}), 400
        
        user_tokens = preprocess(user_input)
        all_identifiers = extract_identifiers_from_codebase(codebase_path)
        
        results = []
        # Use a set to track unique results and avoid duplicates in final output
        seen_results = set()
        
        for identifier_info in all_identifiers:
            ident_tokens = preprocess(identifier_info['name'])
            score = confidence_score(user_tokens, ident_tokens, abbr_dict)
            
            # Create unique key for final result deduplication
            result_key = (identifier_info['name'], identifier_info['full_path'])
            
            if result_key not in seen_results:
                seen_results.add(result_key)
                results.append({
                    'name': identifier_info['name'],
                    'file': identifier_info['file'],
                    'folder': identifier_info['folder'],
                    'full_path': identifier_info['full_path'],
                    'score': round(score, 4)
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'query': user_input,
            'results': results[:10],
            'total_identifiers': len(all_identifiers)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
