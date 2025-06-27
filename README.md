# Fuzzy-Codebase-Search
This project aims to make navigating large codebases more intuitive and developer-friendly. Traditional code search tools rely on exact string matching, which can fail if you don’t know the precise name of a function or variable. Our system uses approximate string matching and token-level alignment to retrieve the most relevant results, even if you only roughly remember what a function is called.

Key features:  
-Accepts partial or approximate queries  
-Returns the top 10 best-matching identifiers  
-Supports abbreviation expansion (e.g., usr → user)  
-Ranks results based on a customized alignment of semantic tokens  
-Provides file and folder location for each match  
-Easy-to-use web interface built with Flask  

Algorithms used: Levenshtein Distance & Needleman Wunsch Algorithm
