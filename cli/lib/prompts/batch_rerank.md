Rank these movies by relevance to the search query.

Query: "{query}"

Movies: {doc_list}

Return ONLY the IDs in order of relevance(best match first). Return a valid JSON list, nothing else. For example

[75, 12, 34, 2, 1]

This means the movie with idx of 75 is most relevant

I will be parsing with json.loads, so nothing but the list

Result: