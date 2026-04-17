# Random Indexing - Experiment Results

## Base Configuration
- Dimension: 2000
- Non-zero: 100
- Window: left=2, right=2
- Normalized vectors
- Corpus: Harry Potter 1-7

## How to reproduce

All results in sections 1 to 4 come from `RandomIndexing/experiments.py`, which tokenizes once and sweeps every configuration. From `RandomIndexing/`:

```bash
python experiments.py
```

Question (b) uses `VectorTester.py` explicitly (see its own section). Sections 5 (Word2Vec) and 6 (Multilingual) have their own reproduce blocks.

The 8 test words are: `harry, gryffindor, chair, wand, good, enter, on, school`.

---

## Question (b) - Nearest neighbors with default configuration

Reproduce:
```bash
python random_indexing.py -f data -n -o vectors.txt
python VectorTester.py -f vectors.txt
# at the prompt, type the 8 test words (space-separated):
# > harry gryffindor chair wand good enter on school
# then: exit
```

Default config: `dim=2000, nz=100, window=2+2, cosine, normalized`.

| Word | 4 nearest neighbors (distance) |
|------|---------------------|
| harry | percy (0.020), hagrid (0.022), snape (0.026), neville (0.026) |
| gryffindor | slytherin (0.081), ravenclaw (0.121), school (0.123), house (0.125) |
| chair | seat (0.049), bag (0.101), hand (0.102), cupboard (0.111) |
| wand | head (0.054), hand (0.055), nose (0.068), leg (0.077) |
| good | nice (0.061), quick (0.067), funny (0.069), quiet (0.093) |
| enter | leave (0.092), according (0.115), break (0.117), owing (0.119) |
| on | in (0.039), from (0.041), over (0.047), through (0.055) |
| school | house (0.054), class (0.055), point (0.059), fire (0.067) |

The results match the example from the assignment (`harry` close to percy, hagrid, neville) and make sense:
- `harry`, `gryffindor`: other characters and Hogwarts houses
- `chair`, `wand`: concrete objects / body parts that share verbs like "held"
- `good`: other adjectives (`nice, quick, funny`)
- `enter`: `leave` as antonym, same syntactic slot ("enter/leave the room")
- `on`, `school`: similar grammatical or topical role

Note: distances vary a bit across runs because `random_indexing.py` doesn't fix a seed. The values above come from `experiments.py` which does use seed=42.

---

## Question (c) - Hyperparameter exploration

Sections 1 to 4 cover the four bullets of (c): similarity metric, normalization, dimension vs non-zero proportion, and window size.

---

## 1. Similarity Metrics (normalized vectors)

### Cosine
| Word | Neighbors |
|------|-----------|
| harry | percy (0.020), hagrid (0.022), snape (0.026), neville (0.026) |
| gryffindor | slytherin (0.081), ravenclaw (0.121), school (0.123), house (0.125) |
| chair | seat (0.049), bag (0.101), hand (0.102), cupboard (0.111) |
| wand | head (0.054), hand (0.055), nose (0.068), leg (0.077) |
| good | nice (0.061), quick (0.067), funny (0.069), quiet (0.093) |
| enter | leave (0.092), according (0.115), break (0.117), owing (0.119) |
| on | in (0.039), from (0.041), over (0.047), through (0.055) |
| school | house (0.054), class (0.055), point (0.059), fire (0.067) |

### Euclidean (normalized)
| Word | Neighbors |
|------|-----------|
| harry | percy (0.198), hagrid (0.209), snape (0.226), neville (0.230) |
| gryffindor | slytherin (0.404), ravenclaw (0.492), school (0.496), house (0.500) |
| chair | seat (0.313), bag (0.450), hand (0.452), cupboard (0.471) |
| wand | head (0.329), hand (0.330), nose (0.368), leg (0.394) |
| good | nice (0.351), quick (0.365), funny (0.372), quiet (0.432) |
| enter | leave (0.430), according (0.480), break (0.483), owing (0.488) |
| on | in (0.279), from (0.288), over (0.307), through (0.331) |
| school | house (0.328), class (0.330), point (0.343), fire (0.367) |

### Manhattan (normalized)
| Word | Neighbors |
|------|-----------|
| harry | hagrid (6.552), percy (6.743), snape (7.435), neville (7.640) |
| gryffindor | slytherin (13.138), school (16.587), house (16.606), ravenclaw (16.817) |
| chair | seat (10.644), hand (14.381), bag (14.710), trunk (15.118) |
| wand | hand (10.900), head (11.250), nose (12.485), leg (13.491) |
| good | nice (12.133), quick (12.649), funny (12.754), well (14.304) |
| enter | leave (15.014), according (16.179), fight (16.778), owing (16.785) |
| on | in (9.150), from (9.718), over (10.427), through (11.227) |
| school | house (10.687), class (10.736), point (11.675), fire (12.180) |

### Observation
With normalized vectors, cosine and euclidean give exactly the same neighbor orderings. That's expected: for unit vectors `||x - y||^2 = 2 - 2*cos(x, y)`, so ranking by one is the same as ranking by the other. Manhattan gives very similar but slightly different rankings (for `harry`, hagrid and percy swap; for `good`, `quiet` is replaced by `well`). Distance magnitudes differ a lot across metrics but the actual neighbors are consistent.

### Conclusion
Cosine is the preferred metric. It's scale-invariant, bounded in [0, 2], and standard for distributional semantics. On normalized vectors it ranks the same way as euclidean but is more interpretable. Manhattan swaps a couple of neighbors here and there but doesn't bring any clear benefit.

---

## 2. Normalized vs Non-normalized Vectors

Tested across all three metrics to show where normalization actually matters.

### Cosine, normalized
| Word | Neighbors |
|------|-----------|
| harry | percy (0.020), hagrid (0.022), snape (0.026), neville (0.026) |
| gryffindor | slytherin (0.081), ravenclaw (0.121), school (0.123), house (0.125) |
| good | nice (0.061), quick (0.067), funny (0.069), quiet (0.093) |
| school | house (0.054), class (0.055), point (0.059), fire (0.067) |

### Cosine, non-normalized
| Word | Neighbors |
|------|-----------|
| harry | percy (0.020), hagrid (0.022), snape (0.026), neville (0.026) |
| gryffindor | slytherin (0.081), ravenclaw (0.121), school (0.123), house (0.125) |
| good | nice (0.061), quick (0.067), funny (0.069), quiet (0.093) |
| school | house (0.054), class (0.055), point (0.059), fire (0.067) |

### Euclidean, normalized
| Word | Neighbors |
|------|-----------|
| harry | percy (0.198), hagrid (0.209), snape (0.226), neville (0.230) |
| gryffindor | slytherin (0.404), ravenclaw (0.492), school (0.496), house (0.500) |
| good | nice (0.351), quick (0.365), funny (0.372), quiet (0.432) |
| school | house (0.328), class (0.330), point (0.343), fire (0.367) |

### Euclidean, non-normalized
| Word | Neighbors |
|------|-----------|
| harry | he (51976), it (78611), i (82286), ron (84680) |
| gryffindor | slytherin (1811), house (1888), fire (1888), class (1942) |
| good | long (3501), its (3640), after (3803), black (3809) |
| school | house (1548), castle (1829), place (1840), window (1880) |

### Manhattan, normalized
| Word | Neighbors |
|------|-----------|
| harry | hagrid (6.552), percy (6.743), snape (7.435), neville (7.640) |
| gryffindor | slytherin (13.138), school (16.587), house (16.606), ravenclaw (16.817) |
| good | nice (12.133), quick (12.649), funny (12.754), well (14.304) |
| school | house (10.687), class (10.736), point (11.675), fire (12.180) |

### Manhattan, non-normalized
| Word | Neighbors |
|------|-----------|
| harry | he (1615936), it (2060714), ron (2174160), hermione (2284596) |
| gryffindor | slytherin (58142), house (62008), fire (63536), class (63602) |
| good | neville (112498), long (112858), its (118598), sirius (122516) |
| school | house (49488), castle (56806), place (60398), window (61740) |

### Observation
- Cosine: normalized and non-normalized give strictly identical results. The formula `1 - x.y / (||x||*||y||)` already divides by the norms, so an extra normalization step does nothing.
- Euclidean and Manhattan: normalization matters a lot. Without it, the nearest neighbor of `harry` is `he`, `it`, `i`, `ron`. These are just the most frequent tokens, which happen to have huge vectors (big sums of random vectors from many contexts). Big magnitude + big magnitude = small euclidean distance regardless of direction. Same story for `good` where we get `long, its, after`.
- Once normalized, euclidean and manhattan give roughly the same neighbors as cosine. That's consistent with section 1: on unit vectors the three metrics rank things similarly.

### Conclusion
Normalization is harmless for cosine and essential for euclidean/manhattan. If the metric depends on magnitude, frequent words dominate and the neighbor ranking collapses to "other frequent words". With unit vectors the three metrics become almost equivalent.

---

## 3. Dimensionality and Non-zero Proportion (cosine, normalized)

### Dimension 10

| Config | harry | gryffindor | good | school |
|--------|-------|------------|------|--------|
| nz=1 (5%) | percy, dumbledore, hagrid, snape | weak, rosebushes, like, dim-witted | sniggering, bubotubers, ravenclaw, dreadful | concealing, for, lurch, beside |
| nz=1 (10%) | percy, dumbledore, hagrid, snape | weak, rosebushes, like, dim-witted | sniggering, bubotubers, ravenclaw, dreadful | concealing, for, lurch, beside |
| nz=2 (20%) | percy, wood, neville, griphook | map, slytherin, weather, spent | macnair, no, oooh, it | burrow, class, country, news |
| nz=5 (50%) | jinxed, hagrid, dobby, snape | modify, park, presence, stinksap | disgusting, funny, shes, there | burrow, sky, ceiling, house |

### Dimension 50

| Config | harry | gryffindor | good | school |
|--------|-------|------------|------|--------|
| nz=2 (5%) | percy, hagrid, neville, fudge | slytherin, ravenclaw, party, house | quick, funny, nice, prefect | house, all, class, point |
| nz=5 (10%) | hagrid, percy, hermione, snape | slytherin, dormitory, most, house | funny, quick, nice, lovely | house, students, class, point |
| nz=10 (20%) | percy, neville, hagrid, dumbledore | slytherin, ravenclaw, fire, one | nice, funny, quick, quiet | house, others, cottage, window |
| nz=25 (50%) | percy, hagrid, fudge, dumbledore | slytherin, ravenclaw, house, malfoys | nice, very, its, shes | house, point, castle, others |

### Dimension 100

| Config | harry | gryffindor | good | school |
|--------|-------|------------|------|--------|
| nz=5 (5%) | percy, hagrid, neville, snape | slytherin, sea, most, school | nice, quick, lovely, funny | students, house, class, water |
| nz=10 (10%) | percy, neville, hagrid, snape | slytherin, ravenclaw, house, four | funny, nice, quick, lovely | house, class, book, fire |
| nz=20 (20%) | percy, hagrid, snape, dumbledore | slytherin, ravenclaw, school, first | funny, nice, quick, theres | class, house, burrow, castle |
| nz=50 (50%) | percy, hagrid, snape, neville | slytherin, ravenclaw, house, school | nice, quick, funny, voice | class, house, castle, fire |

### Dimension 1000

| Config | harry | gryffindor | good | school |
|--------|-------|------------|------|--------|
| nz=50 (5%) | percy, hagrid, snape, neville | slytherin, ravenclaw, school, house | nice, funny, quick, quiet | house, class, point, fire |
| nz=100 (10%) | percy, hagrid, snape, neville | slytherin, ravenclaw, school, house | quick, funny, nice, quiet | house, class, point, burrow |
| nz=200 (20%) | percy, hagrid, snape, neville | slytherin, school, ravenclaw, house | nice, quick, funny, quiet | class, house, point, burrow |
| nz=500 (50%) | percy, hagrid, neville, snape | slytherin, ravenclaw, house, school | nice, quick, funny, quiet | house, class, point, others |

### Dimension 2000, non-zero 100 (5%), baseline
See section 1.

### Observation
- Dimension 10 is broken regardless of the non-zero proportion. Neighbors of `gryffindor` are unrelated words like `weak`, `rosebushes`, `modify`, `park`. The space is too small for 25k words, random vectors aren't quasi-orthogonal anymore and collisions dominate. With nz=1 (both 5% and 10% round to 1) the two runs are identical.
- Dimension 50 already gives clear improvement. `gryffindor` reliably finds `slytherin`, `harry` finds `percy, hagrid`, but the tail is still a bit noisy.
- Dimension 100 is stable. Most neighbors are semantically meaningful and match the baseline. Raising nz above 10% gives diminishing returns.
- Dimension 1000 is essentially the same as the baseline (2000). Neighbors are stable across all non-zero proportions, which confirms convergence.
- The non-zero proportion matters much less than the dimension. Once dim is 100 or more, going from 5% to 50% doesn't change the quality much. 5% is already enough.

### Conclusion
Dimensionality is the main knob. dim=10 is unusable, dim=50 is OK, dim=100 is a good trade-off, and above dim=1000 you're just burning memory. Non-zero proportion has limited impact: 5-10% is a safe default. The assignment default (2000, 5%) is oversized but fine.

---

## 4. Window Size (cosine, normalized, dim=2000)

### Window 0 (symmetric)
All words return the same arbitrary neighbors with distance 1.0 (`288`, `hailstorm`, `287`, `half-polished`, ...). With no context on either side, every context vector stays at zero and can't be compared, so the first words in `id2word` are returned by tie-breaking.

### Window 3 (symmetric)
| Word | Neighbors |
|------|-----------|
| harry | hagrid (0.018), dumbledore (0.018), snape (0.019), neville (0.019) |
| gryffindor | slytherin (0.061), house (0.085), ravenclaw (0.086), school (0.092) |
| good | nice (0.049), quick (0.056), funny (0.058), mad (0.061) |
| school | house (0.050), point (0.057), class (0.057), last (0.063) |

### Window 10 (symmetric)
| Word | Neighbors |
|------|-----------|
| harry | snape (0.008), moody (0.010), malfoy (0.010), dumbledore (0.010) |
| gryffindor | slytherin (0.017), by (0.022), house (0.022), two (0.024) |
| good | here (0.014), nice (0.015), dumbledore (0.015), funny (0.015) |
| school | house (0.016), first (0.019), last (0.019), hogwarts (0.020) |

### Asymmetric windows (more right, less left)

| Config | harry | gryffindor | good |
|--------|-------|------------|------|
| L=0, R=3 | weasley, snape, dumbledore, neville | slytherin, ravenclaw, hufflepuff, house | here, right, sorry, no |
| L=0, R=10 | snape, still, head, again | slytherin, house, morning, now | right, here, no, tonight |

### Asymmetric windows (more left, less right)

| Config | harry | gryffindor | good |
|--------|-------|------------|------|
| L=3, R=0 | professor, hagrid, mr., snape | slytherin, school, death, quidditch | nice, brilliant, quick, funny |
| L=10, R=0 | professor, mr., hagrid, moody | slytherin, next, death, first | nice, funny, big, bad |

### Observation
- Window 0: no context, all context vectors stay at zero, results are meaningless.
- Window 3: tight syntactic neighbors. `good` finds `nice, quick, funny, mad` (same adjective category), `harry` finds the main-character cluster.
- Window 10: more topical neighbors. `harry` finds `moody, malfoy` (characters involved in the same scenes), `school` finds `hogwarts`. But noise grows: `gryffindor` picks up `by, two` and `good` picks up `here`.
- Left-only (L > 0, R = 0) captures words that typically come *before* the focus word. For `harry` this gives titles like `professor, mr., hagrid, moody`, because many sentences follow the "Title Name" pattern or "Name said Mr.". For `good` you still get adjectives, because adjectives tend to follow the same articles.
- Right-only (L = 0, R > 0) captures words that come *after* the focus word. For `harry` this gives co-occurring names (`weasley, snape, dumbledore`), matching "Harry and X" or "Harry looked at X".
- Asymmetric windows expose position effects that symmetric windows average out.

### Conclusion
- Window 0 is useless.
- Small symmetric windows (2 or 3) capture tight syntactic / paradigmatic similarity (same POS, near-synonyms).
- Large windows (10) capture broader topical similarity but dilute the signal with noise.
- Asymmetric windows are useful when position matters (title + name patterns).
- Best default: symmetric window of 2 or 3.

---

## General Conclusion

- Metric: cosine works best, is robust with or without normalization, and doesn't depend on vector magnitude.
- Normalization: required for euclidean/manhattan, irrelevant for cosine.
- Dimension: bigger helps up to around 100-1000, then gains flatten out. 10 is too small.
- Non-zero proportion: low impact. 5-10% is a safe default.
- Window size: 0 is useless, 2-3 gives syntactic/paradigmatic neighbors, 10 gives topical neighbors but noisier, asymmetric windows capture directional patterns (titles before, names after).

---

## 5. Word2Vec Results (cosine, dim=50, 5 epochs, 10 negative samples)

Reproduce (around 1h15 of training on the 7 books):
```bash
cd ../word2vec
python w2v.py -f ../RandomIndexing/data -o vectors.txt
python ../RandomIndexing/VectorTester.py -f vectors.txt
```

### Skip-gram with negative sampling, modified unigram distribution

| Word | Neighbors |
|------|-----------|
| harry | ron (0.092), griphook (0.136), neville (0.140), hagrid (0.148) |
| gryffindor | slytherin (0.111), ravenclaw (0.131), hufflepuff (0.207), keeper (0.211) |
| chair | seat (0.097), desk (0.170), lamp (0.182), cabin (0.187) |
| wand | arm (0.200), hand (0.207), firebolt (0.212), leg (0.218) |
| good | bad (0.168), nice (0.207), interesting (0.216), fine (0.220) |
| enter | discuss (0.191), win (0.191), follow (0.213), occur (0.222) |
| on | upon (0.223), against (0.272), onto (0.278), flat (0.283) |
| school | hogwarts (0.135), journey (0.189), post (0.208), age (0.216) |

### Comparison: Random Indexing vs Word2Vec

| Word | Random Indexing (default, dim=2000) | Word2Vec (dim=50) |
|------|-------------------------------------|-------------------|
| harry | percy, hagrid, snape, neville | ron, griphook, neville, hagrid |
| gryffindor | slytherin, ravenclaw, school, house | slytherin, ravenclaw, hufflepuff, keeper |
| chair | seat, bag, hand, cupboard | seat, desk, lamp, cabin |
| wand | head, hand, nose, leg | arm, hand, firebolt, leg |
| good | nice, quick, funny, quiet | bad, nice, interesting, fine |
| enter | leave, according, break, owing | discuss, win, follow, occur |
| on | in, from, over, through | upon, against, onto, flat |
| school | house, class, point, fire | hogwarts, journey, post, age |

### Observation
- Word2Vec distances are generally larger (0.09 to 0.28) than Random Indexing (0.02 to 0.13), because word2vec uses only 50 dimensions vs RI's 2000. Fewer dimensions leaves less room to spread the vectors apart.
- Word2Vec catches semantic relations that RI misses:
  - `harry` closest to `ron` (best friend, not just another named character)
  - `gryffindor` completing the 4 Hogwarts houses (`slytherin, ravenclaw, hufflepuff`)
  - `wand` close to `firebolt` (a magical object, not a body part)
  - `good` close to `bad` (antonym, same contexts)
  - `school` close to `hogwarts` (instance relation)
- Some word2vec neighbors are more surprising and closer to co-occurrence: `wand` to `arm, hand` comes from the "raised his wand/arm" pattern, which is the same kind of signal RI picks up.
- For very common prepositions like `on`, both models struggle. RI finds syntactic twins (`in, from`), word2vec finds near-synonyms (`upon, onto`).

### Conclusion
Word2Vec gives qualitatively better vectors. It's a predictive model that actively learns which dimensions matter, whereas RI just accumulates co-occurrences. Negative sampling also pushes apart words that shouldn't co-occur, which sharpens the separation. With only 50 dimensions (40x smaller than RI) word2vec still captures richer relations (antonyms, instances, siblings).

---

## 6. Multilingual Word Embeddings Results

Reproduce:
```bash
cd ../multilingual
# 4 variants: Baseline, TF-IDF, Mean-Centered, Mean-Centered + TF-IDF
python run_experiments.py
# Failure analysis: 3 EN->ES + 3 ES->EN failures on the best method
python failure_analysis.py
```

### Configuration
- Dataset: 500 aligned English-Spanish sentences (literature)
- Embeddings: pre-trained FastText (mini.en.vec, mini.es.vec)
- Tokenization: `re.findall(r'\w+', sentence.lower())` (Unicode-aware, handles ñ, á, é)
- TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer` (smoothed IDF)
- Metric: cosine distance

### Accuracy Results

| Method | EN->ES Top-1 | EN->ES Top-3 | ES->EN Top-1 | ES->EN Top-3 |
|--------|-------------|-------------|-------------|-------------|
| Baseline (Simple Mean) | 25.00% | 41.60% | 20.60% | 31.20% |
| TF-IDF Weighted | 35.80% | 50.60% | 31.20% | 45.20% |
| Mean-Centered (Simple) | 72.20% | 85.60% | 75.20% | 87.00% |
| Mean-Centered + TF-IDF | 80.80% | 88.80% | 83.00% | 90.00% |

### Failed Sentences (Mean-Centered + TF-IDF, top-3)

EN->ES failures (56 total):
- [idx 25] *"But as he required the promise, I could not do less than give it; at least I thought so at the time."* Correct translation ranks #3. Top-1 is a semantically adjacent reflection on doing the right amount.
- [idx 31] *"Why, to be sure," said her husband, very gravely, "that would make great difference."* Correct translation ranks #44. Dialogue line with vague referential expressions and few content-word anchors.
- [idx 37] *"I would not wish to do any thing mean," he replied.* Correct translation ranks #3. Short generic dialogue with TF-IDF dominated by one or two words.

ES->EN failures (50 total):
- [idx 3] *"Murió el anciano caballero, se leyó su testamento..."* Correct translation ranks #6. Top-1 is a paragraph-adjacent sentence about the same "solemn promise" scene.
- [idx 10] *"Tenía inteligencia y buen juicio, pero era vehemente en todo..."* Correct translation ranks #3. Thematic words like "moderation" and "eagerness" appear in several sentences.
- [idx 25] *"Pero como él quiso que se lo prometiera..."* Same sentence as EN->ES failure #1, mirrored.

### Observations

Baseline (25% / 20.6%): poor. Simple averaging gives equal weight to function words like "the", "de", "la", which dominate the centroid and drown out the semantic signal.

TF-IDF (35.8% / 31.2%): +10% over baseline. Down-weighting frequent function words and up-weighting rare content words gives a more discriminative centroid.

Mean-Centered Simple (72.2% / 75.2%): +47% over baseline. Each language has a language-specific bias: a shift vector representing the average "English-ness" or "Spanish-ness" in the aligned space. Sentence centroids inherit that shift. Subtracting the per-language mean removes the constant offset and exposes the residual semantic content, which is directly comparable across languages.

Mean-Centered + TF-IDF (80.8% / 83.0%): best, +55% over baseline. TF-IDF amplifies the semantic signal, mean-centering removes the language bias. The two are complementary.

### Why Mean-Centering works so well
Cross-lingual FastText vectors are aligned via a shared subspace, but each language keeps a residual bias (language identity, typical sentence structures, function-word stats). Averaged over 500 sentences this bias is basically a constant vector. When comparing EN->ES distances, that constant dominates the cosine geometry. Mean-centering removes it and leaves only the semantic residuals, which is where translation equivalence actually lives. It also partially fixes the hubness problem (in high-dim spaces some vectors become nearest neighbors of many others).

### Question: Which mean to subtract for Mean-Centered TF-IDF?
The TF-IDF weighted mean, not the original mean. The TF-IDF vectors don't live in the same representation as the simple averages, so the correct centering is the one computed from the same representation. Subtracting the original mean would leave a TF-IDF-specific bias uncorrected.

### Common causes of translation failures
1. Short, abstract dialogue with few content words: not enough signal for the centroid.
2. Paragraph-level topical clustering: several sentences discuss the same scene, so their centroids cluster.
3. Idiomatic or non-literal translations: the translator reformulated instead of matching word-for-word.
4. Function-word dominance that survives TF-IDF when the sentence has very few unique words.

### Conclusion
Mean-centering is the single most impactful transformation for cross-lingual retrieval (+47% top-1). TF-IDF alone helps but isn't enough. Combined, they give 80.8% EN->ES and 83.0% ES->EN top-1. The two techniques fix different problems and stack cleanly.
