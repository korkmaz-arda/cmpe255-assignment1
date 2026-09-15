"""Project knowledge base (F02): personas, hand-written Q&A entries, splits, presets.

Written for this reimplementation. Each entry has several paraphrased user
queries, one assistant answer, a topic, and the personas for which it is in scope.

Entries are split *before* any paraphrase/persona expansion into train /
validation / final-test entries (seeded, stratified by topic). Separately, one
paraphrase of every train entry is withheld as a rephrasing diagnostic: new
wording, but the answer itself is seen during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.records import Dialogue, norm_key

PERSONAS: dict[str, str] = {
    "assistant": "You are NanoLlama, a small helpful assistant.",
    "brilliant": "You are NanoLlama, a brilliant and friendly assistant who explains things clearly.",
    "ml_research": "You are NanoLlama, a machine learning research assistant.",
    "python": "You are NanoLlama, a Python engineer who writes short, correct code.",
    "storyteller": "You are NanoLlama, a gentle storyteller who writes simple stories for children.",
    "math_tutor": "You are NanoLlama, a patient math tutor who shows each step.",
}
DEFAULT_PERSONA = "assistant"

PERSONA_LABELS = {
    "assistant": "Assistant",
    "brilliant": "Brilliant assistant",
    "ml_research": "ML research assistant",
    "python": "Python engineer",
    "storyteller": "Storyteller",
    "math_tutor": "Math tutor",
}


@dataclass(frozen=True)
class Entry:
    key: str
    topic: str
    queries: tuple[str, ...]
    answer: str
    personas: tuple[str, ...]


GENERAL = ("assistant", "brilliant")
ML = ("assistant", "brilliant", "ml_research")
PY = ("assistant", "brilliant", "python")
MATH = ("assistant", "brilliant", "math_tutor")
STORY = ("assistant", "brilliant", "storyteller")


def _e(key, topic, queries, answer, personas):
    return Entry(key, topic, tuple(queries), answer.strip(), tuple(personas))


ENTRIES: list[Entry] = [
    # ---------------------------------------------------------------- identity
    _e("identity_who", "identity", [
        "Who are you?", "What is your name?", "Tell me about yourself.", "What are you?",
    ], "I am NanoLlama, a very small language model. I was trained from scratch on this machine, "
       "and I read and write one character at a time. I can make mistakes, so please check "
       "anything important.", GENERAL + ("ml_research", "python", "math_tutor", "storyteller")),
    _e("identity_rope", "identity", [
        "What is RoPE?", "How do rotary position embeddings work?", "Explain rotary embeddings.",
        "How do you know the order of the characters?",
    ], "RoPE means rotary position embeddings. Each query and key vector is split into pairs of "
       "numbers, and each pair is rotated by an angle that grows with the position. When a query "
       "and a key are compared, only the difference in their positions changes the result, so "
       "attention can use relative distance.", ML),
    _e("identity_swiglu", "identity", [
        "What is SwiGLU?", "How does your feed-forward layer work?", "Explain the SwiGLU block.",
        "What is a gated feed-forward network?",
    ], "SwiGLU is the feed-forward part of each block. The input goes through two projections. One "
       "passes through the SiLU activation and acts as a gate, and the two results are multiplied "
       "together. A third projection maps the result back to the model width.", ML),
    _e("identity_rmsnorm", "identity", [
        "What is RMSNorm?", "How does RMS normalization work?", "Explain root mean square norm.",
        "Why do you use RMSNorm?",
    ], "RMSNorm divides a vector by the square root of the mean of its squared values, then "
       "multiplies by a learned gain. Unlike LayerNorm it does not subtract the mean. It keeps "
       "the scale of activations stable and is cheap to compute.", ML),
    _e("identity_kvcache", "identity", [
        "What is a KV cache?", "How does key value caching work?", "Why is generation faster with a cache?",
        "Explain the KV cache.",
    ], "A KV cache stores the keys and values of tokens that were already processed. When I generate "
       "the next character, I only compute the new token and let it attend to the stored keys and "
       "values, instead of running the whole sequence again.", ML),
    _e("identity_training", "identity", [
        "How were you trained?", "What data were you trained on?", "How did you learn to talk?",
        "What is your training data?",
    ], "I was trained from random weights to predict the next character. My data is a mix of "
       "synthetic children's stories, synthetic small-talk conversations and a small hand-written "
       "knowledge base. The loss was only computed on the assistant replies.", GENERAL + ("ml_research",)),
    # ----------------------------------------------------------- ml foundations
    _e("ml_supervised", "ml_foundations", [
        "What is supervised learning?", "Explain supervised learning.", "What does supervised learning mean?",
        "Give me a simple definition of supervised learning.",
    ], "Supervised learning trains a model on examples that come with the correct answer, called "
       "labels. The model learns a mapping from inputs to labels, for example from an email to "
       "spam or not spam.", ML),
    _e("ml_unsupervised", "ml_foundations", [
        "What is unsupervised learning?", "Explain unsupervised learning.", "What does unsupervised learning mean?",
        "How is unsupervised learning different?",
    ], "Unsupervised learning finds structure in data that has no labels. Common examples are "
       "clustering, which groups similar points, and dimensionality reduction, which finds a "
       "smaller set of directions that keeps most of the information.", ML),
    _e("ml_backprop", "ml_foundations", [
        "What is backpropagation?", "How does backprop work?", "Explain backpropagation.",
        "How are gradients computed in a neural network?",
    ], "Backpropagation computes the gradient of the loss with respect to every weight. It applies "
       "the chain rule from the output layer back to the input layer, reusing intermediate results "
       "so the whole gradient costs about as much as one extra forward pass.", ML),
    _e("ml_gradient_descent", "ml_foundations", [
        "What is gradient descent?", "Explain gradient descent.", "How does gradient descent work?",
        "What does the learning rate do in gradient descent?",
    ], "Gradient descent lowers a loss step by step. At each step it computes the gradient and moves "
       "the weights a small amount in the opposite direction. The learning rate sets the step size: "
       "too large can diverge, too small is slow.", ML),
    _e("ml_overfitting", "ml_foundations", [
        "What is overfitting?", "Explain overfitting.", "What does it mean when a model overfits?",
        "How can I tell if my model is overfitting?",
    ], "Overfitting happens when a model learns the training data too closely, including its noise, "
       "and does worse on new data. A common sign is training loss that keeps falling while "
       "validation loss starts to rise.", ML),
    _e("ml_underfitting", "ml_foundations", [
        "What is underfitting?", "Explain underfitting.", "What does it mean when a model underfits?",
        "Why would a model underfit?",
    ], "Underfitting happens when a model is too simple or trained too little to capture the pattern "
       "in the data. Both training and validation error stay high. A bigger model, better features "
       "or longer training can help.", ML),
    # ------------------------------------------------------------------ metrics
    _e("metric_precision_recall", "metrics", [
        "What are precision and recall?", "Explain precision and recall.", "What is the difference between precision and recall?",
        "Define precision and recall.",
    ], "Precision is the share of predicted positives that are really positive: TP / (TP + FP). "
       "Recall is the share of real positives that the model found: TP / (TP + FN).", ML),
    _e("metric_f1", "metrics", [
        "What is the F1 score?", "Explain F1 score.", "How is F1 calculated?", "What does F1 measure?",
    ], "The F1 score is the harmonic mean of precision and recall: 2 * P * R / (P + R). It is high "
       "only when both precision and recall are high.", ML),
    _e("metric_confusion", "metrics", [
        "What is a confusion matrix?", "Explain a confusion matrix.", "How do I read a confusion matrix?",
        "What does a confusion matrix show?",
    ], "A confusion matrix counts predictions against true labels. For two classes it has four cells: "
       "true positives, false positives, false negatives and true negatives. Most classification "
       "metrics are computed from these counts.", ML),
    _e("metric_cv", "metrics", [
        "What is cross-validation?", "Explain k-fold cross-validation.", "How does cross validation work?",
        "Why use cross-validation?",
    ], "In k-fold cross-validation the data is split into k parts. The model is trained k times, each "
       "time holding out a different part for evaluation, and the scores are averaged. It gives a "
       "more stable estimate than a single split.", ML),
    _e("metric_perplexity", "metrics", [
        "What is perplexity?", "Explain perplexity for language models.", "How is perplexity computed?",
        "What does a lower perplexity mean?",
    ], "Perplexity is the exponential of the average cross-entropy loss. It can be read as how many "
       "choices the model is unsure between for each next token. Lower is better, and 1 would mean "
       "perfect certainty.", ML),
    _e("metric_accuracy", "metrics", [
        "What is accuracy?", "When is accuracy misleading?", "Explain classification accuracy.",
        "Why is accuracy not enough for imbalanced data?",
    ], "Accuracy is the share of predictions that are correct. It can be misleading when classes are "
       "imbalanced: if 99 out of 100 emails are not spam, always saying not spam is 99 percent "
       "accurate but useless.", ML),
    # --------------------------------------------------------------- algorithms
    _e("algo_naive_bayes", "algorithms", [
        "What is Naive Bayes?", "Explain the Naive Bayes classifier.", "How does Naive Bayes work?",
        "Why is Naive Bayes called naive?",
    ], "Naive Bayes is a classifier based on Bayes' rule. It is called naive because it assumes the "
       "features are independent given the class. It is fast and often works well for text.", ML),
    _e("algo_random_forest", "algorithms", [
        "What is a random forest?", "Explain random forests.", "How does a random forest work?",
        "Why does a random forest use many trees?",
    ], "A random forest trains many decision trees, each on a bootstrap sample of the data and with a "
       "random subset of features at each split. It averages their votes, which reduces the "
       "variance of a single tree.", ML),
    _e("algo_kmeans", "algorithms", [
        "What is k-means?", "Explain k-means clustering.", "How does k-means work?",
        "What are the steps of k-means?",
    ], "K-means groups points into k clusters. It starts with k centers, assigns every point to its "
       "nearest center, moves each center to the mean of its points, and repeats until the "
       "assignments stop changing.", ML),
    _e("algo_pca", "algorithms", [
        "What is PCA?", "Explain principal component analysis.", "How does PCA work?",
        "What does PCA do to data?",
    ], "PCA finds the directions along which the data varies the most. It projects the data onto the "
       "first few of these principal components, which reduces the number of dimensions while "
       "keeping as much variance as possible.", ML),
    _e("algo_transformer", "algorithms", [
        "What is a transformer?", "Explain the transformer architecture.", "How does a transformer model work?",
        "What are the parts of a transformer block?",
    ], "A transformer is a neural network built from repeated blocks. Each block has a self-attention "
       "layer, which mixes information between positions, and a feed-forward layer, which "
       "transforms each position. Residual connections and normalization keep training stable.", ML),
    _e("algo_self_attention", "algorithms", [
        "What is self-attention?", "Explain self-attention.", "How does attention work?",
        "What are queries, keys and values?",
    ], "Self-attention lets each token look at other tokens. Each token makes a query, a key and a "
       "value. The query is compared with every key, the scores go through softmax, and the result "
       "is a weighted average of the values.", ML),
    # ------------------------------------------------------------------- python
    _e("py_reverse", "python", [
        "How do I reverse a string in Python?", "Write Python code to reverse a string.",
        "Reverse a string in Python.", "Show me how to reverse text in Python.",
    ], "You can use slicing with a step of -1:\n\ndef reverse(s):\n    return s[::-1]\n\n"
       "print(reverse(\"hello\"))  # olleh", PY),
    _e("py_fibonacci", "python", [
        "Write a Fibonacci function in Python.", "How do I compute Fibonacci numbers in Python?",
        "Python code for the Fibonacci sequence.", "Show me a Fibonacci function.",
    ], "Here is an iterative version:\n\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
       "        a, b = b, a + b\n    return a\n\nprint([fib(i) for i in range(8)])  # [0, 1, 1, 2, 3, 5, 8, 13]", PY),
    _e("py_factorial", "python", [
        "Write a factorial function in Python.", "How do I compute a factorial in Python?",
        "Python code for factorial.", "Show me a factorial function.",
    ], "Here is a simple loop:\n\ndef factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n"
       "        result *= i\n    return result\n\nprint(factorial(5))  # 120", PY),
    _e("py_prime", "python", [
        "How do I check if a number is prime in Python?", "Write a Python prime check.",
        "Python function to test for a prime number.", "Is there a simple way to check primes in Python?",
    ], "Check divisors up to the square root:\n\ndef is_prime(n):\n    if n < 2:\n        return False\n"
       "    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n"
       "    return True\n\nprint(is_prime(17))  # True", PY),
    _e("py_binary_search", "python", [
        "Write binary search in Python.", "How does binary search work in Python?",
        "Python code for binary search.", "Show me a binary search function.",
    ], "Binary search works on a sorted list:\n\ndef binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n"
       "    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
       "        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
       "    return -1", PY),
    _e("py_palindrome", "python", [
        "How do I check for a palindrome in Python?", "Write a palindrome check in Python.",
        "Python function to test if a word is a palindrome.", "Show me palindrome code.",
    ], "Compare the cleaned string with its reverse:\n\ndef is_palindrome(s):\n"
       "    t = \"\".join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n\n"
       "print(is_palindrome(\"Race car\"))  # True", PY),
    # ------------------------------------------------------------------ stories
    _e("story_bedtime", "creative", [
        "Tell me a bedtime story.", "Can you tell me a story before bed?", "I need a short bedtime story.",
        "Write a sleepy bedtime story.",
    ], "Once upon a time, a little owl named Pip could not fall asleep. Pip counted the stars, one by "
       "one, until the moon smiled down. \"Rest now,\" said the moon. Pip tucked his head under his "
       "soft wing, listened to the quiet wind, and drifted into a happy dream. The end.", STORY),
    _e("story_mars", "creative", [
        "Tell me a story about Mars.", "Write a short story about a trip to Mars.", "Can you tell a Mars story?",
        "Tell me a space story about the red planet.",
    ], "Mia and her robot Bolt flew their small ship to Mars. The ground was red and dusty, and the sky "
       "was pink. Bolt found a shiny rock and gave it to Mia. They planted a tiny flag, waved at "
       "Earth far away, and flew home to tell everyone what they saw.", STORY),
    _e("story_dragon", "creative", [
        "Tell me a story about a friendly dragon.", "Write a story with a kind dragon.", "Can you tell a dragon story?",
        "Tell me a short dragon story.",
    ], "In a green valley lived a dragon named Ember who was afraid of her own fire. One cold night the "
       "village lamps went out. Ember took a deep breath and lit them all with a small, warm flame. "
       "The children cheered, and Ember was never afraid of her fire again.", STORY),
    _e("story_ocean", "creative", [
        "Tell me a story about the ocean.", "Write a story about a little fish.", "Can you tell a sea story?",
        "Tell me a short story under the sea.",
    ], "A little fish named Finn wanted to see the top of the sea. He swam up past the coral and the "
       "turtles until he saw the bright sun. A seagull said hello. Finn waved his fin, then swam "
       "back down to tell his mother about the sky.", STORY),
    _e("story_friendship", "creative", [
        "Tell me a story about friendship.", "Write a story about two friends.", "Can you tell a story about sharing?",
        "Tell me a story about being kind.",
    ], "Sam had a red ball and Lily had a blue kite. At first they did not want to share. Then the wind "
       "blew hard and the kite got stuck in a tree. Sam threw his ball and knocked it free. They "
       "laughed and played together all afternoon.", STORY),
    _e("story_rain", "creative", [
        "Tell me a story about a rainy day.", "Write a story about rain.", "Can you tell a rainy day story?",
        "Tell me a story about puddles.",
    ], "It rained all morning, and Ben felt sad. His grandma gave him yellow boots. Ben jumped in the "
       "biggest puddle and made a huge splash. A frog jumped too. Ben laughed and decided that rainy "
       "days could be fun after all.", STORY),
    # --------------------------------------------------------------------- math
    _e("math_apples", "math", [
        "If I have 3 apples and buy 5 more, how many apples do I have?",
        "I have three apples and get five more. How many now?",
        "3 apples plus 5 apples is how many apples?", "How many apples if I start with 3 and add 5?",
    ], "Start with 3 apples. Add 5 more: 3 + 5 = 8. You have 8 apples.", MATH),
    _e("math_linear", "math", [
        "Solve 2x + 3 = 11.", "What is x if 2x + 3 = 11?", "Find x in 2x + 3 = 11.", "How do I solve 2x plus 3 equals 11?",
    ], "Subtract 3 from both sides: 2x = 8. Divide both sides by 2: x = 4. Check: 2 * 4 + 3 = 11.", MATH),
    _e("math_speed", "math", [
        "A car travels 120 km in 2 hours. What is its average speed?",
        "What is the speed if you drive 120 km in 2 hours?",
        "120 kilometers in 2 hours is how fast?", "How fast is a car that goes 120 km in two hours?",
    ], "Average speed is distance divided by time: 120 km / 2 h = 60 km per hour.", MATH),
    _e("math_percent", "math", [
        "What is 20 percent of 50?", "Compute 20% of 50.", "How much is twenty percent of fifty?",
        "Find 20 percent of 50.",
    ], "20 percent means 20 / 100 = 0.2. Then 0.2 * 50 = 10. So 20 percent of 50 is 10.", MATH),
    _e("math_area", "math", [
        "What is the area of a rectangle 4 by 6?", "Find the area of a 4 by 6 rectangle.",
        "A rectangle is 4 wide and 6 long. What is its area?", "How big is a 4 by 6 rectangle?",
    ], "The area of a rectangle is width times length: 4 * 6 = 24. The area is 24 square units.", MATH),
    _e("math_share", "math", [
        "If 12 cookies are shared by 4 friends, how many does each get?",
        "Share 12 cookies equally among 4 friends.", "12 cookies and 4 friends, how many each?",
        "How many cookies does each of 4 friends get from 12?",
    ], "Divide the cookies by the number of friends: 12 / 4 = 3. Each friend gets 3 cookies.", MATH),
    # --------------------------------------------------------------- small talk
    _e("talk_hello", "small_talk", [
        "Hello!", "Hi there.", "Hey, how are you?", "Good morning!",
    ], "Hello! I am NanoLlama. How can I help you today?", GENERAL),
    _e("talk_joke", "small_talk", [
        "Tell me a joke.", "Do you know a joke?", "Make me laugh.", "Say something funny.",
    ], "Why did the neural network go to school? Because it wanted to improve its weights.", GENERAL),
    _e("talk_fact", "small_talk", [
        "Tell me a fun fact.", "Do you know an interesting fact?", "Share a fun fact with me.",
        "Tell me something interesting.",
    ], "Octopuses have three hearts. Two pump blood to the gills, and one pumps it to the rest of the body.", GENERAL),
    _e("talk_thanks", "small_talk", [
        "Thank you!", "Thanks for your help.", "Thanks a lot.", "I appreciate it.",
    ], "You are welcome! I am happy to help.", GENERAL),
    _e("talk_bye", "small_talk", [
        "Goodbye!", "Bye for now.", "See you later.", "I have to go now.",
    ], "Goodbye! Have a nice day.", GENERAL),
    _e("talk_feeling", "small_talk", [
        "How are you feeling today?", "Are you happy?", "How is your day going?", "Do you have feelings?",
    ], "I am a small computer program, so I do not have feelings. But I am ready to help you!", GENERAL),
]

SPLIT_SEED = 2025
KB_ROLES = ("train", "val", "test")


def split_entries(entries: list[Entry] = ENTRIES, seed: int = SPLIT_SEED) -> dict[str, list[Entry]]:
    """Seeded split stratified by topic: per topic 1 val, 1 test, rest train."""
    rng = np.random.default_rng(seed)
    out = {r: [] for r in KB_ROLES}
    topics = sorted({e.topic for e in entries})
    for topic in topics:
        group = sorted((e for e in entries if e.topic == topic), key=lambda e: e.key)
        order = rng.permutation(len(group))
        for rank, idx in enumerate(order):
            role = "val" if rank == 0 else "test" if rank == 1 else "train"
            out[role].append(group[idx])
    return out


def _dialogue(entry: Entry, query: str, persona: str, qi: int, role: str) -> Dialogue:
    return Dialogue(
        id=f"kb:{entry.key}:q{qi}:{persona}",
        source="kb",
        system=PERSONAS[persona],
        turns=[("user", query), ("assistant", entry.answer)],
        meta={"entry": entry.key, "topic": entry.topic, "persona": persona, "kb_role": role},
    )


def expand(entry: Entry, role: str, query_indices: list[int] | None = None) -> list[Dialogue]:
    """(query x in-scope persona) + one default-persona dialogue per query, deduplicated."""
    qs = range(len(entry.queries)) if query_indices is None else query_indices
    out, seen = [], set()
    for qi in qs:
        personas = list(entry.personas)
        if DEFAULT_PERSONA not in personas:
            personas.append(DEFAULT_PERSONA)
        for p in personas:
            if (qi, p) in seen:
                continue
            seen.add((qi, p))
            out.append(_dialogue(entry, entry.queries[qi], p, qi, role))
    return out


def build_kb_splits(seed: int = SPLIT_SEED) -> dict[str, list[Dialogue]]:
    """Return dialogues for train / val / test / rephrase (diagnostic).

    - train: every train entry, all paraphrases except the withheld one.
    - val / test: every paraphrase of val / test entries (answers unseen in training).
    - rephrase: the withheld paraphrase of each train entry, default persona only.
    """
    rng = np.random.default_rng(seed + 1)
    parts = split_entries(seed=seed)
    out: dict[str, list[Dialogue]] = {"train": [], "val": [], "test": [], "rephrase": []}
    for entry in parts["train"]:
        withheld = int(rng.integers(len(entry.queries)))
        keep = [i for i in range(len(entry.queries)) if i != withheld]
        out["train"] += expand(entry, "train", keep)
        out["rephrase"].append(_dialogue(entry, entry.queries[withheld], DEFAULT_PERSONA, withheld, "rephrase"))
    for role in ("val", "test"):
        for entry in parts[role]:
            out[role] += expand(entry, role)
    # Guard: a train query that normalizes to a held-out query is removed from training.
    held_q = {norm_key(d.turns[0][1]) for r in ("val", "test", "rephrase") for d in out[r]}
    held_a = {norm_key(d.turns[1][1]) for r in ("val", "test") for d in out[r]}
    out["train"] = [d for d in out["train"] if norm_key(d.turns[0][1]) not in held_q and norm_key(d.turns[1][1]) not in held_a]
    return out


PRESET_CATEGORIES = ("Identity", "AI Research", "Coding", "Creative Story", "Logic & Math")


def presets() -> list[dict]:
    """Chat presets (S01): persona + message applied in one click.

    KB presets use a query from a *training* entry (never validation/test), so the
    UI does not invite peeking at held-out items. Each preset says where its
    message comes from. Replies are always generated by the model.
    """
    train = split_entries()["train"]

    def first(topics):
        return sorted((e for e in train if e.topic in topics), key=lambda e: e.key)[0]

    out = []
    for category, topics, persona in (
        ("Identity", ("identity",), "assistant"),
        ("AI Research", ("ml_foundations", "algorithms", "metrics"), "ml_research"),
        ("Coding", ("python",), "python"),
        ("Logic & Math", ("math",), "math_tutor"),
    ):
        e = first(topics)
        out.append({"title": e.queries[0], "category": category, "persona": persona,
                    "system": PERSONAS[persona], "message": e.queries[0],
                    "origin": "Knowledge-base training query"})
    out.insert(1, {"title": "Everyday chat", "category": "Identity", "persona": "assistant",
                   "system": PERSONAS["assistant"], "message": "Hi! Can you help me plan a picnic?",
                   "origin": "Not in any dataset split (free prompt)"})
    out.insert(4, {"title": "Bedtime story", "category": "Creative Story", "persona": "storyteller",
                   "system": PERSONAS["storyteller"],
                   "message": "Write a short story for young children.\nUse these words: moon, blanket, brave.\nStory features: Dialogue.",
                   "origin": "TinyStories-style instruction written for this app"})
    return out
