# NO ADDITIONAL IMPORTS!
from text_tokenize import tokenize_sentences


class Trie:
    def __init__(self):
        self.value = None
        self.children = {}
        self.type = None

    def __setitem__(self, key, value):
        """
        Add a key with the given value to the trie, or reassign the associated
        value if it is already present in the trie.  Assume that key is an
        immutable ordered sequence.  Raise a TypeError if the given key is of
        the wrong type.
        """
        if self.type is None:
            self.type = type(key)
        elif not isinstance(key, self.type):
            raise TypeError
        if key[:1] not in self.children:
            new_trie = Trie()
            self.children[key[:1]] = new_trie
        if len(key) > 1:
            self.children[key[:1]][key[1:]] = value
        else:
            self.children[key[:1]].value = value

    def __getitem__(self, key):
        """
        Return the value for the specified prefix.  If the given key is not in
        the trie, raise a KeyError.  If the given key is of the wrong type,
        raise a TypeError.
        """
        if not isinstance(key, self.type):
            raise TypeError
        if key[:1] not in self.children:
            raise KeyError
        if len(key) > 1:
            return self.children[key[:1]][key[1:]]
        return self.children[key[:1]].value

    def __delitem__(self, key):
        """
        Delete the given key from the trie if it exists. If the given key is not in
        the trie, raise a KeyError.  If the given key is of the wrong type,
        raise a TypeError.
        """
        if not isinstance(key, self.type):
            raise TypeError
        if key not in self:
            raise KeyError
        if len(key) > 1:
            # Delete nodes lower in the tree first, THEN check if those deletions result in empty branches (which
            # can then be deleted by other deletion calls. This method essentially combines the steps of checking
            # for length-1 dictionaries and deleting nodes.
            child = self.children[key[:1]]
            del child[key[1:]]  # Delete child of child
        elif len(key) == 1:
            if len(self.children) == 0:
                # If we've reached the desired trie, and it has no children, we delete the entire node
                del self.children[key[:1]]
            else:
                # If we've reached the desired trie, but it does have children, we just delete its value
                self.children[key[:1]].value = None

    def __contains__(self, key):
        """
        Is key a key in the trie? return True or False.
        """
        if len(key) == 0:
            return False
        if key[:1] in self.children:
            child_at_key = self.children[key[:1]]
            if len(key) > 1:
                return key[1:] in child_at_key
            return child_at_key.value is not None
        return False

    def __iter__(self):
        """
        Generator of (key, value) pairs for all keys/values in this trie and
        its children.  Must be a generator!
        """
        if len(self.children) > 0:
            for i in self.children:
                # Iterate through self's child tries, if they exist
                child = self.children[i]
                if child.value is not None:
                    # If this child has a value, yield the corresponding (key, value) pair
                    yield i, child.value
                for key, val in iter(child):
                    # Then, iterate through all of this child's (key, value) pairs, adding the child's key
                    # to the front of the yielded key to create the full key corresponding to that value in
                    # the overall trie.
                    yield i + key, val


def make_word_trie(text):
    """
    Given a piece of text as a single string, create a Trie whose keys are the
    words in the text, and whose values are the number of times the associated
    word appears in the text
    """

    text = tokenize_sentences(text)
    counter = text_data(text)['word_count']  # Build a dictionary counting the number of times each word appears

    trie = Trie()
    for word in counter:
        # Build trie
        trie[word] = counter[word]
    return trie


def make_phrase_trie(text):
    """
    Given a piece of text as a single string, create a Trie whose keys are the
    sentences in the text (as tuples of individual words) and whose values are
    the number of times the associated sentence appears in the text.
    """

    text = tokenize_sentences(text)
    # Build a dictionary counting the number of times each sentence appears and a list representation of each
    # sentence where each element is a word:
    data = text_data(text)
    counter = data['sentence_count']
    sentences = data['sentence_lists']

    trie = Trie()
    for sentence in sentences:
        # Build trie
        trie[sentence] = counter[sentence]
    return trie


def text_data(sentences):
    word_count = {}
    sentence_count = {}
    sentence_tuples = []
    for sentence in sentences:
        word = ''
        list_rep = []  # A list representation of this sentence
        for char in sentence:
            if char == ' ':
                # Spaces mark the bounds of words
                if not word == '':
                    # Step up counter in dictionary by one
                    word_count[word] = word_count.get(word, 0) + 1
                    list_rep.append(word)
                    word = ''

            else:
                word += char
        word_count[word] = word_count.get(word, 0) + 1  # Add count of last word
        list_rep.append(word)  # Add last word to sentence list
        tup_rep = tuple(list_rep)

        sentence_tuples.append(tup_rep)
        # Add count of sentence:
        sentence_count[tup_rep] = sentence_count.get(tup_rep, 0) + 1
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'sentence_lists': sentence_tuples
    }


def autocomplete(trie, prefix, max_count=None):
    """
    Return the list of the most-frequently occurring elements that start with
    the given prefix.  Include only the top max_count elements if max_count is
    specified, otherwise return all.

    Raise a TypeError if the given prefix is of an inappropriate type for the
    trie.
    """
    if not isinstance(prefix, trie.type):
        raise TypeError

    if not (prefix == '' or prefix == ()):
        # Check if prefix is contained in trie
        try:
            trie[prefix]
        except KeyError:
            return []

    potentials = []  # List of potential completions and their number of occurrences
    if not (prefix == '' or prefix == ()) and trie[prefix] is not None:
        potentials.append((prefix, trie[prefix]))

    key = prefix
    while not (key == '' or key == ()):
        # Get prefix's trie
        trie = trie.children[key[:1]]
        key = key[1:]
    for suffix, count in iter(trie):
        # Build prefix's derivations from the children of prefix's trie
        if count is not None:
            derivation = (prefix + suffix, count)
            potentials.append(derivation)

    # Sort potential completions from highest to lowest number of occurrences
    potentials = sorted(potentials, key=lambda x: -x[1])
    if max_count is None:
        max_count = len(potentials)
    return [i[0] for i in potentials[:max_count]]


def autocorrect(trie, prefix, max_count=None):
    """
    Return the list of the most-frequent words that start with prefix or that
    are valid words that differ from prefix by a small edit.  Include up to
    max_count elements from the autocompletion.  If autocompletion produces
    fewer than max_count elements, include the most-frequently-occurring valid
    edits of the given word as well, up to max_count total elements.
    """

    completions = []
    for comp in autocomplete(trie, prefix, max_count):
        # Add completions to potential corrections
        completions.append((comp, trie[comp]))

    if max_count is not None:
        # If we have enough completions, we can just return them without worrying about generating corrections
        if len(completions) >= max_count:
            return [i[0] for i in completions[:max_count]]
        else:
            max_count -= len(completions)

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # CREATE INSERTION EDITS:
    insertions = []
    for char in alphabet:
        for split in range(len(prefix) + 1):
            # split is the index before which the new character is inserted
            if split < len(prefix):
                if prefix[split] == char:
                    # If the character directly after the position at which char would be inserted is the same
                    # as char, we don't make this edit. This it to avoid situations where multiple identical edits
                    # are made, i.e. ba + a + r = b + a + ar = baar, so we only create an edit in the former case.
                    continue
            edit = prefix[:split] + char + prefix[split:]
            if edit in trie:
                insertions.append((edit, trie[edit]))

    # CREATE DELETION EDITS:
    deletions = []
    if len(prefix) > 1:
        for remove in range(len(prefix)):
            if remove < len(prefix) - 1:
                # remove is the index of the removed character
                if prefix[remove] == prefix[remove + 1]:
                    # For similar reasons as in the insertion edit creation loop, we use an if-statement here
                    # to avoid creating duplicate edits. If there are multiple consecutive characters in a word,
                    # removing any one of them will create the same edit. Therefore, we only remove the last
                    # character in a consecutive sequence.
                    continue
            edit = prefix[:remove] + prefix[remove + 1:]
            if edit in trie:
                deletions.append((edit, trie[edit]))

    # CREATE REPLACEMENT EDITS:
    replacements = []
    if len(prefix) > 0:
        for char in alphabet:
            for split in range(len(prefix)):
                # split is the index that's replaced by the new character
                if prefix[split] == char:
                    # Don't waste time replacing a character with itself
                    continue
                edit = prefix[:split] + char + prefix[split + 1:]
                if edit in trie:
                    replacements.append((edit, trie[edit]))

    # CREATE TRANSPOSE EDITS:
    transposes = []
    if len(prefix) > 1:
        for left in range(len(prefix)):
            for right in range(left + 1, len(prefix)):
                # left and right are the indices of the characters that we're swapping
                if prefix[left] == prefix[right]:
                    # Don't waste time swapping a character with the same character.
                    continue
                edit = prefix[:left] + prefix[right] + prefix[left + 1:right] + prefix[left] + prefix[right + 1:]
                if edit in trie:
                    replacements.append((edit, trie[edit]))

    # List of potential corrections and their number of occurrences:
    potentials = insertions + deletions + replacements + transposes
    # Sort potential corrections from highest to lowest number of occurrences
    potentials = sorted(potentials, key=lambda x: -x[1])
    if max_count is None:
        max_count = len(potentials)

    corrections = []
    for i in potentials[:max_count]:
        # Remove duplicates
        if i not in completions:
            corrections.append(i)

    return [i[0] for i in completions + corrections]


def word_filter(trie, pattern):
    """
    Return list of (word, freq) for all words in trie that match pattern.
    pattern is a string, interpreted as explained below:
         * matches any sequence of zero or more characters,
         ? matches any single character,
         otherwise char in pattern char must equal char in word.
    """
    i = 0
    while i < len(pattern) - 1:
        # Remove consecutive *'s
        if pattern[i] == pattern[i + 1] == '*':
            pattern = pattern[0:i] + pattern[i + 1:]
        else:
            i += 1

    def matches(word, pat):
        if pat == '*':
            # This pattern matches every possible word
            return True
        if len(word) == 0:
            return False

        len_p = 0  # length of pattern discounting *'s
        for i in pat:
            if not i == '*':
                len_p += 1

        if pat[0] == '*':
            for i in range(len(word) - len_p + 2):
                # If we get a match by matching some number of characters at the beginning of word to *, return True
                if matches(word[i:], pat[1:]):
                    return True
            return False  # There exists no match for word, no matter how many or few characters * replaces
        elif pat[0] == '?':
            if len(pat) > 1:
                # As long as we have more pattern to match, we run match() again
                return matches(word[1:], pat[1:])
            if len(word) == len(pat) == 1:
                # If both strings are length one and they match, we're done
                return True
            return False
        else:
            if word[0] == pat[0]:
                if len(pat) > 1:
                    # As long as we have more pattern to match, we run match() again
                    return matches(word[1:], pat[1:])
                if len(word) == len(pat) == 1:
                    # If both strings are length one and they match, we're done
                    return True
                return False
            else:
                return False

    matching = []
    for word, count in iter(trie):
        if matches(word, pattern):
            matching.append((word, count))
    print(pattern)
    print(matching)
    return matching


# you can include test cases of your own in the block below.
if __name__ == '__main__':
    with open("Pride.txt", encoding="utf-8") as f:
        pride = f.read()
    with open("Alice.txt", encoding="utf-8") as f:
        alice = f.read()
    with open("Dracula.txt", encoding="utf-8") as f:
        dracula = f.read()
    with open("TwoCities.txt", encoding="utf-8") as f:
        cities = f.read()
    with open("Meta.txt", encoding="utf-8") as f:
        meta = f.read()

    print(match('mat', 'mat*'))

