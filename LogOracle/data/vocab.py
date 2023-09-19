class CharVocab():
    def __init__(self):
        self._id2char = [
            '<pad>', '<unk>', '<space>', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '@',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', ';', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z',
            '{', '|', '}', '~'
        ]
        self._char2id = {char: idx for idx, char in enumerate(self._id2char)}

    def id2char(self, id):
        if id < len(self._id2char):
            return self._id2char[id]
        else:
            return '<unk>'

    def char2id(self, char):
        if char in self._char2id.keys():
            return self._char2id[char]
        else:
            return 1

    @property
    def vocab_size(self):
        return len(self._id2char)