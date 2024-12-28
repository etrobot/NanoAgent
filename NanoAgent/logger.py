class DebugLogger:
    def __init__(self, debug=False):
        self.debug = debug
        self.colors = {
            'user': '\033[93m',
            'assistant': '\033[94m',
            'action': '\033[95m',
            'reason': '\033[96m',
            'next_prompt': '\033[92m',
            'error': '\033[91m',
            'reset': '\033[0m'
        }

    def log(self, category, message, end='\n', flush=True):
        if not self.debug:
            return
        if category in self.colors:
            color = self.colors[category]
            label = category.title() + ':'
            if isinstance(message, Exception):
                message = f"{type(message).__name__}: {str(message)}"
            print(f"{color}{label}\033[0m {message}", end=end, flush=flush)

    def print(self, message, end='', flush=True):
        if not self.debug:
            return
        print(message, end=end, flush=flush) 