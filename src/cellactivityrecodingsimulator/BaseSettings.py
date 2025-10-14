
class BaseSettings():
    def __init__(self, data: dict=None):
        self.data = data

    def to_dict(self):
        return self.data

    def validate(self):
        if isinstance(self.data, dict):
            return True
        else:
            return False
