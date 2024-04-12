class RelationshipType:
    def __init__(self, name: str, is_symmetric: bool = False):
        self.name = name
        self.is_symmetric = is_symmetric

    def __repr__(self):
        return f"{self.name}({'Symmetric' if self.is_symmetric else 'Asymmetric'})"
