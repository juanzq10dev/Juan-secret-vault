from app.domain.entities.moves import Move
from typing import List

class Pokemon:
    def __init__(self, name: str, hp: int, moves: List[Move]):
        self.name = name
        self.hp = hp
        self.moves = moves
    
    def take_damage(self, amount):
        self.hp -= amount

    def is_fainted(self):
        return self.hp <= 0
