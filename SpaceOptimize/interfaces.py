from typing import Protocol, List
from abc import abstractmethod
from dataclasses import dataclass, asdict

class OptimizedObject(Protocol):  # Проработать API объекта
    @abstractmethod
    def get_coords(self, ):
        pass


class GroupObject(Protocol):  # расчет лосса для группы, так и между группами
    """
        Все интересующие параметры будут извлекаться из групп
        Произвести операции с коордианатами объектов, принадлежащих группе
        Габариты объектов - матрица габаритов
        Освещенность - вектор для группы, для объекта число
    """

    pass


class Builder(Protocol):
    pass