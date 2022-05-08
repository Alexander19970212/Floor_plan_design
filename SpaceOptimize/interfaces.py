from typing import Protocol, List
from abc import abstractmethod
from dataclasses import dataclass, asdict

"""
The classes of objects with optimazited parameters won't be created as constant class. Users could create any new class 
with parameters what they wanted. But that classes will inherit methods from one special class with methods for updating 
and sending parameters.

The main idea of Group_object is that it consist of the classes with the same set of parameters. That class should be 
able to create sets of object, added new one, and remove it. Additionally it should be able send parameters, update 
them and send the names of parameters. Updating happens by protocol which is described (TEMPORARY) in class object.
For example, user creates object "desk" and it has parameters coord, illumination....

Writen above is impossible (I think). 

Space_of_objects contains all classes of objects needed for optimization, including constant object. Exactly in here 
cooperation between objects happens. User writes special guide for updating parameters. It is very important sequences 
how parameters will be extracted and how will be updated. And without user should be prepared with matrix or without 
parameters will be updated, it's about calculation. 
"""

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