
class Builder:
    def __init__(self, area, space_objects, instruction):
        """
        :param area: Object with type Area
        :param space_objects: Object with type Space_object
        :param instruction: function for convert vectors
        """
        self.area = area
        self.space_objects = space_objects
        self.convert_function = instruction
        self.current_vectors = None

    def convert_gens(self, gens):
        """
        list with torch tensors with params which is necessary for each object in space_objects.
        :param gens:
        :return:
        """
        self.current_vectors = self.convert_function(gens)

        # Не плохо бы сделать метки изменений, так бы получилось сэкономить время не обновляя неизмененные

    def temporary_step(self, gens):
        """
        Function get order to objects to create (update) copy of parameters
        :param gens: torch tensor with float 0-1 values and size (batch size, N)
        :return:
        """
        self.convert_gens(gens)

        # self.area.update_temporary(self)

        for i, obj in self.space_objects.iterator():
            obj.update_temp(self.current_vectors[:, i])  # подгружаются координаты

        self.area.update_temp(self.space_objects)  # Это немного противоречит концепции, но может помочь съэкономить на расчете освещенности

        self.space_objects.reload_parametеrs_temp(self.area)  # обновляем параметры типа освещенность
        self.space_objects.reload_acquired_objects_temp()  # Обновляем объекты типа пути

    def optimize_step(self, gens):
        """
        Function get order to objects to create (update) copy of parameters
        :param gens: torch tensor with float 0-1 values and size (batch size, N)
        :return:
        """
        self.convert_gens(gens)

        # self.area.update_temporary(self)

        for i, obj in self.space_objects.iterator():
            obj.update_perm(self.current_vectors[:, i])  # подгружаются координаты

        self.area.update_perm(
            self.space_objects)  # Это немного противоречит концепции, но может помочь съэкономить на расчете освещенности

        self.space_objects.reload_parametеrs_perm(self.area)  # обновляем параметры типа освещенность
        self.space_objects.reload_acquired_objects_perm()  # Обновляем объекты типа пути





    