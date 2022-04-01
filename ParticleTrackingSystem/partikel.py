class Partikel:
    def __init__(self, p_id, x_pos, y_pos, is_moving):
        self.__p_id = p_id
        self._x_pos = x_pos
        self._y_pos = y_pos
        self.is_moving = is_moving

    def get_p_id(self):
        return self.__p_id

    def set_p_id(self, p_id):
        self.__p_id = p_id

    def get_x_pos(self):
        return self._x_pos

    def get_y_pos(self):
        return self._y_pos

    def set_x_pos(self, new_x_pos):
        self._x_pos = new_x_pos

    def set_y_pos(self, new_y_pos):
        self._y_pos = new_y_pos

