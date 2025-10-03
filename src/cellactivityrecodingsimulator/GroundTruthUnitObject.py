from .Cell import Cell
import numpy as np



class GTUnitObject:

    def __init__(
        self, 
        num_column:int=1, 
        num_unit_per_column:list[int]=[16],
        xpitch:float=0,
        ypitch:float=25.0,
        y_shift_per_column:list[float]=[0],
        id_major_order:str="column",
        ):
        self._num_column = num_column
        self._num_unit_per_column = num_unit_per_column
        self._xpitch = xpitch
        self._ypitch = ypitch
        self._y_shift_per_column = y_shift_per_column
        self._id_major_order = id_major_order

        self._cells = self.generate()

    def __str__(self):
        return f"GTUnitObject(num_column={self._num_column}, num_unit_per_column={self._num_unit_per_column}, xpitch={self._xpitch}, ypitch={self._ypitch}, y_shift_per_column={self._y_shift_per_column})"

    def __repr__(self):
        return self.__str__()

    @property
    def num_column(self):
        return self._num_column
    
    @num_column.setter
    def num_column(self, value):
        self._num_column = value

    @property
    def num_unit_per_column(self):
        return self._num_unit_per_column

    @num_unit_per_column.setter
    def num_unit_per_column(self, value):
        self._num_unit_per_column = value

    @property
    def xpitch(self):
        return self._xpitch
    
    @xpitch.setter
    def xpitch(self, value):
        self._xpitch = value

    @property
    def ypitch(self):
        return self._ypitch
    
    @ypitch.setter
    def ypitch(self, value):
        self._ypitch = value

    @property
    def y_shift_per_column(self):
        return self._y_shift_per_column

    @y_shift_per_column.setter
    def y_shift_per_column(self, value):
        self._y_shift_per_column = value

    @property
    def id_major_order(self):
        return self._id_major_order

    @id_major_order.setter
    def id_major_order(self, value):
        self._id_major_order = value

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, value):
        self._cells = value

    def generate(self):
        cells = []
        current_id = 0
        
        if self._id_major_order == "column":
            # 列優先: 各列を順番に処理
            for column in range(self._num_column):
                for unit in range(self._num_unit_per_column[column]):
                    cells.append(
                        Cell(
                            id=current_id, 
                            group=0, 
                            x=self._xpitch * unit, 
                            y=self._ypitch * column,
                            z=0
                        ))
                    current_id += 1
        elif self._id_major_order == "row":
            # 行優先: 各行を順番に処理
            max_units = max(self._num_unit_per_column) if self._num_unit_per_column else 0
            for unit in range(max_units):
                for column in range(self._num_column):
                    if unit < self._num_unit_per_column[column]:
                        cells.append(
                            Cell(
                                id=current_id, 
                                group=0, 
                                x=self._xpitch * unit, 
                                y=self._ypitch * column,
                                z=0
                            ))
                        current_id += 1
        else:
            raise ValueError(f"Invalid id_major_order: {self._id_major_order}")

        return cells