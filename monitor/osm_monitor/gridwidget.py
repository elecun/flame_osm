""" Grid Graphic Window Module """

try:
    from PyQt5.QtWidgets import QApplication, QFrame, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QVBoxLayout
    from PyQt5.QtGui import QBrush, QColor
    from PyQt5.QtCore import QRectF
except ImportError:
    from PyQt6.QtWidgets import QApplication, QFrame, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QVBoxLayout
    from PyQt6.QtGui import QBrush, QColor
    from PyQt6.QtCore import QRectF


class GridGraphicFrame(QFrame):
    def __init__(self, rows, cols, parent=None):
        super().__init__(parent)

        # create grphic scene
        self.__scene = QGraphicsScene()
        self.__create_grid(rows, cols, 50, 50)

        self.__graphicview = QGraphicsView(self.__scene, self)
        self.__graphicview.setDragMode(QGraphicsView.DragMode.ScrollHandDrag) # dragable

        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.__graphicview)
        self.setLayout(layout)

    def __create_grid(self, rows, cols, cell_width=50, cell_height=50):
        """create grid cell block """

        for row in range(rows):
            for col in range(cols):
                rect = QRectF(
                    col * cell_width, # X
                    row * cell_height, # Y
                    cell_width, # width
                    cell_height # height
                )
                item = QGraphicsRectItem(rect)

                # set color
                #color = QColor(255, col * 10 % 255, row * 25 % 255)
                #item.setBrush(QBrush(color))
                item.setPen(QColor(200, 200, 200))  # 셀 테두리

                self.__scene.addItem(item)

    