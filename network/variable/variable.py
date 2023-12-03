from abc import ABC, abstractmethod



class Variable(ABC):
    """
    Abstract class for variables (layers and parameters).

    Attributes
    ----------
    shape (tuple of int): shape of the tensor used to represent the state of the variable
    state (Tensor): the state of the variable

    Methods
    -------
    init_state()
        Initializes the state of the variable
    """

    def __init__(self, shape):
        """Initializes an instance of Variable

        Args:
            shape (tuple of int): shape of the tensor used to represent the variable's state
        """

        self._shape = shape

    @property
    def shape(self):
        """Gets the shape of the variable (the tensor)"""

        return self._shape

    @property
    def state(self):
        """Gets and sets the current state of the variable"""

        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def set_device(self, device):
        """Set the variable's Tensor state on a given device

        Args:
            device (str): Either 'cpu' or 'cuda'.
        """

        self._state = self._state.to(device)

    @abstractmethod
    def init_state(self):
        """Initializes the variable"""
        pass