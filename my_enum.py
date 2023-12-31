from enum import Enum, auto

class Feature(Enum):
    CLOSING = auto()
    VOLUME = auto()
    HASHRATE = auto()
    TRANSACTIONCOUNT = auto()
    
class ModelType(Enum):
    LSTM = auto()
    BILSTM = auto()
    GRU = auto()
    
    
