import torch
import io as sys_io
from utils import io_ as io
from utils import str_ as str
from utils import rbox_util
from utils import ocr_util
from utils import locality_aware_nms


def np_to_variable(x, dtype=torch.FloatTensor):
    v = torch.from_numpy(x).type(dtype)
    v = v.cuda()
    return v
