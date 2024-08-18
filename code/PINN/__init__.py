
from .pinn_base import pin_base_exe
from .pinn_modified import pin_modified_exe
from .pinn_new_arc import pin_new_arc_exe
from .pinn_new_arc_MLP import pin_new_arc_MLP_exe
from .pinn_MLP import pin_MLP_exe


PINNS = [
    {'model': pin_base_exe, 'name': 'PINN base'},
    #{'model': pin_modified_exe, 'name': 'PINN modified'},
    {'model': pin_new_arc_exe, 'name': 'PINN new architecture'},
    {'model': pin_new_arc_MLP_exe, 'name': 'PINN new architecture MLP'},
    {'model': pin_MLP_exe, 'name': 'PINN MLP'},
]